import os
os.environ["HF_HOME"] = "/home/smukher5/.cache/huggingface/"

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import torch.nn.functional as F

import numpy as np
from sklearn.cluster import KMeans

# -------------------------------
# 0. Load dataset & split
# -------------------------------
dataset = load_dataset("chronopt-research/cropped-vggface2-224")
print(dataset)

train_dataset = dataset["train"]
train_subset = train_dataset.select(range(0, 9223))  # 0..9222

print(train_subset)
print(len(train_subset))

split = train_subset.train_test_split(
    test_size=0.2,
    stratify_by_column="label",
    seed=42
)

training_dataset = split["train"]
validation_dataset = split["test"]

print(training_dataset)
print(validation_dataset)

# -------------------------------
# 1. Transforms & loaders
# -------------------------------
image_size = 224
img_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet stats
        std=[0.229, 0.224, 0.225],
    ),
])

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------
# 2. Wrap HF dataset in a PyTorch Dataset
# -------------------------------
class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, label2idx=None):
        self.ds = hf_dataset
        self.transform = transform
        self.label2idx = label2idx

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]               # PIL Image
        label = item["label"]

        # Make sure labels are 0..num_classes-1 using a mapping
        if self.label2idx is not None:
            label = self.label2idx[label]

        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

# Build a consistent label mapping based on the training split
train_labels = sorted(set(training_dataset["label"]))
label2idx = {lab: i for i, lab in enumerate(train_labels)}
num_classes = len(train_labels)
print("Num classes:", num_classes)

train_torch_dataset = HFImageDataset(training_dataset, img_transform, label2idx)
val_torch_dataset   = HFImageDataset(validation_dataset, img_transform, label2idx)

batch_size = 64

train_loader = DataLoader(
    train_torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

val_loader = DataLoader(
    val_torch_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# -------------------------------
# 3. Model: ResNet-50 (pretrained) + new head
# -------------------------------
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Replace the final FC layer to match your number of identities
in_features = resnet50.fc.in_features
resnet50.fc = nn.Linear(in_features, num_classes)

# NOTE: whole network is unfrozen; no requires_grad changes.

resnet50 = resnet50.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, resnet50.parameters()),
    lr=1e-4,
    weight_decay=1e-4,
)

# -------------------------------
# 4. Training & validation loops
# -------------------------------
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# -------------------------------
# 5. Run training
# -------------------------------
num_epochs = 20
best_val_acc = 0.0

for epoch in range(1, num_epochs + 1):
    train_loss, train_acc = train_one_epoch(
        resnet50, train_loader, optimizer, criterion, device
    )
    val_loss, val_acc = evaluate(
        resnet50, val_loader, criterion, device
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc

    print(
        f"Epoch [{epoch}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
    )

print(f"Best validation accuracy: {best_val_acc:.4f}")
