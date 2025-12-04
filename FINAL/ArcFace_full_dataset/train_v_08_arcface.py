import os
os.environ["HF_HOME"] = "/home/smukher5/.cache/huggingface/"

import random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torchvision import models, transforms
from datasets import load_dataset

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment


# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 512
EMBED_DIM = 1024
NUM_EPOCHS = 3
LEARNING_RATE = 1e-3      # ArcFace prefers SGD+momentum or higher LR
WEIGHT_DECAY = 1e-4
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
dataset = load_dataset("chronopt-research/cropped-vggface2-224")

train_full = dataset["train"]
val_full   = dataset["validation"]

n_train = max(1, int(1.00 * len(train_full)))
n_val   = max(1, int(1.00 * len(val_full)))

# Shuffle only for training
train_subset = train_full.shuffle(seed=RANDOM_SEED).select(range(n_train))
val_subset   = val_full.shuffle(seed=RANDOM_SEED).select(range(n_val))

# train_subset = train_full.select(range(n_train))
# val_subset   = val_full.select(range(n_val))

print(f"Train size: {len(train_subset)}")
print(f"Val size:   {len(val_subset)}")

# ---------------------------------------------------------
# Label indexing (ArcFace needs compact class IDs)
# ---------------------------------------------------------
train_labels = np.array(train_subset["label"])
unique_labels = np.unique(train_labels)
label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}
num_classes = len(unique_labels)

print("ArcFace classes:", num_classes)


# ---------------------------------------------------------
# Transforms
# ---------------------------------------------------------
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])


# ---------------------------------------------------------
# Standard Image Dataset (NO PAIRS NEEDED)
# ---------------------------------------------------------
class ImageClassDataset(Dataset):
    def __init__(self, hf_dataset, transform, label_map):
        self.ds = hf_dataset
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = self.transform(sample["image"])
        original_label = sample["label"]
        label = self.label_map[original_label]
        return img, label


train_ds = ImageClassDataset(train_subset, transform, label_to_idx)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)


# ---------------------------------------------------------
# For validation: Keep original labels (for clustering)
# ---------------------------------------------------------
class SingleImageDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = self.transform(sample["image"])
        label = sample["label"]
        return img, label


val_ds = SingleImageDataset(val_subset, transform)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=8,  # subh: 4 works
    pin_memory=True,
)


# ---------------------------------------------------------
# ArcFace Head
# ---------------------------------------------------------
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, scale=30.0, margin=0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale
        self.margin = margin

    def forward(self, embeddings, labels):
        # Normalize features & weights
        emb_norm = F.normalize(embeddings)
        W_norm = F.normalize(self.weight)

        # Cosine similarity
        cos_theta = torch.matmul(emb_norm, W_norm.t())

        # Add margin
        theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))
        target_logits = torch.cos(theta + self.margin)

        one_hot = F.one_hot(labels, num_classes=W_norm.shape[0]).float().to(embeddings.device)

        # Replace target logit with margin-modified
        logits = cos_theta * (1 - one_hot) + target_logits * one_hot

        # Scale
        return logits * self.scale


# ---------------------------------------------------------
# Model Definition
# ---------------------------------------------------------
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = resnet50.fc.in_features

# Replace FC with embedding layer
resnet50.fc = nn.Linear(num_ftrs, EMBED_DIM)
resnet50 = resnet50.to(device)

arcface = ArcFaceHead(EMBED_DIM, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    list(resnet50.parameters()) + list(arcface.parameters()),
    lr=LEARNING_RATE, momentum=0.9, weight_decay=WEIGHT_DECAY
)


def forward_embeddings(x):
    z = resnet50(x)
    return F.normalize(z, p=2, dim=1)


# ---------------------------------------------------------
# Training Loop (ArcFace)
# ---------------------------------------------------------
resnet50.train()
arcface.train()

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for step, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        emb = forward_embeddings(imgs)
        logits = arcface(emb, labels)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{step+1}/{len(train_loader)}] Loss: {running_loss/50:.4f}")
            running_loss = 0.0

print("ArcFace Training finished.")

torch.save({
    "resnet50": resnet50.state_dict(),
    "arcface": arcface.state_dict()
}, "resnet50_arcface_vggface2.pth")


# ---------------------------------------------------------
# Validation Embedding Extraction
# ---------------------------------------------------------
resnet50.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        emb = forward_embeddings(imgs)
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

print("Validation embeddings shape:", all_embeddings.shape)


# ---------------------------------------------------------
# Clustering & Hungarian Accuracy
# ---------------------------------------------------------
unique_labels = np.unique(all_labels)
label_to_compact = {lab: i for i, lab in enumerate(unique_labels)}
y_true = np.array([label_to_compact[lab] for lab in all_labels])
n_classes = len(unique_labels)

kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=RANDOM_SEED)
y_pred = kmeans.fit_predict(all_embeddings)

cm = contingency_matrix(y_true, y_pred)
cost = cm.max() - cm
row_ind, col_ind = linear_sum_assignment(cost)
acc = cm[row_ind, col_ind].sum() / cm.sum()

print(f"Clustering accuracy: {acc * 100:.2f}%")


# ---------------------------------------------------------
# t-SNE Visualization
# ---------------------------------------------------------
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="random",
    random_state=RANDOM_SEED,
)
emb_2d = tsne.fit_transform(all_embeddings)

# Plot TRUE labels
plt.figure(figsize=(6,6))
for cls in np.unique(y_true):
    mask = (y_true == cls)
    plt.scatter(emb_2d[mask,0], emb_2d[mask,1], s=5, alpha=0.7, label=f"class {cls}")
plt.title("t-SNE True Labels")
plt.legend(markerscale=3, fontsize=6)
plt.tight_layout()
plt.savefig("tsne_true_labels_arcface.png", dpi=300)
plt.close()

# Plot KMeans clusters
plt.figure(figsize=(6,6))
for clus in np.unique(y_pred):
    mask = (y_pred == clus)
    plt.scatter(emb_2d[mask,0], emb_2d[mask,1], s=5, alpha=0.7, label=f"cluster {clus}")
plt.title("t-SNE KMeans Labels")
plt.legend(markerscale=3, fontsize=6)
plt.tight_layout()
plt.savefig("tsne_kmeans_labels_arcface.png", dpi=300)
plt.close()

print("Saved ArcFace plots.")
