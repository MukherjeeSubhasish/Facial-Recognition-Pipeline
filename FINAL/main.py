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

BATCH_SIZE = 384        # subh 128
EMBED_DIM = 1024        # subh 128 ... 1024 gave 7% more accuracy as compared to 128
NUM_EPOCHS = 5
POSITIVE_PROB = 0.5     # subh 0.5 ... 0.1 gave 7% lower accuracy
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------
# Load HF dataset and take first 1% subsets (NO SHUFFLE)
# ---------------------------------------------------------
dataset = load_dataset("chronopt-research/cropped-vggface2-224")  # train, validation

train_full = dataset["train"]
val_full = dataset["validation"]

n_train = max(1, int(1.0 * len(train_full)))
n_val = max(1, int(1.0 * len(val_full)))

# IMPORTANT: no shuffle here, just take first 1%
train_subset = train_full.shuffle(seed=RANDOM_SEED).select(range(n_train))
val_subset = val_full.shuffle(seed=RANDOM_SEED).select(range(n_val))

print(f"Train full size: {len(train_full)}, using first 1% -> {len(train_subset)}")
print(f"Val full size:   {len(val_full)}, using first 1% -> {len(val_subset)}")

# ---------------------------------------------------------
# unique label check
# ---------------------------------------------------------
val_labels = np.array(val_subset["label"])
unique_labels = np.unique(val_labels)
num_classes = len(unique_labels)
print("Unique identities in val subset:", num_classes)

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
# Pairwise training dataset: (img1, img2, target) with target=1 (same) or -1 (different)
# ---------------------------------------------------------
class PairDataset(Dataset):
    def __init__(self, hf_dataset, transform=None, pos_prob=0.5):
        self.ds = hf_dataset
        self.transform = transform
        self.pos_prob = pos_prob

        labels = self.ds["label"]
        self.label_to_indices = defaultdict(list)
        for idx, lbl in enumerate(labels):
            self.label_to_indices[lbl].append(idx)
        self.labels = labels
        self.classes = list(self.label_to_indices.keys())

    def __len__(self):
        return len(self.ds)

    def _get_image(self, idx):
        sample = self.ds[idx]
        img = sample["image"]  # PIL
        if self.transform is not None:
            img = self.transform(img)
        label = sample["label"]
        return img, label

    def __getitem__(self, idx):
        img1, label1 = self._get_image(idx)

        is_positive = random.random() < self.pos_prob

        if is_positive:
            same_indices = self.label_to_indices[label1]
            if len(same_indices) > 1:
                idx2 = idx
                while idx2 == idx:
                    idx2 = random.choice(same_indices)
            else:
                # only one sample for this label -> fall back to negative
                is_positive = False

        if not is_positive:
            neg_label = random.choice([c for c in self.classes if c != label1])
            idx2 = random.choice(self.label_to_indices[neg_label])

        img2, label2 = self._get_image(idx2)

        target = 1.0 if is_positive else -1.0
        target = torch.tensor(target, dtype=torch.float32)

        return img1, img2, target


train_pairs = PairDataset(train_subset, transform=transform, pos_prob=POSITIVE_PROB)

train_loader = DataLoader(
    train_pairs,
    batch_size=BATCH_SIZE,
    shuffle=True,       # shuffle pairs each epoch (OK; underlying subset ordering is fixed)
    num_workers=4,
    pin_memory=True,
    drop_last=True,
)

# ---------------------------------------------------------
# Simple dataset for embedding extraction on validation
# ---------------------------------------------------------
class SingleImageDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        img = sample["image"]
        if self.transform is not None:
            img = self.transform(img)
        label = sample["label"]
        return img, label


val_images = SingleImageDataset(val_subset, transform=transform)

val_loader = DataLoader(
    val_images,
    batch_size=BATCH_SIZE,
    shuffle=False,      # preserve order for validation
    num_workers=4,
    pin_memory=True,
)

# ---------------------------------------------------------
# ResNet50 embedding model (all layers trainable)
# ---------------------------------------------------------
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, EMBED_DIM)  # embedding head
resnet50 = resnet50.to(device)

# Ensure everything is trainable (no freezing)
for p in resnet50.parameters():
    p.requires_grad = True

criterion = nn.CosineEmbeddingLoss(margin=0.5)  # subh 0.5
optimizer = torch.optim.Adam(resnet50.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)


def forward_embeddings(x):
    z = resnet50(x)
    z = F.normalize(z, p=2, dim=1)
    return z


# ---------------------------------------------------------
# Training loop
# ---------------------------------------------------------
resnet50.train()
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for step, (img1, img2, target) in enumerate(train_loader):
        img1 = img1.to(device, non_blocking=True)
        img2 = img2.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        emb1 = forward_embeddings(img1)
        emb2 = forward_embeddings(img2)

        loss = criterion(emb1, emb2, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (step + 1) % 50 == 0:
            avg_loss = running_loss / 50
            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Step [{step + 1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f}"
            )
            running_loss = 0.0

print("Training finished.")

# Optionally save model
torch.save(resnet50.state_dict(), "resnet50_vggface2_contrastive_first1pct.pth")

# ---------------------------------------------------------
# Embedding extraction on validation subset
# ---------------------------------------------------------
resnet50.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device, non_blocking=True)
        emb = forward_embeddings(imgs)  # [B, EMBED_DIM]
        all_embeddings.append(emb.cpu().numpy())
        all_labels.append(labels.numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)  # [N_val, EMBED_DIM]
all_labels = np.concatenate(all_labels, axis=0)          # [N_val]

print("Validation embeddings shape:", all_embeddings.shape)
print("Validation labels shape:", all_labels.shape)

# ---------------------------------------------------------
# Clustering + accuracy (Hungarian matched)
# ---------------------------------------------------------
unique_labels = np.unique(all_labels)
label_to_compact = {lab: i for i, lab in enumerate(unique_labels)}
y_true = np.array([label_to_compact[lab] for lab in all_labels])
n_classes = len(unique_labels)

print("Number of unique identities in val subset:", n_classes)

kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=RANDOM_SEED)
y_pred = kmeans.fit_predict(all_embeddings)

cm = contingency_matrix(y_true, y_pred)  # [n_true_classes, n_clusters]

cost_matrix = cm.max() - cm
row_ind, col_ind = linear_sum_assignment(cost_matrix)
optimal_matches = cm[row_ind, col_ind].sum()
clustering_accuracy = optimal_matches / cm.sum()

print(f"Clustering accuracy on first 1% val subset: {clustering_accuracy * 100:.2f}%")


import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# ---------------------------------------------------------
# 2D visualization of embeddings (t-SNE)
# ---------------------------------------------------------
# Compact ground-truth labels
unique_labels = np.unique(all_labels)
label_to_compact = {lab: i for i, lab in enumerate(unique_labels)}
y_true = np.array([label_to_compact[lab] for lab in all_labels])

# t-SNE to 2D
tsne = TSNE(
    n_components=2,
    perplexity=30,       # you can tune this (5–50)
    learning_rate="auto",
    init="random",
    random_state=RANDOM_SEED,
)
emb_2d = tsne.fit_transform(all_embeddings)  # [N_val, 2]

# ---------------------------------------------------------
# Plot 1: colored by TRUE identity
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
for cls in np.unique(y_true):
    mask = (y_true == cls)
    plt.scatter(
        emb_2d[mask, 0],
        emb_2d[mask, 1],
        s=5,
        alpha=0.7,
        label=f"class {cls}",
    )

plt.title("t-SNE of embeddings (colored by TRUE identity)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.savefig("tsne_true_labels.png", dpi=300)
plt.close()

# ---------------------------------------------------------
# Plot 2: colored by KMEANS cluster assignment
# ---------------------------------------------------------
plt.figure(figsize=(6, 6))
for clus in np.unique(y_pred):
    mask = (y_pred == clus)
    plt.scatter(
        emb_2d[mask, 0],
        emb_2d[mask, 1],
        s=5,
        alpha=0.7,
        label=f"cluster {clus}",
    )

plt.title("t-SNE of embeddings (colored by KMEANS cluster)")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.legend(markerscale=3, fontsize=8)
plt.tight_layout()
plt.savefig("tsne_kmeans_labels.png", dpi=300)
plt.close()

print("Saved plots: tsne_true_labels.png, tsne_kmeans_labels.png")
