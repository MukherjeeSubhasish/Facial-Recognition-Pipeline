import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

# ----------------------------
# 1. Image preprocessing
# ----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure grayscale
    transforms.Resize((128, 128)),                # resize for CNN
    transforms.ToTensor(),                        # convert to tensor [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])   # scale to [-1,1]
])

# ----------------------------
# 2. Load dataset
# ----------------------------
data_dir = "./orl_faces"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Optional: subset for faster PCA computation
subset_size = min(200, len(dataset))
subset_indices = torch.randperm(len(dataset))[:subset_size]
subset = torch.utils.data.Subset(dataset, subset_indices)

loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
images, labels = next(iter(loader))  # shape: [N, 1, 128, 128]

# ----------------------------
# 3. Flatten images for PCA
# ----------------------------
X = images.view(images.size(0), -1).numpy()  # shape: [N, 16384]
y = labels.numpy()

# ----------------------------
# 4. Run PCA
# ----------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

# ----------------------------
# 5. Visualization
# ----------------------------
output_dir = "pca_visualizations"
os.makedirs(output_dir, exist_ok=True)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', s=40, alpha=0.8)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization of ORL Faces Dataset')
plt.colorbar(scatter, label='Class Label')
plt.tight_layout()

# Save to file
output_path = os.path.join(output_dir, "orl_faces_pca_2d.png")
plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()

print(f" PCA plot saved to: {output_path}")

# import torch
# from torch.utils.data import DataLoader
# from torchvision import datasets, transforms
# import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
# import numpy as np
# import os

# # ----------------------------
# # 1. Image preprocessing
# # ----------------------------
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# # ----------------------------
# # 2. Load dataset
# # ----------------------------
# data_dir = "./orl_faces"
# dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# # Take 40 samples only (10 per group)
# subset_size = min(40, len(dataset))
# subset_indices = torch.randperm(len(dataset))[:subset_size]
# subset = torch.utils.data.Subset(dataset, subset_indices)

# loader = DataLoader(subset, batch_size=len(subset), shuffle=False)
# images, labels = next(iter(loader))

# # ----------------------------
# # 3. Flatten images for PCA
# # ----------------------------
# X = images.view(images.size(0), -1).numpy()
# y = labels.numpy()

# # ----------------------------
# # 4. Run PCA
# # ----------------------------
# pca = PCA(n_components=2)
# X_pca = pca.fit_transform(X)

# # ----------------------------
# # 5. Assign 4 color groups (10 samples each)
# # ----------------------------
# n_colors = 4
# samples_per_color = len(X_pca) // n_colors
# color_groups = np.repeat(np.arange(n_colors), samples_per_color)

# # ----------------------------
# # 6. Visualization
# # ----------------------------
# output_dir = "pca_visualizations"
# os.makedirs(output_dir, exist_ok=True)

# plt.figure(figsize=(8, 6))
# scatter = plt.scatter(
#     X_pca[:, 0], X_pca[:, 1],
#     c=color_groups, cmap='tab10',
#     s=50, alpha=0.85
# )
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.title('PCA Visualization (4 Color Groups, 10 Samples Each)')
# plt.colorbar(scatter, label='Group Index')
# plt.tight_layout()

# # Save plot to file
# output_path = os.path.join(output_dir, "orl_faces_pca_4colors.png")
# plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
# plt.close()

# print(f" PCA plot saved to: {output_path}")
