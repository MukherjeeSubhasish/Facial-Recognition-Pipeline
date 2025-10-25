import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
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
data_dir = "./orl_faces"  # adjust path as needed
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# ----------------------------
# 3. Create output directory
# ----------------------------
save_dir = "visualized_samples"
os.makedirs(save_dir, exist_ok=True)

# ----------------------------
# 4. Helper: denormalize & save
# ----------------------------
def save_image_grid(tensor_batch, filename, nrow=8):
    """
    Saves a grid of denormalized images to disk.
    """
    # Denormalize from [-1,1] → [0,1]
    tensor_batch = tensor_batch * 0.5 + 0.5
    grid = utils.make_grid(tensor_batch, nrow=nrow, padding=2)
    npimg = grid.numpy()
    
    # Save using matplotlib
    plt.figure(figsize=(8, 8))
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', pad_inches=0)
    plt.close()

# ----------------------------
# 5. Fetch and save a batch
# ----------------------------
# Get one batch of data
dataiter = iter(train_loader)
images, labels = next(dataiter)

# Save first 16 images as a grid
save_image_grid(images[:16], "train_samples.png", nrow=4)

print(f" Saved sample image grid to: {os.path.join(save_dir, 'train_samples.png')}")


# ----------------------------
# 6. (Fixed) Save individual samples
# ----------------------------
for i in range(5):  # save first 5 images individually
    img = images[i] * 0.5 + 0.5  # denormalize
    npimg = img.numpy().transpose((1, 2, 0))  # (128,128,1)
    npimg = npimg.squeeze()  # remove the 3rd dim -> (128,128)
    
    plt.imsave(os.path.join(save_dir, f"sample_{i}_class_{labels[i].item()}.png"),
               npimg, cmap='gray')

print(f" Also saved 5 individual sample images in '{save_dir}' folder.")
