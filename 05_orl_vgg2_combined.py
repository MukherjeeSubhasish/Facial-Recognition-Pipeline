import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader

# -----------------------------
# Model definition
# -----------------------------
class ORL_CNN_FeatureExtractor(nn.Module):
    def __init__(self):
        super(ORL_CNN_FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return x  # 256-D features


# -----------------------------
# Load pretrained (feature-only) weights
# -----------------------------
model = ORL_CNN_FeatureExtractor()
state_dict = torch.load("orl_cnn_features.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
print("Loaded feature extractor weights successfully!")


# -----------------------------
# Load & preprocess VGGFace2
# -----------------------------
dataset = load_dataset("chronopt-research/cropped-vggface2-224")

vggface2_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # match ORL model
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

def transform_batch(examples):
    examples["pixel_values"] = [vggface2_transform(img.convert("RGB")) for img in examples["image"]]
    return examples

dataset = dataset.with_transform(transform_batch)
print("VGGFace2 dataset preprocessing ready for the ORL CNN feature extractor.")


# -----------------------------
# DataLoader with custom collate_fn
# -----------------------------
def collate_fn(batch):
    # stack tensors from "pixel_values" field
    pixel_values = torch.stack([example["pixel_values"] for example in batch])
    labels = torch.tensor([example["label"] for example in batch])
    return {"pixel_values": pixel_values, "labels": labels}

loader = DataLoader(dataset["train"], batch_size=8096, shuffle=False, collate_fn=collate_fn)


# -----------------------------
# Extract embeddings
# -----------------------------
all_features = []
model.eval()
with torch.no_grad():
    for batch in loader:
        imgs = batch["pixel_values"]
        feats = model(imgs)
        all_features.append(feats)

features = torch.cat(all_features, dim=0)
print("Extracted features shape:", features.shape)

# Optionally save for later use
torch.save(features, "vggface2_orl_features.pt")
print("Saved features to vggface2_orl_features.pt")
