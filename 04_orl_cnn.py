import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Image preprocessing
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # ensure grayscale
    transforms.Resize((128, 128)),                # resize for CNN
    transforms.ToTensor(),                        # convert to tensor [0,1]
    transforms.Normalize(mean=[0.5], std=[0.5])   # scale to [-1,1]
])

# Load dataset
data_dir = "./orl_faces"  # adjust if necessary
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split into train/val
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

import torch.nn as nn
import torch.nn.functional as F

class ORL_CNN(nn.Module):
    def __init__(self, num_classes=40):
        super(ORL_CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

import torch.optim as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ORL_CNN(num_classes=40).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

model.eval()
correct, total = 0, 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# remove final classification layer (fc2) weights
state_dict = model.state_dict()

# create a new state dict without fc2 weights
filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('fc2.')}

# save the filtered weights
torch.save(filtered_state_dict, "orl_cnn_features.pth")

print("✅ Saved model weights (excluding final classification layer) to orl_cnn_features.pth")

