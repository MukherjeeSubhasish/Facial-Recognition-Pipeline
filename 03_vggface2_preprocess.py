from torchvision import transforms
from datasets import load_dataset

# Load dataset
dataset = load_dataset("chronopt-research/cropped-vggface2-224")

# Define the ResNet preprocessing transform
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # just to be safe
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]) # why? RESNET was trained using these values
])

# Apply the transform
def transform_batch(examples):
    examples["pixel_values"] = [resnet_transform(img.convert("RGB")) for img in examples["image"]]
    return examples

new_dataset = dataset.with_transform(transform_batch)
pass
