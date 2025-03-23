import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm
from PIL import Image
import shutil

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_model():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    model.load_state_dict(torch.load("beauty_classifier.pth", map_location=device))
    model.to(device)
    model.eval()
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# -------------------------------
# 📌 8. Function to Classify a Single Image
# -------------------------------
def classify_image(image_path, model):
    """Classify a single image as Beautiful or Not Beautiful."""
    image = Image.open(image_path).convert("RGB")
    image = val_transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        label = "Beautiful" if output.item() > 0.5 else "Not Beautiful"

    return label



def sort_images(input_folder, output_folder):
    """Sorts images into 'Beautiful' or 'Not Beautiful' folders."""
    beautiful_folder = os.path.join(output_folder, "Beautiful")
    not_beautiful_folder = os.path.join(output_folder, "Not_Beautiful")

    os.makedirs(beautiful_folder, exist_ok=True)
    os.makedirs(not_beautiful_folder, exist_ok=True)

    model = load_model()

    for image_name in os.listdir(input_folder):
        image_path = os.path.join(input_folder, image_name)
        if image_path.lower().endswith(('png', 'jpg', 'jpeg')):
            label = classify_image(image_path, model)
            dest_folder = beautiful_folder if label == "Beautiful" else not_beautiful_folder
            shutil.move(image_path, os.path.join(dest_folder, image_name))
            print(f"Moved {image_name} -> {dest_folder}")

# -------------------------------
# 📌 10. Run the Classifier on New Images
# -------------------------------
input_folder = "new_images"  # Folder containing images to classify
output_folder = "sorted_images"  # Destination folders

sort_images(input_folder, output_folder)
print("✅ Image sorting completed!")