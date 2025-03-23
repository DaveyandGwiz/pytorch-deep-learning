import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader

# Define transformations for training & validation
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to match model input size
    transforms.RandomHorizontalFlip(),  # Augmentations to improve generalization
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Standard normalization
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder("train", transform=train_transforms)
val_dataset = datasets.ImageFolder("val", transform=val_transforms)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
import torch.nn as nn
import torchvision.models as models

# Load pre-trained ResNet18
model = models.resnet18(pretrained=True)

# Modify the last layer for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_ftrs, 256),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(256, 1),  # Output: single neuron for binary classification
    nn.Sigmoid()  # Convert logits to probability
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
import torch.optim as optim

# Binary Cross-Entropy loss for binary classification
criterion = nn.BCELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)
from tqdm import tqdm

num_epochs = 10  # Number of training epochs

for epoch in range(num_epochs):
    model.train()  # Set to training mode
    running_loss = 0.0

    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.float().to(device)  # Move to GPU
        labels = labels.view(-1, 1)  # Reshape labels for binary classification

        optimizer.zero_grad()  # Reset gradients
        outputs = model(images)  # Forward pass
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
        loop.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.view(-1, 1).float()
            outputs = model(images)
            predictions = (outputs > 0.5).float()  # Convert probability to class label
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(), "beauty_classifier.pth")
model.load_state_dict(torch.load("beauty_classifier.pth"))
model.eval()
