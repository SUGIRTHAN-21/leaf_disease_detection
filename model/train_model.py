import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Define the LeafDiseaseModel class
class LeafDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(LeafDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

def train_model():
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Optimized Data transformations (faster)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load dataset
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(project_root, 'datasets', 'PlantVillage')
    print(f"Loading dataset from: {data_dir}")

    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory {data_dir} does not exist.")
        return

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    print(f"Found {len(full_dataset)} images across {len(full_dataset.classes)} classes")

    # Optional: Use a subset of the data (e.g. 10%) to speed up testing
    subset_size = int(0.2 * len(full_dataset))
    full_dataset, _ = random_split(full_dataset, [subset_size, len(full_dataset) - subset_size])

    # Train/validation split
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # Save class labels
    class_labels = {idx: label for idx, label in enumerate(full_dataset.dataset.classes)}
    with open(os.path.join(current_dir, 'class_labels.json'), 'w') as f:
        json.dump(class_labels, f)

    # Create model
    num_classes = len(full_dataset.dataset.classes)
    model = LeafDiseaseModel(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 5
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training
        model.train()
        train_loss, train_correct = 0.0, 0
        total_train = 0

        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels.data).item()
            total_train += inputs.size(0)

        train_acc = train_correct / total_train
        print(f"Train Loss: {train_loss/total_train:.4f}, Accuracy: {train_acc:.4f}")

        # Validation
        model.eval()
        val_loss, val_correct = 0.0, 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data).item()
                total_val += inputs.size(0)

        val_acc = val_correct / total_val
        print(f"Val Loss: {val_loss/total_val:.4f}, Accuracy: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = os.path.join(current_dir, 'model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved best model to {model_path}")

    print("Training completed.")

if __name__ == "__main__":
    train_model()