import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json

# Define your model architecture
class LeafDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(LeafDiseaseModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)

# Use GPU if available, else CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Image preprocessing transforms (adjust size if you trained with different input size)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_labels = None  # To hold class index -> disease name mapping
model = None         # The PyTorch model

def load_model():
    global class_labels
    
    # Paths relative to this file
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, 'model.pth')
    labels_path = os.path.join(base_dir, 'class_labels.json')
    
    # Load class labels mapping
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Class labels file not found: {labels_path}")
    
    with open(labels_path, 'r') as f:
        class_labels = json.load(f)
    
    # Number of classes based on label map
    num_classes = len(class_labels)
    
    # Initialize the model and load weights
    model = LeafDiseaseModel(num_classes)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_disease(image_path):
    global model, class_labels
    
    if model is None or class_labels is None:
        model = load_model()
    
    # Load image and preprocess
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        confidence, predicted = torch.max(probabilities, 1)
        predicted_index = predicted.item()
        confidence_value = confidence.item() * 100
        
        # Get disease name from class_labels JSON
        disease_name = class_labels.get(str(predicted_index), "unknown")
    
    return disease_name, confidence_value

# Test prediction if run standalone (optional)
if __name__ == "__main__":
    test_img_path = "test_leaf.jpg"  # Replace with a valid test image path
    disease, conf = predict_disease(test_img_path)
    print(f"Predicted disease: {disease}, Confidence: {conf:.2f}%")
