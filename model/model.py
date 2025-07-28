import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import json
import cv2
import numpy as np

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

def is_leaf_image(image_path, debug=False):
    """
    Advanced leaf detection using multiple validation techniques.
    
    Args:
        image_path (str): Path to the image file
        debug (bool): Whether to print debug information
        
    Returns:
        tuple: (is_leaf: bool, confidence_score: float, reason: str)
    """
    try:
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, "Could not load image"
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray.shape
        total_pixels = height * width
        
        validation_scores = []
        reasons = []
        
        # 1. GREEN COLOR ANALYSIS
        # Define green color ranges in HSV
        lower_green1 = np.array([35, 40, 40])   # Light green
        upper_green1 = np.array([85, 255, 255])
        
        lower_green2 = np.array([25, 30, 30])   # Yellow-green
        upper_green2 = np.array([35, 255, 255])
        
        # Create masks for green colors
        mask_green1 = cv2.inRange(img_hsv, lower_green1, upper_green1)
        mask_green2 = cv2.inRange(img_hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask_green1, mask_green2)
        
        # Calculate green percentage
        green_pixels = np.sum(green_mask > 0)
        green_percentage = (green_pixels / total_pixels) * 100
        
        # Green validation (leaves should have significant green content)
        if green_percentage > 15:  # At least 15% green
            validation_scores.append(0.8)
            reasons.append(f"Good green content: {green_percentage:.1f}%")
        elif green_percentage > 8:
            validation_scores.append(0.5)
            reasons.append(f"Moderate green content: {green_percentage:.1f}%")
        else:
            validation_scores.append(0.2)
            reasons.append(f"Low green content: {green_percentage:.1f}%")
        
        # 2. EDGE DETECTION AND SHAPE ANALYSIS
        # Apply Gaussian blur and edge detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Calculate edge density
        edge_pixels = np.sum(edges > 0)
        edge_density = (edge_pixels / total_pixels) * 100
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (likely the main object)
            largest_contour = max(contours, key=cv2.contourArea)
            contour_area = cv2.contourArea(largest_contour)
            contour_percentage = (contour_area / total_pixels) * 100
            
            # Calculate contour properties
            perimeter = cv2.arcLength(largest_contour, True)
            if perimeter > 0:
                # Circularity (4π * area / perimeter²) - leaves are typically not circular
                circularity = (4 * np.pi * contour_area) / (perimeter * perimeter)
                
                # Aspect ratio analysis
                x, y, w, h = cv2.boundingRect(largest_contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Leaf-like shape validation
                if 0.1 < circularity < 0.7 and contour_percentage > 10:  # Not too circular, reasonable size
                    validation_scores.append(0.7)
                    reasons.append(f"Leaf-like shape detected (circularity: {circularity:.2f})")
                else:
                    validation_scores.append(0.3)
                    reasons.append(f"Questionable shape (circularity: {circularity:.2f})")
            else:
                validation_scores.append(0.2)
                reasons.append("No clear object boundary detected")
        else:
            validation_scores.append(0.1)
            reasons.append("No contours detected")
        
        # 3. TEXTURE ANALYSIS
        # Calculate texture features using Local Binary Pattern concept
        # Simplified texture analysis using standard deviation in local regions
        texture_scores = []
        kernel_size = 15
        
        for i in range(0, height - kernel_size, kernel_size):
            for j in range(0, width - kernel_size, kernel_size):
                region = gray[i:i+kernel_size, j:j+kernel_size]
                texture_scores.append(np.std(region))
        
        avg_texture = np.mean(texture_scores) if texture_scores else 0
        
        # Leaves typically have moderate texture variation
        if 10 < avg_texture < 60:
            validation_scores.append(0.6)
            reasons.append(f"Good texture variation: {avg_texture:.1f}")
        else:
            validation_scores.append(0.3)
            reasons.append(f"Unusual texture: {avg_texture:.1f}")
        
        # 4. COLOR DISTRIBUTION ANALYSIS
        # Check if image has natural color distribution
        # Calculate color histogram
        hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
        
        # Check for color diversity (avoid pure white/black/single color images)
        color_peaks = 0
        for hist in [hist_r, hist_g, hist_b]:
            peaks = np.where(hist > total_pixels * 0.1)[0]  # Bins with >10% of pixels
            color_peaks += len(peaks)
        
        if 3 <= color_peaks <= 20:  # Reasonable color diversity
            validation_scores.append(0.5)
            reasons.append("Good color diversity")
        else:
            validation_scores.append(0.2)
            reasons.append("Poor color diversity")
        
        # 5. BRIGHTNESS AND CONTRAST CHECK
        mean_brightness = np.mean(gray)
        brightness_std = np.std(gray)
        
        # Avoid extremely dark, bright, or low-contrast images
        if 30 < mean_brightness < 220 and brightness_std > 15:
            validation_scores.append(0.4)
            reasons.append("Good brightness and contrast")
        else:
            validation_scores.append(0.2)
            reasons.append(f"Poor lighting (brightness: {mean_brightness:.1f}, contrast: {brightness_std:.1f})")
        
        # Calculate final confidence score
        final_confidence = np.mean(validation_scores)
        
        # Decision threshold
        is_leaf = final_confidence > 0.45  # Adjust threshold as needed
        
        if debug:
            print(f"Leaf Detection Analysis:")
            print(f"Green percentage: {green_percentage:.1f}%")
            print(f"Edge density: {edge_density:.1f}%")
            print(f"Average texture: {avg_texture:.1f}")
            print(f"Color peaks: {color_peaks}")
            print(f"Brightness: {mean_brightness:.1f}, Contrast: {brightness_std:.1f}")
            print(f"Validation scores: {validation_scores}")
            print(f"Final confidence: {final_confidence:.3f}")
            print(f"Is leaf: {is_leaf}")
        
        main_reason = f"Confidence: {final_confidence:.2f}, " + "; ".join(reasons[:3])
        
        return is_leaf, final_confidence, main_reason
        
    except Exception as e:
        return False, 0.0, f"Error in leaf detection: {str(e)}"

def predict_disease(image_path):
    """
    Enhanced prediction function with leaf validation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (disease_name: str, confidence: float, is_leaf: bool, leaf_confidence: float)
    """
    global model, class_labels
    
    # First, validate if the image is a leaf
    is_leaf, leaf_confidence, leaf_reason = is_leaf_image(image_path, debug=True)
    
    if not is_leaf:
        return "NOT_A_LEAF", 0.0, False, leaf_confidence
    
    # If it's a leaf, proceed with disease classification
    if model is None or class_labels is None:
        model = load_model()
    
    # Load image and preprocess
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict disease
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        confidence, predicted = torch.max(probabilities, 1)
        predicted_index = predicted.item()
        confidence_value = confidence.item() * 100
        
        # Additional confidence check - if model confidence is too low, might not be a proper leaf
        if confidence_value < 30:  # Very low confidence
            return "UNCLEAR_IMAGE", confidence_value, True, leaf_confidence
        
        # Get disease name from class_labels JSON
        disease_name = class_labels.get(str(predicted_index), "unknown")
    
    return disease_name, confidence_value, True, leaf_confidence

# Test prediction if run standalone (optional)
if __name__ == "__main__":
    test_img_path = "test_leaf.jpg"  # Replace with a valid test image path
    if os.path.exists(test_img_path):
        disease, conf, is_leaf, leaf_conf = predict_disease(test_img_path)
        print(f"Is leaf: {is_leaf} (confidence: {leaf_conf:.2f})")
        if is_leaf:
            print(f"Predicted disease: {disease}, Confidence: {conf:.2f}%")
        else:
            print("Image is not recognized as a leaf.")
    else:
        print("Test image not found.")