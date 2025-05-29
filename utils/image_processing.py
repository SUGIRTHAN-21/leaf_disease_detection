from PIL import Image
import numpy as np
import cv2

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocess the input image for the model.
    
    Args:
        image_path (str): Path to the image file
        target_size (tuple): Target size for resizing
        
    Returns:
        PIL.Image: Preprocessed image
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Resize image
    image = image.resize(target_size, Image.LANCZOS)
    
    return image

def apply_image_enhancement(image_path, output_path=None):
    """
    Apply image enhancement techniques to improve disease detection.
    
    Args:
        image_path (str): Path to the input image
        output_path (str, optional): Path to save the enhanced image
        
    Returns:
        str: Path to the enhanced image
    """
    # Read image using OpenCV
    img = cv2.imread(image_path)
    
    # Convert from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply image enhancements
    # 1. Contrast enhancement using CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # 2. Denoise
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    
    # 3. Sharpen the image
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)
    
    # Convert back to BGR for saving
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Save the enhanced image
    if output_path is None:
        output_path = image_path.replace('.', '_enhanced.')
    
    cv2.imwrite(output_path, img)
    
    return output_path

def extract_features(image_path):
    """
    Extract features from the image that might be useful for disease analysis.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Read image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Calculate color statistics
    mean_color = np.mean(img, axis=(0, 1))
    std_color = np.std(img, axis=(0, 1))
    
    # Calculate texture features using GLCM
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Simple texture analysis using standard deviation
    texture_std = np.std(gray)
    
    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
    
    features = {
        "color_mean_r": mean_color[0],
        "color_mean_g": mean_color[1],
        "color_mean_b": mean_color[2],
        "color_std_r": std_color[0],
        "color_std_g": std_color[1],
        "color_std_b": std_color[2],
        "texture_std": texture_std,
        "edge_density": edge_density
    }
    
    return features