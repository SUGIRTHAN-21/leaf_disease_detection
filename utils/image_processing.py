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

def validate_image_quality(image_path):
    """
    Validate image quality for better leaf detection.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Quality metrics and recommendations
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {"valid": False, "reason": "Cannot load image"}
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        quality_metrics = {}
        
        # 1. Resolution check
        min_resolution = 100
        if height < min_resolution or width < min_resolution:
            quality_metrics["resolution"] = "Too low"
        else:
            quality_metrics["resolution"] = "Good"
        
        # 2. Brightness check
        mean_brightness = np.mean(gray)
        if mean_brightness < 50:
            quality_metrics["brightness"] = "Too dark"
        elif mean_brightness > 200:
            quality_metrics["brightness"] = "Too bright"
        else:
            quality_metrics["brightness"] = "Good"
        
        # 3. Contrast check
        contrast = np.std(gray)
        if contrast < 20:
            quality_metrics["contrast"] = "Too low"
        else:
            quality_metrics["contrast"] = "Good"
        
        # 4. Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < 100:
            quality_metrics["sharpness"] = "Blurry"
        else:
            quality_metrics["sharpness"] = "Sharp"
        
        # Overall quality assessment
        good_metrics = sum(1 for v in quality_metrics.values() if v == "Good" or v == "Sharp")
        total_metrics = len(quality_metrics)
        
        quality_score = good_metrics / total_metrics
        
        return {
            "valid": quality_score >= 0.5,
            "quality_score": quality_score,
            "metrics": quality_metrics,
            "recommendations": get_quality_recommendations(quality_metrics)
        }
        
    except Exception as e:
        return {"valid": False, "reason": f"Error analyzing image: {str(e)}"}

def get_quality_recommendations(metrics):
    """
    Get recommendations based on image quality metrics.
    
    Args:
        metrics (dict): Quality metrics
        
    Returns:
        list: List of recommendations
    """
    recommendations = []
    
    if metrics.get("resolution") == "Too low":
        recommendations.append("Use a higher resolution image (at least 224x224 pixels)")
    
    if metrics.get("brightness") == "Too dark":
        recommendations.append("Take the photo in better lighting or increase brightness")
    elif metrics.get("brightness") == "Too bright":
        recommendations.append("Reduce exposure or avoid direct sunlight")
    
    if metrics.get("contrast") == "Too low":
        recommendations.append("Improve contrast by using better lighting conditions")
    
    if metrics.get("sharpness") == "Blurry":
        recommendations.append("Ensure the camera is focused and stable when taking the photo")
    
    if not recommendations:
        recommendations.append("Image quality is good for analysis")
    
    return recommendations

def extract_leaf_specific_features(image_path):
    """
    Extract features specifically useful for leaf validation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Leaf-specific features
    """
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        features = {}
        
        # 1. Green color analysis
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(img_hsv, lower_green, upper_green)
        green_percentage = (np.sum(green_mask > 0) / green_mask.size) * 100
        features["green_percentage"] = green_percentage
        
        # 2. Shape analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            if perimeter > 0:
                circularity = (4 * np.pi * area) / (perimeter * perimeter)
                features["circularity"] = circularity
                features["contour_area_ratio"] = area / gray.size
            else:
                features["circularity"] = 0
                features["contour_area_ratio"] = 0
        else:
            features["circularity"] = 0
            features["contour_area_ratio"] = 0
        
        # 3. Texture analysis
        # Calculate Local Binary Pattern-like texture
        texture_score = np.std(gray)
        features["texture_variation"] = texture_score
        
        # 4. Color distribution
        # Check if image has natural color variation
        color_std = np.std(img_rgb, axis=(0, 1))
        features["color_variation"] = np.mean(color_std)
        
        return features
        
    except Exception as e:
        return {"error": str(e)}

def create_leaf_validation_report(image_path):
    """
    Create a comprehensive report for leaf validation.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        dict: Comprehensive validation report
    """
    quality_report = validate_image_quality(image_path)
    leaf_features = extract_leaf_specific_features(image_path)
    general_features = extract_features(image_path)
    
    report = {
        "image_path": image_path,
        "quality_assessment": quality_report,
        "leaf_features": leaf_features,
        "general_features": general_features,
        "timestamp": np.datetime64('now').astype(str)
    }
    
    return report