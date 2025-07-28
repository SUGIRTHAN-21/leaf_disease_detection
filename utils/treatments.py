import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Your existing treatments database
TREATMENTS = {
    "Apple___Apple_scab": {
        "disease_name": "Apple Scab",
        "description": "A fungal disease that causes dark, scabby lesions on leaves and fruit.",
        "treatments": [
            "Apply fungicides containing myclobutanil, captan, or sulfur at 7-10 day intervals starting at bud break.",
            "Remove and destroy fallen leaves in autumn to reduce fungal spores.",
            "Prune trees to improve air circulation and reduce humidity in the canopy.",
            "Plant resistant varieties when establishing new orchards."
        ],
        "prevention": [
            "Maintain good orchard sanitation by removing fallen leaves and fruit.",
            "Apply protective fungicide sprays before rainy periods.",
            "Ensure adequate spacing between trees for better airflow.",
            "Avoid overhead irrigation which can spread spores."
        ],
        "organic_options": [
            "Spray with organic sulfur-based fungicides.",
            "Apply neem oil every 7-14 days during growing season.",
            "Use compost tea as a preventive spray."
        ]
    },

    "Apple___Black_rot": {
        "disease_name": "Apple Black Rot",
        "description": "A fungal disease causing leaf spots, fruit rot, and cankers on branches.",
        "treatments": [
            "Apply fungicides containing captan, myclobutanil, or thiophanate-methyl.",
            "Remove mummified fruit and cankers from trees.",
            "Prune out dead or diseased wood in late winter.",
            "Ensure proper fertility to maintain tree vigor."
        ],
        "prevention": [
            "Maintain good orchard sanitation.",
            "Remove nearby wild apple trees that may harbor disease.",
            "Avoid tree wounds which provide entry points for the fungus.",
            "Control insects that create wounds in the fruit and bark."
        ],
        "organic_options": [
            "Apply copper-based fungicides before bud break.",
            "Use potassium bicarbonate sprays during growing season.",
            "Apply compost tea to boost tree immunity."
        ]
    },

    "Apple___Cedar_apple_rust": {
        "disease_name": "Cedar Apple Rust",
        "description": "A fungal disease requiring both apple trees and junipers/cedars to complete its life cycle.",
        "treatments": [
            "Apply fungicides containing myclobutanil, propiconazole, or mancozeb from bud break until early summer.",
            "Remove nearby juniper or cedar trees if practical.",
            "Focus protective sprays during spring when spores are active."
        ],
        "prevention": [
            "Plant resistant apple varieties.",
            "Avoid planting apples near cedar or juniper trees.",
            "Apply preventive fungicides before spring rain periods."
        ],
        "organic_options": [
            "Apply sulfur sprays before infection periods.",
            "Use neem oil as a preventive measure.",
            "Remove visible orange telial horns from nearby juniper trees."
        ]
    },

    "Apple___healthy": {
        "disease_name": "Healthy Apple Tree",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatments": ["No treatment necessary."],
        "prevention": [
            "Continue regular orchard maintenance.",
            "Monitor for early signs of disease.",
            "Follow recommended fertilization schedule.",
            "Maintain proper pruning for airflow."
        ],
        "organic_options": [
            "Apply compost tea to promote soil health.",
            "Use seaweed extracts as a natural growth promoter.",
            "Release beneficial insects for pest prevention."
        ]
    },

    "Tomato___Early_blight": {
        "disease_name": "Tomato Early Blight",
        "description": "Dark concentric spots on lower leaves, which yellow and drop.",
        "treatments": [
            "Apply fungicides containing chlorothalonil, mancozeb, or copper.",
            "Remove and destroy infected leaves.",
            "Rotate crops every 2-3 years.",
            "Avoid overhead watering."
        ],
        "prevention": [
            "Use disease-free seeds and transplants.",
            "Provide adequate plant spacing.",
            "Mulch around plants.",
            "Stake or cage plants to prevent leaf contact with soil."
        ],
        "organic_options": [
            "Apply copper-based fungicides.",
            "Spray compost tea.",
            "Use potassium bicarbonate sprays."
        ]
    },

    "Tomato___Late_blight": {
        "disease_name": "Tomato Late Blight",
        "description": "Dark water-soaked lesions on leaves and fruit.",
        "treatments": [
            "Apply fungicides with chlorothalonil, mancozeb, or copper hydroxide.",
            "Destroy infected plants immediately.",
            "Harvest healthy fruit early."
        ],
        "prevention": [
            "Plant resistant varieties.",
            "Ensure good airflow.",
            "Avoid overhead irrigation.",
            "Monitor blight-prone weather."
        ],
        "organic_options": [
            "Apply copper fungicides preventively.",
            "Use organic phosphorus acid sprays.",
            "Remove volunteer tomato and potato plants."
        ]
    },

    "Tomato___healthy": {
        "disease_name": "Healthy Tomato Plant",
        "description": "The plant appears healthy with no visible disease symptoms.",
        "treatments": ["No treatment necessary."],
        "prevention": [
            "Maintain regular garden care.",
            "Ensure consistent watering and fertilizing.",
            "Monitor for early signs of pests or disease.",
            "Support plants properly."
        ],
        "organic_options": [
            "Use compost tea for plant vigor.",
            "Plant basil or marigolds nearby.",
            "Use neem oil preventively."
        ]
    },

    "unknown": {
        "disease_name": "Unknown Disease",
        "description": "The disease could not be identified with certainty.",
        "treatments": [
            "Consult a local agriculture expert.",
            "Remove affected plants if necessary.",
            "Use a broad-spectrum fungicide if unsure."
        ],
        "prevention": [
            "Improve air circulation.",
            "Avoid overhead watering.",
            "Practice crop rotation.",
            "Keep the garden clean and debris-free."
        ],
        "organic_options": [
            "Use neem oil or copper-based sprays.",
            "Apply compost tea.",
            "Introduce beneficial microbes or insects."
        ]
    }
}

class PlantDiseaseAnalyzer:
    def __init__(self, model_path, class_labels_path):
        """
        Initialize the plant disease analyzer with model and class labels.
        
        Args:
            model_path (str): Path to the trained model file
            class_labels_path (str): Path to the class labels JSON file
        """
        self.model = None
        self.class_labels = []
        self.load_model(model_path)
        self.load_class_labels(class_labels_path)
    
    def load_model(self, model_path):
        """Load the trained TensorFlow model."""
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    
    def load_class_labels(self, class_labels_path):
        """Load class labels from JSON file."""
        try:
            with open(class_labels_path, 'r') as f:
                self.class_labels = json.load(f)
            print(f"Class labels loaded: {len(self.class_labels)} classes")
        except Exception as e:
            print(f"Error loading class labels: {str(e)}")
            # Fallback to common plant disease classes if file not found
            self.class_labels = [
                "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
                "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___healthy"
            ]
    
    def preprocess_image(self, image_path, target_size=(224, 224)):
        """
        Preprocess image for model prediction.
        
        Args:
            image_path (str): Path to the image file
            target_size (tuple): Target size for resizing
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = image.convert('RGB')  # Ensure RGB format
            image = image.resize(target_size)
            
            # Convert to array and normalize
            image_array = np.array(image)
            image_array = image_array / 255.0  # Normalize to [0,1]
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict_disease(self, image_path, confidence_threshold=0.5):
        """
        Predict plant disease from image.
        
        Args:
            image_path (str): Path to the image file
            confidence_threshold (float): Minimum confidence for prediction
            
        Returns:
            tuple: (predicted_class, confidence_score)
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(processed_image)
            predicted_index = np.argmax(predictions[0])
            confidence = float(predictions[0][predicted_index])
            
            # Get predicted class name
            if predicted_index < len(self.class_labels):
                predicted_class = self.class_labels[predicted_index]
            else:
                predicted_class = "unknown"
            
            print(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
            
            # Return unknown if confidence is too low
            if confidence < confidence_threshold:
                return "unknown", confidence
            
            return predicted_class, confidence
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return "unknown", 0.0

def analyze_and_treat(image_path, model_path="model/model.pth", class_labels_path="model/class_labels.json"):
    """
    Complete workflow: analyze image and return treatment recommendations.
    
    Args:
        image_path (str): Path to the uploaded image
        model_path (str): Path to the trained model
        class_labels_path (str): Path to class labels file
        
    Returns:
        dict: Complete analysis and treatment information
    """
    try:
        # Initialize analyzer
        analyzer = PlantDiseaseAnalyzer(model_path, class_labels_path)
        
        # Predict disease
        predicted_disease, confidence = analyzer.predict_disease(image_path)
        
        # Get treatment information
        treatment_info = get_treatment(predicted_disease)
        
        # Add prediction metadata
        result = {
            "prediction": {
                "disease_class": predicted_disease,
                "confidence": confidence,
                "model_used": model_path
            },
            "treatment": treatment_info
        }
        
        return result
        
    except Exception as e:
        print(f"Error in analysis workflow: {str(e)}")
        # Return fallback response
        return {
            "prediction": {
                "disease_class": "unknown",
                "confidence": 0.0,
                "error": str(e)
            },
            "treatment": get_treatment("unknown")
        }

def get_treatment(disease_name):
    """
    Get treatment information for a specific disease.
    
    Args:
        disease_name (str): The name of the detected disease
        
    Returns:
        dict: Treatment information
    """
    # Clean and standardize input
    disease_name = disease_name.strip()

    # Optional debug output
    print(f"[DEBUG] Disease name received: '{disease_name}'")

    # Return treatment if available
    if disease_name in TREATMENTS:
        return TREATMENTS[disease_name]
    else:
        print(f"[WARNING] Disease '{disease_name}' not found. Returning fallback.")
        return TREATMENTS["unknown"]

# Quick test function
def test_system():
    """Test the complete system with sample data."""
    print("Testing Plant Disease Analysis System...")
    
    # Test treatment lookup
    test_diseases = ["Apple___Apple_scab", "Tomato___Early_blight", "unknown"]
    for disease in test_diseases:
        treatment = get_treatment(disease)
        print(f"\nDisease: {treatment['disease_name']}")
        print(f"Description: {treatment['description']}")
        print(f"Treatments: {len(treatment['treatments'])} available")

if __name__ == "__main__":
    test_system()