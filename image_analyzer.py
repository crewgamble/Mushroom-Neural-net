import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from collections import Counter

class MushroomImageAnalyzer:
    def __init__(self):
        # Load pre-trained ResNet model
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        
        # Define image transformations
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Color mappings for cap-color
        self.color_mappings = {
            'brown': ('n', [(101, 67, 33), (139, 69, 19), (160, 82, 45)]),  # various browns
            'red': ('e', [(178, 34, 34), (220, 20, 60), (139, 0, 0)]),      # various reds
            'yellow': ('y', [(255, 215, 0), (218, 165, 32), (184, 134, 11)]), # various yellows
            'white': ('w', [(255, 255, 255), (245, 245, 245), (240, 240, 240)]) # various whites
        }
        
        # Define feature mappings (these would need to be trained/fine-tuned)
        self.feature_extractors = {
            'cap-color': self._extract_cap_color,
            'cap-shape': self._extract_cap_shape,
            # Add more feature extractors as needed
        }
    
    def analyze_image(self, image_path):
        """Analyze mushroom image and extract features"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            
            # Extract features
            features = {}
            
            # Extract cap color
            features['cap-color'] = self._extract_cap_color(image)
            
            print(f"Extracted features: {features}")  # Debug print
            return features
            
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")
            return {}
    
    def _extract_cap_color(self, image):
        """Extract cap color from image using color analysis"""
        try:
            # Resize image for faster processing
            image = image.resize((100, 100))
            pixels = list(image.getdata())
            
            # Get the most common colors in the image
            color_counts = Counter(pixels)
            dominant_colors = color_counts.most_common(5)
            
            # Find the closest matching predefined color
            best_match = None
            min_distance = float('inf')
            
            for dominant_color, _ in dominant_colors:
                for color_name, (code, reference_colors) in self.color_mappings.items():
                    for ref_color in reference_colors:
                        distance = sum((a - b) ** 2 for a, b in zip(dominant_color, ref_color))
                        if distance < min_distance:
                            min_distance = distance
                            best_match = code
            
            print(f"Detected color code: {best_match}")  # Debug print
            return best_match or 'n'  # Default to brown if no good match
            
        except Exception as e:
            print(f"Error in color extraction: {str(e)}")
            return 'n'  # Default to brown on error
    
    def _extract_cap_shape(self, image_tensor, original_image):
        """Extract cap shape from image"""
        # This is a simplified example - would need proper training
        shapes = ['bell', 'conical', 'flat', 'convex']
        return np.random.choice(['b', 'c', 'f', 'x'])

# Usage example:
# analyzer = MushroomImageAnalyzer()
# features = analyzer.analyze_image('mushroom.jpg') 