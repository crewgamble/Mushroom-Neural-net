from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import torch
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
import os
from werkzeug.utils import secure_filename
from image_analyzer import MushroomImageAnalyzer
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Simple CORS configuration
CORS(app)

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

@app.route('/api/predict', methods=['OPTIONS'])
def handle_options():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type, Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.add("Access-Control-Allow-Credentials", "true")
    return response

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize image analyzer
try:
    logger.info("Initializing MushroomImageAnalyzer...")
    image_analyzer = MushroomImageAnalyzer()
    logger.info("MushroomImageAnalyzer initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize MushroomImageAnalyzer: {str(e)}")
    logger.error(f"Current working directory: {os.getcwd()}")
    logger.error(f"Files in current directory: {os.listdir('.')}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Load and preprocess the dataset for feature names
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
X_encoded = pd.get_dummies(X)
feature_names = X_encoded.columns.tolist()

# Define the neural network architecture (same as in mushroom_classifier.py)
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, num_classes),
            torch.nn.LogSoftmax(dim=1)
        )
    
    def forward(self, x):
        return self.layers(x)

# Initialize model with same parameters
input_size = len(feature_names)
hidden_size = 100
num_classes = 2
model = NeuralNet(input_size, hidden_size, num_classes)

# Load the trained model weights
model_path = 'model.pth'
if os.path.exists(model_path):
    # Load the model weights with map_location to handle CPU environment
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
else:
    print("Warning: No trained model found. Please train the model first.")

@app.route('/')
def root():
    return jsonify({'status': 'healthy', 'message': 'Mushroom Classifier API is running'})

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print("Received data:", data)  # Debug print
        
        # Create a DataFrame with all possible features set to 0
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Required features (most important for classification)
        required_features = [
            'odor',           # Strong indicator of toxicity
            'spore-print-color', # Important taxonomic feature
            'gill-color',     # Key visual identifier
            'cap-color',      # Primary visual feature
            'bruises',        # Important toxicity indicator
            'ring-type',      # Key taxonomic feature
            'gill-spacing',   # Important morphological feature
            'cap-shape',      # Basic visual identifier
            'population',     # Environmental indicator
            'habitat',        # Growth context
            'stalk-surface-above-ring', # Texture indicator
            'cap-surface'     # Surface characteristic
        ]
        
        # Check if all required features are present
        missing_features = [f for f in required_features if not data.get(f)]
        if missing_features:
            return jsonify({
                'error': f'Missing required features: {", ".join(missing_features)}'
            }), 400
        
        # Set the features that are present in the input
        for feature, value in data.items():
            if value:  # Process all provided features, required or not
                feature_col = f"{feature}_{value}"
                print(f"Processing feature: {feature_col}")  # Debug print
                if feature_col in feature_names:
                    input_data[feature_col] = 1
                else:
                    print(f"Warning: Feature column {feature_col} not found in feature_names")  # Debug print
        
        print("Input data:", input_data)  # Debug print
        
        # Convert to tensor and get prediction
        with torch.no_grad():
            input_tensor = torch.FloatTensor(input_data.values)
            output = model(input_tensor)
            probabilities = torch.exp(output)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            print(f"Prediction: {prediction}, Confidence: {confidence}")  # Debug print
        
        return jsonify({
            'prediction': 'edible' if prediction == 0 else 'poisonous',
            'confidence': confidence
        })
    
    except Exception as e:
        print("Error:", str(e))  # Debug print
        return jsonify({'error': str(e)}), 400

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    try:
        logger.info("Received image analysis request")
        # Check if image file is present in request
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            # Save the file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Analyze the image
            print(f"Analyzing image: {filepath}")
            features = image_analyzer.analyze_image(filepath)
            
            # Clean up - delete the uploaded file
            os.remove(filepath)
            
            print(f"Extracted features: {features}")
            return jsonify({'features': features})
            
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        print(f"Error in image analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port) 