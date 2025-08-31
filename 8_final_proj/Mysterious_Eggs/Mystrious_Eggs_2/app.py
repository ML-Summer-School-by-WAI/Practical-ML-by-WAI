from flask import Flask, request, jsonify, send_file
import tensorflow as tf
import numpy as np
import io
import base64
from PIL import Image
import cv2
import os
import glob
from datetime import datetime

# Model creation is handled directly in the loading function

app = Flask(__name__)

# Global model variable
loaded_model = None
MODEL_PATH = None
IMG_SIZE = (128, 128)

# Initialize model loading
def initialize_model():
    global loaded_model
    if loaded_model is None:
        print("Initializing model...")
        success, message = load_latest_model()
        if success:
            print(f"✅ {message}")
        else:
            print(f"⚠️  {message}")
        return success
    return True

def load_latest_model():
    global loaded_model, MODEL_PATH
    
    # Look for models in current directory and saved_models directory
    model_patterns = [
        '*.keras',
        'saved_models/*.keras',
        'best_model*.keras',
        'cat_dog_segmentation*.keras'
    ]
    all_models = []
    
    for pattern in model_patterns:
        all_models.extend(glob.glob(pattern))
    
    if not all_models:
        return False, "No trained models found"
    
    # Prefer best_model or cat_dog_segmentation models
    best_models = [m for m in all_models if 'best_model' in m or 'cat_dog_segmentation' in m]
    if best_models:
        latest_model = max(best_models, key=os.path.getctime)
    else:
        latest_model = max(all_models, key=os.path.getctime)
    
    try:
        loaded_model = tf.keras.models.load_model(latest_model)
        MODEL_PATH = latest_model
        return True, f"Model loaded from {latest_model}"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def preprocess_image(image_data):
    # Convert image data to numpy array
    if isinstance(image_data, str):
        # Assume base64 encoded image
        image_data = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_data))
    else:
        image = Image.open(io.BytesIO(image_data))
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize to model input size
    image = image.resize(IMG_SIZE)
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def postprocess_mask(prediction):
    # Convert prediction to mask
    mask = np.argmax(prediction[0], axis=-1)
    
    # Convert to 3-channel image for visualization
    mask_colored = np.zeros((*mask.shape, 3), dtype=np.uint8)
    
    # Get number of classes dynamically  
    num_classes = prediction[0].shape[-1]
    
    # Color mapping: background=black, pet=white
    mask_colored[mask == 0] = [0, 0, 0]      # Background - black
    mask_colored[mask == 1] = [255, 255, 255]  # Pet - white
    if num_classes > 2:  # Only if there's a third class
        mask_colored[mask == 2] = [128, 128, 128]  # Border - gray
    
    return mask_colored

@app.route('/')
@app.route('/health')
def health_check():
    global loaded_model, MODEL_PATH
    
    model_status = "No model loaded"
    if loaded_model is not None:
        model_status = f"Model loaded from {MODEL_PATH}"
    
    return jsonify({
        'status': 'healthy',
        'tensorflow_version': tf.__version__,
        'message': 'Flask API with TensorFlow Segmentation is running!',
        'model_status': model_status,
        'endpoints': {
            '/': 'Health check',
            '/segment': 'POST - Image segmentation',
            '/model/load': 'POST - Load specific model',
            '/model/status': 'GET - Model information',
            '/train/status': 'GET - Training status'
        }
    })

@app.route('/segment', methods=['POST'])
def segment_image():
    global loaded_model
    
    try:
        # Check if model is loaded
        if loaded_model is None:
            if not initialize_model():
                return jsonify({'error': 'Failed to load model'}), 400
        
        # Check if image data is provided
        if 'image' not in request.files and 'image_data' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Handle file upload
        if 'image' in request.files:
            image_file = request.files['image']
            if image_file.filename == '':
                return jsonify({'error': 'No image file selected'}), 400
            
            image_data = image_file.read()
        else:
            # Handle base64 encoded image
            image_data = request.json['image_data']
        
        # Preprocess image
        processed_image = preprocess_image(image_data)
        
        # Make prediction
        prediction = loaded_model.predict(processed_image)
        
        # Postprocess mask
        mask = postprocess_mask(prediction)
        
        # Convert mask to base64 for response
        mask_pil = Image.fromarray(mask)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format='PNG')
        mask_base64 = base64.b64encode(mask_buffer.getvalue()).decode('utf-8')
        
        # Calculate confidence scores dynamically
        num_classes = prediction[0].shape[-1]
        confidence_scores = {
            'background': float(np.mean(prediction[0][:,:,0])),
            'pet': float(np.mean(prediction[0][:,:,1]))
        }
        if num_classes > 2:  # Only if there's a third class
            confidence_scores['border'] = float(np.mean(prediction[0][:,:,2]))
        
        return jsonify({
            'success': True,
            'mask_base64': mask_base64,
            'confidence_scores': confidence_scores,
            'input_shape': processed_image.shape,
            'mask_shape': mask.shape
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/load', methods=['POST'])
def load_model():
    global loaded_model, MODEL_PATH
    
    try:
        data = request.get_json()
        model_path = data.get('model_path')
        
        if not model_path:
            return jsonify({'error': 'No model path provided'}), 400
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 400
        
        loaded_model = tf.keras.models.load_model(model_path)
        MODEL_PATH = model_path
        
        return jsonify({
            'success': True,
            'message': f'Model loaded from {model_path}',
            'model_params': loaded_model.count_params()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/status', methods=['GET'])
def model_status():
    global loaded_model, MODEL_PATH
    
    if loaded_model is None:
        return jsonify({
            'model_loaded': False,
            'message': 'No model currently loaded'
        })
    
    return jsonify({
        'model_loaded': True,
        'model_path': MODEL_PATH,
        'model_params': loaded_model.count_params(),
        'input_shape': str(loaded_model.input_shape),
        'output_shape': str(loaded_model.output_shape)
    })

@app.route('/train/status', methods=['GET'])
def training_status():
    # Check for training artifacts
    keras_models = glob.glob('*.keras')
    saved_models = glob.glob('saved_models/*.keras')
    logs = glob.glob('logs/*')
    
    all_models = keras_models + saved_models
    
    return jsonify({
        'models_available': len(all_models),
        'root_models': len(keras_models),
        'saved_models': len(saved_models),
        'training_logs_available': len(logs),
        'latest_model': max(all_models, key=os.path.getctime) if all_models else None,
        'available_models': all_models
    })

@app.route('/predict', methods=['POST'])
def predict():
    # Keep original endpoint for backward compatibility
    try:
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'No data provided'}), 400
        
        input_data = np.array(data['data'])
        tensor = tf.constant(input_data)
        result = tf.reduce_sum(tensor).numpy()
        
        return jsonify({
            'input': data['data'],
            'result': float(result),
            'shape': list(input_data.shape)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    print("Loading model on startup...")
    initialize_model()
    
    app.run(host='0.0.0.0', port=5001, debug=True)