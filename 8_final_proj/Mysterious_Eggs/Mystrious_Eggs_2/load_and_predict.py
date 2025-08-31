#!/usr/bin/env python3
"""
Load saved model and make predictions
"""

import os
import cv2
import numpy as np
import json

def load_saved_model(model_path):
    """
    Load a saved TensorFlow model
    """
    try:
        import tensorflow as tf
        print(f"Loading model from: {model_path}")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully!")
        return model
    except ImportError:
        print("❌ TensorFlow not installed. Install with: pip install tensorflow")
        return None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def preprocess_image(image_path, target_size=(128, 128)):
    """
    Preprocess image for prediction
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Could not load image: {image_path}")
        return None, None
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image_resized = cv2.resize(image_rgb, target_size)
    
    # Normalize
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch, image_rgb

def predict_mask(model, image_batch):
    """
    Predict segmentation mask
    """
    try:
        # Make prediction
        prediction = model.predict(image_batch, verbose=0)
        
        # Convert to mask (take argmax for multi-class)
        mask = np.argmax(prediction[0], axis=-1)
        
        return mask
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return None

def save_results(original_image, predicted_mask, output_prefix):
    """
    Save prediction results
    """
    os.makedirs('predictions', exist_ok=True)
    
    # Resize mask to original image size
    mask_resized = cv2.resize(predicted_mask.astype(np.uint8), 
                             (original_image.shape[1], original_image.shape[0]), 
                             interpolation=cv2.INTER_NEAREST)
    
    # Save original
    original_path = f'predictions/{output_prefix}_original.png'
    cv2.imwrite(original_path, cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR))
    
    # Save mask
    mask_path = f'predictions/{output_prefix}_mask.png'
    cv2.imwrite(mask_path, mask_resized * 255)  # Scale to 0-255
    
    # Create colored overlay
    mask_colored = np.zeros_like(original_image)
    mask_colored[:, :, 0] = mask_resized * 255  # Red channel for animals
    
    # Blend with original
    overlay = cv2.addWeighted(original_image, 0.7, mask_colored, 0.3, 0)
    overlay_path = f'predictions/{output_prefix}_overlay.png'
    cv2.imwrite(overlay_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    
    print(f"✅ Results saved:")
    print(f"  - {original_path}")
    print(f"  - {mask_path}")
    print(f"  - {overlay_path}")
    
    # Calculate statistics
    total_pixels = mask_resized.size
    animal_pixels = np.sum(mask_resized)
    animal_percentage = (animal_pixels / total_pixels) * 100
    
    print(f"📊 Prediction stats:")
    print(f"  - Total pixels: {total_pixels:,}")
    print(f"  - Animal pixels: {animal_pixels:,}")
    print(f"  - Animal coverage: {animal_percentage:.1f}%")

def main():
    """
    Main prediction function
    """
    print("=== Model Prediction Demo ===")
    
    # Configuration
    model_path = 'saved_models/cat_dog_segmentation.keras'
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        print("First train a model using: python train_and_save_model.py")
        return
    
    # Load model
    model = load_saved_model(model_path)
    if model is None:
        return
    
    # Load model info if available
    info_path = 'saved_models/model_info.json'
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        print(f"📋 Model info: {model_info.get('model_architecture', 'Unknown')}")
        print(f"📊 Trained on: {model_info.get('dataset_info', {}).get('training_samples', 'Unknown')} samples")
    
    # Test with sample images
    test_images = []
    
    # Look for test images in various locations
    test_dirs = [
        'dataset/cat_and_dog_dataset/cats',
        'dataset/cat_and_dog_dataset/dogs',
        '.'
    ]
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            jpg_files = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg')]
            for jpg_file in jpg_files[:3]:  # Take first 3
                test_images.append(os.path.join(test_dir, jpg_file))
            if test_images:
                break
    
    if not test_images:
        print("❌ No test images found. Place some .jpg files in the current directory.")
        return
    
    print(f"🧪 Testing on {len(test_images)} images...")
    
    # Process each test image
    for i, image_path in enumerate(test_images):
        print(f"\n📸 Processing: {os.path.basename(image_path)}")
        
        # Preprocess
        image_batch, original_image = preprocess_image(image_path)
        if image_batch is None:
            continue
        
        # Predict
        predicted_mask = predict_mask(model, image_batch)
        if predicted_mask is None:
            continue
        
        # Save results
        output_prefix = f"test_{i+1}_{os.path.splitext(os.path.basename(image_path))[0]}"
        save_results(original_image, predicted_mask, output_prefix)
    
    print(f"\n🎉 Prediction completed! Check the 'predictions/' folder for results.")

if __name__ == "__main__":
    main()