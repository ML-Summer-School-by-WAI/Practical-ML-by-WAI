import tensorflow as tf
import numpy as np
import cv2
from typing import Union
import io
from PIL import Image


class SemanticSegmentation:
    """
    Semantic segmentation model for cats and dogs using U-Net architecture.
    """
    
    def __init__(self):
        self.model = None
        self.img_height = 128
        self.img_width = 128
        self.color_map = np.array([
            [0, 0, 0],       # Class 0: background → black
            [0, 255, 0],     # Class 1: cat → green
            [255, 0, 0],     # Class 2: dog → red
        ], dtype=np.uint8)
    
    def load_model(self, model_path: str):
        """
        Load the trained model from the specified path.
        
        Args:
            model_path (str): Path to the model file (.h5 or .keras)
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load model from {model_path}: {str(e)}")
    
    def preprocess_image(self, image_bytes: bytes) -> tuple:
        """
        Preprocess the input image bytes for model prediction.
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            tuple: (preprocessed_image_tensor, original_image_tensor)
        """
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to tensor and resize
        img_tensor = tf.convert_to_tensor(img_array)
        img_resized = tf.image.resize(img_tensor, [self.img_height, self.img_width])
        
        # Normalize pixel values to [0, 1]
        img_normalized = tf.cast(img_resized, tf.float32) / 255.0
        
        # Add batch dimension
        img_batch = tf.expand_dims(img_normalized, axis=0)
        
        return img_batch, img_resized
    
    def predict(self, image_bytes: bytes) -> np.ndarray:
        """
        Make a prediction on the input image.
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            np.ndarray: Predicted mask as numpy array
        """
        if self.model is None:
            raise Exception("Model not loaded. Please call load_model() first.")
        
        # Preprocess the image
        img_batch, _ = self.preprocess_image(image_bytes)
        
        # Make prediction
        predicted_masks = self.model.predict(img_batch)
        
        # Get the class with highest probability for each pixel
        pred_mask = tf.argmax(predicted_masks, axis=-1)[0].numpy()
        
        return pred_mask
    
    def mask_to_rgb(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert a grayscale mask to RGB using the color map.
        
        Args:
            mask (np.ndarray): Grayscale mask with class indices
            
        Returns:
            np.ndarray: RGB colored mask
        """
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.color_map):
            rgb_mask[mask == class_idx] = color
        return rgb_mask
    
    def predict_and_visualize_color(self, image_bytes: bytes) -> np.ndarray:
        """
        Predict segmentation and create a colored overlay visualization.
        
        Args:
            image_bytes (bytes): Raw image bytes
            
        Returns:
            np.ndarray: Overlay image as numpy array (BGR format for OpenCV)
        """
        if self.model is None:
            raise Exception("Model not loaded. Please call load_model() first.")
        
        # Preprocess the image
        img_batch, img_resized = self.preprocess_image(image_bytes)
        
        # Make prediction
        predicted_masks = self.model.predict(img_batch)
        pred_mask = tf.argmax(predicted_masks, axis=-1)[0].numpy()
        
        # Convert mask to color
        color_mask = self.mask_to_rgb(pred_mask)
        
        # Convert input image tensor to numpy array
        input_image_numpy = tf.keras.utils.img_to_array(img_resized, dtype=np.uint8)
        
        # Blend original image and mask overlay
        overlay = cv2.addWeighted(input_image_numpy, 0.6, color_mask, 0.4, 0)
        
        # Convert RGB to BGR for OpenCV compatibility
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        
        return overlay_bgr
