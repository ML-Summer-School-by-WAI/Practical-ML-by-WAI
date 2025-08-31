import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

COLOR_MAP = np.array([
    [0, 0, 0],       # Class 0: background
    [0, 255, 0],     # Class 1: green stays green
    [0, 0, 255],     # Class 2: red in BGR
], dtype=np.uint8)

class ImageModel():
    def __init__(self):
        # Docker path (model copied to /app)
        self.model_path = os.path.join(os.getcwd(), "cat_dog_segmentation_unet.keras")
        print(f"[DEBUG] Model path set to: {self.model_path}")
        self.IMG_HEIGHT = 128
        self.IMG_WIDTH = 128
        self.model = None

    def load_model(self):
        print(f"Loading model from: {self.model_path}")
        try:
            self.model = load_model(self.model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
            return False
        return True

    def model_predict(self, img):
        if self.model is None:
            raise ValueError("Model not loaded")

        # If img is bytes (uploaded), decode to numpy first
        if isinstance(img, (bytes, bytearray)):
            img_array = np.frombuffer(img, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        img_resized = cv2.resize(img, (self.IMG_WIDTH, self.IMG_HEIGHT))
        img_normalized = img_resized.astype(np.float32) / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        predicted_masks = self.model.predict(img_batch)
        return predicted_masks, img

    def mask_to_rgb(self, mask, color_map):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(color_map):
            rgb_mask[mask == class_idx] = color
        return rgb_mask

    def predict_and_overlay(self, img):
        predicted_masks, img_resized = self.model_predict(img)
        pred_mask = np.argmax(predicted_masks, axis=-1)[0]

        color_mask = self.mask_to_rgb(pred_mask, COLOR_MAP)

        # Ensure img_resized is same shape as mask
        img_resized_rgb = cv2.resize(img_resized, (color_mask.shape[1], color_mask.shape[0]))

        overlay = cv2.addWeighted(img_resized_rgb.astype(np.uint8), 0.6, color_mask, 0.4, 0)
        return overlay