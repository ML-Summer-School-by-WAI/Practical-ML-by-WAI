from tensorflow.keras.models import load_model
import os
import tensorflow as tf
import numpy as np
import cv2

class SegmentationModel:
    
    def __init__(self):
        self.model_path = os.getcwd() + "/models/cat_and_dog_unet.keras"
        self.input_img_size = (128,128)  # Assuming the model expects 224x224 images
        self.model = None
        
        self.color_map = np.array([
        [0, 0, 0],       # Class 0: background → black
        [0, 255, 0],     # Class 1: cat → green
        [255, 0, 0],     # Class 2: dog → red
    ], dtype=np.uint8)

    async def load_model(self):
        print("Loading model...")
        try:
            self.model = load_model(self.model_path)

            print("Model and class indices loaded successfully.")
            print(type(self.model))
        except Exception as e:
            print(f"Error loading model or class indices: {e}")
            return False
 
        return True
        
    def preprocess_image(self, image_bytes):
        # img = tf.io.decode_image(image_bytes)
        img = tf.image.decode_png(image_bytes, channels=3)
        img = tf.image.resize(img, self.input_img_size)
        img_normalized  = tf.cast(img, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def mask_to_rgb(self, mask):
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.color_map):
            rgb_mask[mask == class_idx] = color
        return rgb_mask
    


    def predict_image(self, image_bytes):
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        preprocessed_image = self.preprocess_image(image_bytes)
        print("<--- Image preprocessed --->")
            
        predicted_masks =self.model.predict(preprocessed_image)
        print("<--- Prediction done --->")
        

        pred_mask = tf.argmax(predicted_masks, axis=-1)[0].numpy()
        rgb_mask = self.mask_to_rgb(pred_mask)
        
                # Ensure mask is same size as original image
        if rgb_mask.shape[:2] != original_img.shape[:2]:
            rgb_mask = cv2.resize(rgb_mask, (original_img.shape[1], original_img.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            
        overlayed = cv2.addWeighted(original_img, 0.5, rgb_mask, 0.5, 0)
        
        
        print("<--- Overlay done --->")
        
        return overlayed
        
        
        
        
        
        
