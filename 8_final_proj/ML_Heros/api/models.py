import tensorflow as tf
import numpy as np
import cv2

class MLHerosSegmentation:
    def __init__(self, model_path="model/ML_Heros_model.h5"):
        self.model_path = model_path
        self.model = None

    async def load_model(self):
        self.model = tf.keras.models.load_model(self.model_path)
        print("Model loaded!")

    def predict_image(self, image_bytes):
        # Convert bytes to NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Resize to model input size
        img_resized = cv2.resize(img, (128, 128))
        img_input = np.expand_dims(img_resized, axis=0) / 255.0
        
        # Predict segmentation
        pred_mask = self.model.predict(img_input)[0]
        pred_mask = (pred_mask * 255).astype(np.uint8)
        pred_mask = cv2.resize(pred_mask, (img.shape[1], img.shape[0]))
        return pred_mask
