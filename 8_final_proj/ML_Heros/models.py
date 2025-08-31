import os
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2

class SegmentationModel:
    def __init__(self, json_file="cat&dog.json", weights_file="cat&dogs_weights.h5"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.json_path = os.path.join(current_dir, "models", json_file)
        self.weights_path = os.path.join(current_dir, "models", weights_file)

        self.model = None

        self.color_map = np.array([
            [0, 0, 0],       # Class 0: background → black
            [0, 0, 255],     # Class 1: cat → blue
            [255, 0, 0],     # Class 2: dog → red
        ], dtype=np.uint8)

    async def load_model(self):
        """
        Asynchronously loads the model architecture from a JSON file and weights from an H5 file.
        """
        # Load JSON architecture
        if not os.path.exists(self.json_path):
            raise FileNotFoundError(f"JSON file not found: {self.json_path}")
        
        with open(self.json_path, "r") as f:
            json_model = f.read()

        self.model = model_from_json(json_model)

        # Load weights
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"Weights file not found: {self.weights_path}")
        
        self.model.load_weights(self.weights_path)
        print(f"Model loaded from {self.json_path} with weights {self.weights_path}")

    def predict_image(self, image_bytes):
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            print("Error: Failed to decode image bytes. The input file may be corrupted or not a valid image.")
            return None

        img_resized = cv2.resize(img, (128, 128))
        img_input = np.expand_dims(img_resized, axis=0) / 255.0
        pred_mask = self.model.predict(img_input)[0]
        pred_class_map = np.argmax(pred_mask, axis=-1)
        output_mask = self.color_map[pred_class_map]
        output_mask = cv2.resize(output_mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        return output_mask