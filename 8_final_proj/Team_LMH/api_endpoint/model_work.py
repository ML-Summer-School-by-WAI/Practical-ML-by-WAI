import tensorflow as tf
import numpy as np
import os
import json

class CatDogModel:
    def __init__(self, model_path: str, class_names_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        self.model = tf.keras.models.load_model(model_path)

        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names file not found: {class_names_path}")
        with open(class_names_path, "r") as f:
            self.class_names = json.load(f)

    def predict(self, img_array):
        # Add batch dimension
        preds = self.model.predict(np.expand_dims(img_array, axis=0))  
        # preds shape: (1, 128, 128, 3)

        # Take argmax across channels -> per-pixel class index
        mask = np.argmax(preds, axis=-1)[0]  # shape: (128,128)

        # Count pixels per class
        unique, counts = np.unique(mask, return_counts=True)
        class_distribution = dict(zip(unique, counts))

        # Pick majority class
        class_index = int(max(class_distribution, key=class_distribution.get))
        confidence = class_distribution[class_index] / (128*128)

        return {
            "class": self.class_names[class_index],
            "confidence": float(confidence)
        }
