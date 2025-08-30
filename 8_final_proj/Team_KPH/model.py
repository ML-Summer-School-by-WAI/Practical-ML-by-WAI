import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

class CatAndDogModel:

    def __init__(self):
        self.model_path = os.path.join(os.getcwd(), "models", "dog_cat_classification.keras")
        self.class_path = os.path.join(os.getcwd(), "models", "class_names.json")
        self.input_img_size = (128, 128)
        self.model = None
        self.class_indices = None

    def load_model(self):
        print("Loading model and class indices...")
        try:
            self.model = load_model(self.model_path)

            with open(self.class_path, "r") as f:
                self.class_indices = json.load(f)

            print("Model and class indices loaded successfully.")
        except Exception as e:
            print(f"Error loading model or class indices: {e}")
            return False

        return True

    def preprocess_image(self, img_path):
        img = tf.io.read_file(img_path)
        img = tf.image.decode_png(img, channels=3)
        img = tf.image.resize(img, self.input_img_size)
        img_normalized = tf.cast(img, tf.float32) / 255.0
        img_batch = tf.expand_dims(img_normalized, axis=0)
        return img,img_batch

    def predict(self, img_batch):
        predictions = self.model.predict(img_batch)
        return predictions