
import numpy as np
import os
import json
from tensorflow.keras.models import load_model

class CatAndDogModel:

    def __init__(self):
        self.model_path = os.getcwd() + "/cat_dog_unet.keras"
        self.input_img_size = (128,128) 
        self.model = None

    def load_model(self):
        print("Loading model and class indices...")
        try:
            self.model = load_model(self.model_path)
    
            print("Model loaded successfully.")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
 
        return True
