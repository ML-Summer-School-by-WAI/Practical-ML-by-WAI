# model_loader.py

import os
import tensorflow as tf

# Define the number of classes. It should be 3 for "Background", "Dog", and "Cat".
NUM_CLASSES = 3

# IMPORTANT: Replace this with the path to your trained Keras model file (.h5 or .keras)
MODEL_PATH = "/home/koder/Documents/Segmentation Process/cat_and_dog.keras"

def load_segmentation_model():
    """
    Loads the segmentation model from the specified path.
    Raises FileNotFoundError if the model file does not exist.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        # Optional: Check that the model's output shape matches expectations
        if model.output_shape[-1] != NUM_CLASSES:
            print(f"Warning: Model output classes ({model.output_shape[-1]}) do not match NUM_CLASSES ({NUM_CLASSES}).")
        
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        # In a real-world scenario, you might want to re-raise a more specific exception
        raise RuntimeError("Failed to load the segmentation model.") from e

# Initialize a global variable to store the loaded model.
# This variable will be populated by the startup event in main.py.
segmentation_model = None