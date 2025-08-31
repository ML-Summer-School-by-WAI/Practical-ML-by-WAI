"""
Overlay Model Working Module
"""

import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


class SemanticSegmentation:
    """Model Loading Class for Semantic Segmentation on Cat and Dog"""

    def __init__(self, model=None, image_height=512, image_width=512):
        """Model Loading Object for Semantic Segmentation on Cat and Dog"""
        self.__model = model
        self.image_path = None
        self.image_height = image_height
        self.image_width = image_width
        self.color_map = np.array(
            [
                [0, 0, 0],  # Class 0: Black
                [255, 0, 0],  # Class 1: Red
                [0, 255, 0],  # Class 2: Green
            ],
            dtype=np.uint8,
        )

    @property
    def model(self):
        """
        Get the currently loaded semantic segmentation model.

        Returns:
            tensorflow.keras.Model | None: The loaded model instance,
            or None if no model has been loaded.
        """
        return self.__model

    @model.setter
    def model(self, model):
        """
        Set the semantic segmentation model manually.

        Args:
            model (tensorflow.keras.Model): A preloaded Keras model
            to assign as the semantic segmentation model.
        """
        self.__model = model

    def load_model(self, model_path):
        """Load a semantic segmentation model from a given file path.

        Args:
            model_path (str): Path to the saved Keras model file (e.g., .h5 or .keras).

        Returns:
            tensorflow.keras.Model: The loaded Keras model.
        """
        self.__model = tf.keras.models.load_model(model_path)
        return self.__model

    def semantic_segmentation(self, image_path=None):
        """
        Run semantic segmentation on an input image.

        Args:
            image_path (bytes | None): Raw image bytes (e.g., from an uploaded file).
                                        If None, no prediction is performed.

        Returns:
            tuple:
                - predicted_masks (numpy.ndarray): Model's raw predicted mask array.
                - image (PIL.Image.Image): The preprocessed input image (resized to 128x128).
        """
        if image_path is not None:
            self.image_path = image_path
            image = Image.open(io.BytesIO(self.image_path))
            image = image.resize((128, 128))
            image_array = np.array(image) / 255.0
            img_batch = tf.expand_dims(image_array, axis=0)
            predicted_masks = self.__model.predict(img_batch)
            return predicted_masks, image
        return None, None

    def mask_to_rgb(self, mask):
        """
        Convert a segmentation mask into a color RGB image.

        Args:
            mask (numpy.ndarray): 2D array with class indices for each pixel.

        Returns:
            numpy.ndarray: 3D RGB mask (H, W, 3) with colors applied per class.
        """
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, color in enumerate(self.color_map):
            rgb_mask[mask == class_idx] = color
        return rgb_mask

    def predict_and_visualize_color(self, image_path=None, overlay_alpha=0.4):
        """
        Predict and overlay the segmentation mask on the input image.

        Args:
            image_bytes (bytes | None): Raw image bytes of the input.
            overlay_alpha (float): Transparency factor for overlay
                                   (0.0 = only input image, 1.0 = only mask).

        Returns:
            numpy.ndarray: RGB image with segmentation mask overlay.
        """
        predicted_masks, img = self.semantic_segmentation(image_path)
        pred_mask = tf.argmax(predicted_masks, axis=-1)[0].numpy()
        color_mask = self.mask_to_rgb(pred_mask)
        input_image_numpy = np.array(img).astype(np.uint8)

        overlay = cv2.addWeighted(
            input_image_numpy, 1 - overlay_alpha, color_mask, overlay_alpha, 0
        )
        return overlay
