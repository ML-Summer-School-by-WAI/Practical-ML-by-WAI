"""
Overlay Model Working Module with Multi-Class Support
"""

import io
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


class SemanticSegmentation:
    """Semantic Segmentation for Cat and Dog (multi-class overlay)"""

    def __init__(self, model=None, image_height=128, image_width=128):
        self.__model = model
        self.image_height = image_height
        self.image_width = image_width

        # Fixed color map (Dog = Red, Cat = Green, Background = Black)
        self.class_labels = {
            0: ("Background", (0, 0, 0)),   # Black
            1: ("Dog", (255, 0, 0)),        # Red
            2: ("Cat", (0, 255, 0)),        # Green
        }

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def load_model(self, model_path):
        self.__model = tf.keras.models.load_model(model_path)
        return self.__model

    def preprocess_image(self, image_bytes):
        """Load and resize image"""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = image.resize((self.image_width, self.image_height))
        image_array = np.array(image) / 255.0
        return image, np.expand_dims(image_array, axis=0)

    def mask_to_rgb(self, mask):
        """Convert class indices to RGB mask"""
        rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for class_idx, (_, color) in self.class_labels.items():
            rgb_mask[mask == class_idx] = color
        return rgb_mask

    def predict(self, image_bytes):
        """Predict segmentation mask"""
        img_pil, img_batch = self.preprocess_image(image_bytes)
        predictions = self.__model.predict(img_batch)
        mask = tf.argmax(predictions, axis=-1)[0].numpy()
        rgb_mask = self.mask_to_rgb(mask)

        # Overlay mask on original image
        overlay = Image.blend(img_pil, Image.fromarray(rgb_mask), alpha=0.4)

        return img_pil, rgb_mask, overlay, mask

    def generate_legend(self):
        """Create a legend image showing class → color mapping"""
        legend_height = 30 * len(self.class_labels)
        legend = Image.new("RGB", (200, legend_height), (255, 255, 255))
        draw = ImageDraw.Draw(legend)

        y = 0
        for _, (label, color) in self.class_labels.items():
            draw.rectangle([10, y + 5, 30, y + 25], fill=color)
            draw.text((40, y + 5), label, fill=(0, 0, 0))
            y += 30

        return legend
