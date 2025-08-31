import os
import json
import numpy as np
from PIL import Image, ImageDraw

# ---------- CONFIG ----------
json_folder = r"/home/koder/Documents/Segmentation Process/Dog_JSON"  # Folder with JSON files
image_folder = r"/home/koder/Documents/Segmentation Process/Dogs"   # Folder with original JPEGs
output_mask_folder = r"/home/koder/Documents/Segmentation Process/VOC/encoded_mask"  # Where masks will be saved

class_mapping = {
    "Dog": 1,
    "background": 0
}

# Make output folder if not exists
os.makedirs(output_mask_folder, exist_ok=True)

# ---------- PROCESS ----------
for filename in os.listdir(json_folder):
    if filename.endswith(".json"):
        json_path = os.path.join(json_folder, filename)
        with open(json_path, "r") as f:
            data = json.load(f)
        
        # Get image size
        width = data["imageWidth"]
        height = data["imageHeight"]

        # Create empty mask (all background)
        mask = Image.new("L", (width, height), class_mapping["background"])
        draw = ImageDraw.Draw(mask)

        # Draw each labeled shape
        for shape in data["shapes"]:
            label = shape["label"]
            points = shape["points"]
            
            # Convert points to tuple format
            polygon = [tuple(point) for point in points]
            
            # Fill polygon with class value
            if label in class_mapping:
                draw.polygon(polygon, fill=class_mapping[label])
            else:
                print(f"[WARN] {filename}: label '{label}' not in class_mapping")
        
        # Save mask
        mask_name = os.path.splitext(filename)[0] + ".jpg"
        mask.save(os.path.join(output_mask_folder, mask_name))
        print(f"[DONE] {filename} -> {mask_name}")
