import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt

def create_color_to_label_map(mask_folder_path):
    """
    Scans all masks in a folder to find unique colors and create a mapping
    from color to a class label.
    For binary masks: 0 (black) = background, 255 (white) = foreground
    """
    print(f"Scanning masks in '{mask_folder_path}'...")
    
    unique_colors = set()
    
    mask_files = [f for f in os.listdir(mask_folder_path) if f.endswith('.png')]
    print(f"Found {len(mask_files)} mask files")
   
    for filename in tqdm(mask_files[:5], desc="Finding unique colors"):  # Sample first 5 files
        mask_path = os.path.join(mask_folder_path, filename)
        
        # Read as grayscale since our masks are binary
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Get unique pixel values
        unique_values = np.unique(mask)
        for value in unique_values:
            unique_colors.add(value)
            
    sorted_colors = sorted(list(unique_colors))
    
    # Create the color-to-label dictionary
    # Background (0) -> 0, Foreground (255) -> 1
    color_to_label = {}
    for i, color in enumerate(sorted_colors):
        if color == 0:
            color_to_label[color] = 0  # Background
        else:
            color_to_label[color] = 1  # Foreground (cat/dog)
    
    print("\nScan complete!")
    print(f"Found {len(color_to_label)} unique classes: {color_to_label}")
    
    return color_to_label


mask_folder_path = "dataset/cat_and_dog_dataset/SegmentationClass"
COLOR_TO_LABEL = create_color_to_label_map(mask_folder_path)

def encode_mask_to_grayscale(mask_path, color_map):
    """
    Converts a binary segmentation mask to encoded grayscale mask with class labels.
    """
    # Read as grayscale since our masks are already binary
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
        
    height, width = mask.shape
    
    # Create encoded mask
    encoded_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Map pixel values to class labels
    for pixel_value, class_label in color_map.items():
        encoded_mask[mask == pixel_value] = class_label
        
    return encoded_mask

def batch_encode_masks(input_folder, output_folder, color_map):
    """
    Encode all masks in a folder
    """
    os.makedirs(output_folder, exist_ok=True)
    
    mask_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
    print(f"Encoding {len(mask_files)} masks...")
    
    for filename in tqdm(mask_files, desc="Encoding masks"):
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace('.png', '.png')  # Keep as PNG
        output_path = os.path.join(output_folder, output_filename)
        
        encoded_mask = encode_mask_to_grayscale(input_path, color_map)
        if encoded_mask is not None:
            cv2.imwrite(output_path, encoded_mask)
    
    print(f"All masks encoded and saved to '{output_folder}'")

# Batch encode all masks
input_folder = "dataset/cat_and_dog_dataset/SegmentationClass"
output_folder = "dataset/cat_and_dog_dataset/encoded_masks"
batch_encode_masks(input_folder, output_folder, COLOR_TO_LABEL)

# Test with one mask
test_files = [f for f in os.listdir(input_folder) if f.endswith('.png')]
if test_files:
    test_mask_path = os.path.join(input_folder, test_files[0])
    encoded_mask = encode_mask_to_grayscale(test_mask_path, COLOR_TO_LABEL)
    
    print(f"Testing with: {test_files[0]}")
    print(f"Original mask shape: {encoded_mask.shape if encoded_mask is not None else 'None'}")
    print(f"Unique values in encoded mask: {np.unique(encoded_mask) if encoded_mask is not None else 'None'}")
    print(f"Class mapping: {COLOR_TO_LABEL}")