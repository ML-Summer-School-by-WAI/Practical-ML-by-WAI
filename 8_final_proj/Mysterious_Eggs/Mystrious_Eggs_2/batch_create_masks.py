#!/usr/bin/env python3
import json
import numpy as np
from PIL import Image, ImageDraw
import os
import glob
from pathlib import Path

def create_segmentation_mask(json_path, output_dir):
    """
    Create segmentation mask from LabelMe JSON annotation
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get image dimensions
        img_height = data['imageHeight']
        img_width = data['imageWidth']
        
        # Create blank mask
        mask = Image.new('L', (img_width, img_height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Process each shape
        for shape in data['shapes']:
            label = shape['label'].lower()
            if label in ['dog', 'cat']:
                # Convert points to list of tuples
                points = [(point[0], point[1]) for point in shape['points']]
                # Fill polygon with value 255 (foreground)
                draw.polygon(points, fill=255)
        
        # Save mask
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        mask_path = os.path.join(output_dir, f"{base_name}.png")
        
        mask.save(mask_path)
        print(f"✓ Created: {mask_path}")
        return mask_path
        
    except Exception as e:
        print(f"✗ Error processing {json_path}: {e}")
        return None

def process_all_annotations(base_dir):
    """
    Process all JSON annotations in the dataset
    """
    # Define paths
    dog_dir = os.path.join(base_dir, "dogs", "dog")
    cat_dir = os.path.join(base_dir, "cats", "cat")
    output_dir = os.path.join(base_dir, "SegmentationClass")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files
    dog_jsons = glob.glob(os.path.join(dog_dir, "*.json"))
    cat_jsons = glob.glob(os.path.join(cat_dir, "*.json"))
    
    print(f"Found {len(dog_jsons)} dog annotations")
    print(f"Found {len(cat_jsons)} cat annotations")
    print(f"Total: {len(dog_jsons) + len(cat_jsons)} annotations to process")
    print("-" * 50)
    
    # Process dog annotations
    print("Processing dog annotations...")
    dog_success = 0
    for json_path in sorted(dog_jsons):
        if create_segmentation_mask(json_path, output_dir):
            dog_success += 1
    
    print(f"Dogs processed: {dog_success}/{len(dog_jsons)}")
    print("-" * 50)
    
    # Process cat annotations
    print("Processing cat annotations...")
    cat_success = 0
    for json_path in sorted(cat_jsons):
        if create_segmentation_mask(json_path, output_dir):
            cat_success += 1
    
    print(f"Cats processed: {cat_success}/{len(cat_jsons)}")
    print("-" * 50)
    
    total_success = dog_success + cat_success
    total_files = len(dog_jsons) + len(cat_jsons)
    
    print(f"Summary:")
    print(f"✓ Successfully processed: {total_success}/{total_files}")
    print(f"✗ Failed: {total_files - total_success}/{total_files}")
    print(f"Output directory: {output_dir}")

def main():
    base_dir = "dataset/cat_and_dog_dataset"
    
    if not os.path.exists(base_dir):
        print(f"Error: Dataset directory '{base_dir}' not found!")
        return
    
    process_all_annotations(base_dir)

if __name__ == "__main__":
    main()