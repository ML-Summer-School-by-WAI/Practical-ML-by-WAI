#!/usr/bin/env python3
"""
Simple test of data loading without visualization
"""

import os
import glob

# Dataset paths
DATASET_BASE = 'dataset/cat_and_dog_dataset'
CATS_DIR = os.path.join(DATASET_BASE, 'cats')
DOGS_DIR = os.path.join(DATASET_BASE, 'dogs')
ENCODED_MASKS_DIR = os.path.join(DATASET_BASE, 'encoded_masks')

def main():
    print("=== Simple Data Loading Test ===")
    
    # Check directories
    print(f"Cats directory exists: {os.path.exists(CATS_DIR)}")
    print(f"Dogs directory exists: {os.path.exists(DOGS_DIR)}")
    print(f"Encoded masks directory exists: {os.path.exists(ENCODED_MASKS_DIR)}")
    
    # Get all JPG images from cats directory and subdirectories
    cat_images = glob.glob(os.path.join(CATS_DIR, '*.jpg'))
    cat_images.extend(glob.glob(os.path.join(CATS_DIR, 'cat', '*.jpg')))
    
    # Get all JPG images from dogs directory and subdirectories
    dog_images = glob.glob(os.path.join(DOGS_DIR, '*.jpg'))
    dog_images.extend(glob.glob(os.path.join(DOGS_DIR, 'dog', '*.jpg')))
    
    all_images = cat_images + dog_images
    
    print(f"Found {len(cat_images)} cat images")
    print(f"Found {len(dog_images)} dog images")
    print(f"Total images: {len(all_images)}")
    
    # Count matching masks
    matched_pairs = 0
    unmatched = []
    
    for img_path in all_images[:10]:  # Test first 10
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Look for corresponding mask
        mask_name = base_name + '.png'
        mask_path = os.path.join(ENCODED_MASKS_DIR, mask_name)
        
        if os.path.exists(mask_path):
            matched_pairs += 1
            print(f"✓ {img_name} -> {mask_name}")
        else:
            unmatched.append(img_name)
            print(f"✗ {img_name} (no mask found)")
    
    print(f"\nMatched pairs (from first 10): {matched_pairs}")
    print(f"Unmatched: {len(unmatched)}")
    
    if unmatched:
        print("Unmatched files:", unmatched)

if __name__ == "__main__":
    main()