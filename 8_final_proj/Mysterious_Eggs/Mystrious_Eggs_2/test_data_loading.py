#!/usr/bin/env python3
"""
Test data loading for cat and dog segmentation
"""

import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Dataset paths
DATASET_BASE = 'dataset/cat_and_dog_dataset'
CATS_DIR = os.path.join(DATASET_BASE, 'cats')
DOGS_DIR = os.path.join(DATASET_BASE, 'dogs')
ENCODED_MASKS_DIR = os.path.join(DATASET_BASE, 'encoded_masks')

def collect_image_mask_pairs():
    """
    Collect image and mask file pairs from the dataset
    """
    image_paths = []
    mask_paths = []
    
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
    
    # Match images with their corresponding encoded masks
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        
        # Look for corresponding mask
        mask_name = base_name + '.png'
        mask_path = os.path.join(ENCODED_MASKS_DIR, mask_name)
        
        if os.path.exists(mask_path):
            image_paths.append(img_path)
            mask_paths.append(mask_path)
    
    print(f"Found {len(image_paths)} image-mask pairs")
    return image_paths, mask_paths

def visualize_sample_pairs(image_paths, mask_paths, num_samples=3):
    """
    Visualize some sample image-mask pairs
    """
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
    
    for i in range(min(num_samples, len(image_paths))):
        # Load image
        img = cv2.imread(image_paths[i])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = cv2.imread(mask_paths[i], cv2.IMREAD_GRAYSCALE)
        
        # Display original image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image\n{os.path.basename(image_paths[i])}')
        axes[i, 0].axis('off')
        
        # Display mask
        axes[i, 1].imshow(mask, cmap='viridis')
        axes[i, 1].set_title(f'Encoded Mask\nValues: {np.unique(mask)}')
        axes[i, 1].axis('off')
        
        # Display overlay
        mask_colored = np.zeros_like(img)
        mask_colored[:, :, 0] = mask * 255  # Red channel for mask
        overlay = cv2.addWeighted(img, 0.7, mask_colored, 0.3, 0)
        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title('Overlay')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_samples.png', dpi=150, bbox_inches='tight')
    plt.show()

def main():
    """
    Main function to test data loading
    """
    print("=== Testing Data Loading ===")
    
    # Check if directories exist
    print(f"Cats directory exists: {os.path.exists(CATS_DIR)}")
    print(f"Dogs directory exists: {os.path.exists(DOGS_DIR)}")
    print(f"Encoded masks directory exists: {os.path.exists(ENCODED_MASKS_DIR)}")
    
    # Collect data
    image_paths, mask_paths = collect_image_mask_pairs()
    
    if len(image_paths) == 0:
        print("No image-mask pairs found! Check your dataset structure.")
        return
    
    # Show some examples
    print(f"\nExample pairs:")
    for i in range(min(5, len(image_paths))):
        print(f"Image: {image_paths[i]}")
        print(f"Mask:  {mask_paths[i]}")
        print()
    
    # Test loading and visualizing
    print("Creating visualizations...")
    visualize_sample_pairs(image_paths, mask_paths, 3)
    
    print("Data loading test completed!")

if __name__ == "__main__":
    main()