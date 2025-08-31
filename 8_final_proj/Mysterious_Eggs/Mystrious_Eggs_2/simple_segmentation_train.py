#!/usr/bin/env python3
"""
Simple segmentation training script for cat and dog dataset
"""

import os
import glob
import cv2
import numpy as np
import json
import random

# Configuration
IMG_SIZE = 128
DATASET_BASE = 'dataset/cat_and_dog_dataset'
CATS_DIR = os.path.join(DATASET_BASE, 'cats')
DOGS_DIR = os.path.join(DATASET_BASE, 'dogs')
ENCODED_MASKS_DIR = os.path.join(DATASET_BASE, 'encoded_masks')

def collect_data():
    """Collect all image-mask pairs"""
    print("Collecting data...")
    
    # Get all JPG images
    cat_images = glob.glob(os.path.join(CATS_DIR, '*.jpg'))
    cat_images.extend(glob.glob(os.path.join(CATS_DIR, 'cat', '*.jpg')))
    
    dog_images = glob.glob(os.path.join(DOGS_DIR, '*.jpg'))
    dog_images.extend(glob.glob(os.path.join(DOGS_DIR, 'dog', '*.jpg')))
    
    all_images = cat_images + dog_images
    
    # Match with masks
    matched_pairs = []
    for img_path in all_images:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(ENCODED_MASKS_DIR, base_name + '.png')
        
        if os.path.exists(mask_path):
            matched_pairs.append((img_path, mask_path))
    
    print(f"Found {len(matched_pairs)} image-mask pairs")
    return matched_pairs

def load_and_preprocess_batch(pairs, start_idx, batch_size):
    """Load and preprocess a batch of data"""
    images = []
    masks = []
    
    end_idx = min(start_idx + batch_size, len(pairs))
    
    for i in range(start_idx, end_idx):
        img_path, mask_path = pairs[i]
        
        # Load and resize image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype(np.float32) / 255.0
        
        # Load and resize mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        images.append(img)
        masks.append(mask)
    
    return np.array(images), np.array(masks)

def calculate_statistics(pairs):
    """Calculate dataset statistics"""
    print("Calculating dataset statistics...")
    
    total_pixels = 0
    background_pixels = 0
    animal_pixels = 0
    
    # Sample a subset for statistics
    sample_pairs = pairs[::10]  # Every 10th image
    
    for img_path, mask_path in sample_pairs:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        
        unique, counts = np.unique(mask, return_counts=True)
        pixel_counts = dict(zip(unique, counts))
        
        total_pixels += mask.size
        background_pixels += pixel_counts.get(0, 0)
        animal_pixels += pixel_counts.get(1, 0)
    
    bg_ratio = background_pixels / total_pixels
    animal_ratio = animal_pixels / total_pixels
    
    print(f"Background pixels: {bg_ratio:.2%}")
    print(f"Animal pixels: {animal_ratio:.2%}")
    
    return bg_ratio, animal_ratio

def create_simple_model_data(train_pairs, val_pairs, batch_size=32):
    """Create training data for simple model"""
    print("Preparing training data...")
    
    # Load training data in batches
    train_images, train_masks = [], []
    
    for i in range(0, len(train_pairs), batch_size):
        batch_imgs, batch_masks = load_and_preprocess_batch(train_pairs, i, batch_size)
        train_images.extend(batch_imgs)
        train_masks.extend(batch_masks)
        print(f"Loaded training batch {i//batch_size + 1}/{(len(train_pairs)-1)//batch_size + 1}")
    
    # Load validation data
    val_images, val_masks = [], []
    for i in range(0, len(val_pairs), batch_size):
        batch_imgs, batch_masks = load_and_preprocess_batch(val_pairs, i, batch_size)
        val_images.extend(batch_imgs)
        val_masks.extend(batch_masks)
        print(f"Loaded validation batch {i//batch_size + 1}/{(len(val_pairs)-1)//batch_size + 1}")
    
    return (np.array(train_images), np.array(train_masks), 
            np.array(val_images), np.array(val_masks))

def save_training_info(train_pairs, val_pairs, stats):
    """Save training information"""
    info = {
        'dataset_info': {
            'total_samples': len(train_pairs) + len(val_pairs),
            'training_samples': len(train_pairs),
            'validation_samples': len(val_pairs),
            'image_size': IMG_SIZE,
            'num_classes': 2
        },
        'statistics': {
            'background_ratio': stats[0],
            'animal_ratio': stats[1]
        },
        'class_mapping': {
            0: 'background',
            1: 'animal'
        }
    }
    
    with open('dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)
    
    print("Dataset information saved to dataset_info.json")

def main():
    """Main function"""
    print("=== Cat and Dog Segmentation Data Preparation ===")
    
    # Collect data
    pairs = collect_data()
    
    if len(pairs) < 10:
        print("Not enough data pairs found!")
        return
    
    # Calculate statistics
    stats = calculate_statistics(pairs)
    
    # Split data manually
    random.seed(42)
    random.shuffle(pairs)
    split_idx = int(0.8 * len(pairs))
    train_pairs = pairs[:split_idx]
    val_pairs = pairs[split_idx:]
    print(f"Training pairs: {len(train_pairs)}")
    print(f"Validation pairs: {len(val_pairs)}")
    
    # Save dataset info
    save_training_info(train_pairs, val_pairs, stats)
    
    # Prepare a small sample for testing
    print("\nPreparing sample data...")
    sample_train = train_pairs[:20]  # First 20 for testing
    sample_val = val_pairs[:5]       # First 5 for validation
    
    train_X, train_y, val_X, val_y = create_simple_model_data(sample_train, sample_val)
    
    print(f"\nData shapes:")
    print(f"Training images: {train_X.shape}")
    print(f"Training masks: {train_y.shape}")
    print(f"Validation images: {val_X.shape}")  
    print(f"Validation masks: {val_y.shape}")
    
    # Save preprocessed data
    print("\nSaving preprocessed data...")
    np.save('train_images_sample.npy', train_X)
    np.save('train_masks_sample.npy', train_y)
    np.save('val_images_sample.npy', val_X)
    np.save('val_masks_sample.npy', val_y)
    
    print("\nData preparation completed!")
    print("Files saved:")
    print("- dataset_info.json")
    print("- train_images_sample.npy")
    print("- train_masks_sample.npy") 
    print("- val_images_sample.npy")
    print("- val_masks_sample.npy")
    
    print(f"\nYou can now use these files to train a segmentation model.")
    print(f"The data is ready for TensorFlow/PyTorch training when you have the libraries installed.")

if __name__ == "__main__":
    main()