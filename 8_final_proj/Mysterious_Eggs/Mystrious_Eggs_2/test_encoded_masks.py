#!/usr/bin/env python3
import cv2
import numpy as np
import os

def test_encoded_mask(mask_path):
    """Test an encoded mask to verify its values"""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not load mask: {mask_path}")
        return
    
    unique_values = np.unique(mask)
    print(f"File: {os.path.basename(mask_path)}")
    print(f"Shape: {mask.shape}")
    print(f"Unique values: {unique_values}")
    print(f"Value counts: {[(val, np.sum(mask == val)) for val in unique_values]}")
    print("-" * 50)

# Test a few encoded masks
encoded_dir = "dataset/cat_and_dog_dataset/encoded_masks"
test_files = ["cat.1.png", "dog.2436.png", "cat.2366.png"]

print("Testing encoded masks:")
print("=" * 50)

for test_file in test_files:
    test_path = os.path.join(encoded_dir, test_file)
    if os.path.exists(test_path):
        test_encoded_mask(test_path)
    else:
        print(f"File not found: {test_file}")

# Also test original vs encoded
print("\nComparing original vs encoded:")
print("=" * 50)

original_path = "dataset/cat_and_dog_dataset/SegmentationClass/cat.1.png"
encoded_path = "dataset/cat_and_dog_dataset/encoded_masks/cat.1.png"

if os.path.exists(original_path) and os.path.exists(encoded_path):
    original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
    encoded = cv2.imread(encoded_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Original unique values: {np.unique(original)}")
    print(f"Encoded unique values: {np.unique(encoded)}")
    
    # Check if encoding mapping is correct
    mapping_correct = True
    if 0 in np.unique(original) and 0 not in np.unique(encoded):
        mapping_correct = False
    if 255 in np.unique(original) and 1 not in np.unique(encoded):
        mapping_correct = False
        
    print(f"Encoding mapping correct: {mapping_correct}")
else:
    print("Could not find files for comparison")