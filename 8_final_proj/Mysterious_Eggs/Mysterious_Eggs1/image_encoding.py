# %%
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm 
import matplotlib.pyplot as plt
# %%
def create_color_to_label_map(mask_folder_path, valid_exts=(".png", ".jpg", ".jpeg")):
    """
    Scans all masks in a folder to find unique colors and create a mapping
    from color to a class label.
    """
    print(f"Scanning masks in '{mask_folder_path}'...")

    unique_colors = set()

    # Filter only valid image files
    mask_files = [f for f in os.listdir(mask_folder_path) if f.lower().endswith(valid_exts)]

    if not mask_files:
        raise ValueError(f"No image files found in {mask_folder_path}")

    for filename in tqdm(mask_files, desc="Finding unique colors"):
        mask_path = os.path.join(mask_folder_path, filename)

        # Read image
        mask_bgr = cv2.imread(mask_path)
        if mask_bgr is None:
            print(f"⚠️ Skipping unreadable file: {filename}")
            continue

        mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

        # Reshape to list of pixels
        pixels = mask_rgb.reshape(-1, 3)

        # Get unique RGB colors from this mask
        unique_pixel_colors = np.unique(pixels, axis=0)

        # Add to global set
        for color in unique_pixel_colors:
            unique_colors.add(tuple(color))

    # Sort colors and assign labels
    sorted_colors = sorted(list(unique_colors))
    color_to_label = {color: label for label, color in enumerate(sorted_colors)}

    print("\nScan complete!")
    print(f"Found {len(color_to_label)} unique classes across {len(mask_files)} files.")

    return color_to_label

# %%
mask_folder = "dataset/cat_and_dog_dataset/SegmentationClass"
COLOR_TO_LABEL = create_color_to_label_map(mask_folder)
COLOR_TO_LABEL

# %%


# %%
import numpy as np
import cv2
from PIL import Image

def encode_mask_to_grayscale(mask_path, color_map):
    """
    Converts an RGB segmentation mask to a grayscale mask with class labels.
    """
    mask_bgr = cv2.imread(mask_path)
    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = mask_rgb.shape
    
    # Create an empty grayscale mask (height x width)
    mask_grayscale = np.zeros((height, width), dtype=np.uint8)
    
    # For each color in our map, find where it is in the mask and assign the label
    for color, label in color_map.items():
        # Find pixels matching the color
        matches = np.where(np.all(mask_rgb == color, axis=-1))
        mask_grayscale[matches] = label
        
    return mask_grayscale




# %%
input_folder = "dataset/cat_and_dog_dataset/SegmentationClass"
output_folder = "dataset/cat_and_dog_dataset/encoded_masks"
os.makedirs(output_folder, exist_ok=True)

# Loop through all mask images
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        mask_path = os.path.join(input_folder, filename)
        
        # Encode to grayscale
        grayscale_label_mask = encode_mask_to_grayscale(mask_path, COLOR_TO_LABEL)
        
        # Save with same filename but as .png (better for masks than jpg)
        save_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".png")
        cv2.imwrite(save_path, grayscale_label_mask)

        print(f"✅ Saved encoded mask: {save_path}")

print("\nAll masks processed and saved!")
# %%


# %% [markdown]
# ### Reread the encoded mask

# %%

new_grayscale_label_mask = cv2.imread(save_path)
new_grayscale_label_mask = cv2.cvtColor(new_grayscale_label_mask, cv2.COLOR_BGR2GRAY)
plt.imshow(new_grayscale_label_mask)

# %%


# %%



