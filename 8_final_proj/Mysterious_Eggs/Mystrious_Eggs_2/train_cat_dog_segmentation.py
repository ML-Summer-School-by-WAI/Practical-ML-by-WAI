#!/usr/bin/env python3
"""
Cat and Dog Segmentation Training Script
Based on U-Net with MobileNetV2 backbone
"""

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import glob

# Configuration
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 16
EPOCHS = 20
NUM_CLASSES = 2  # Background (0) and Animal (1)

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

def load_and_preprocess(img_path, mask_path):
    """
    Load and preprocess image and mask
    """
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    img = tf.cast(img, tf.float32) / 255.0
    
    # Load and preprocess mask
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_HEIGHT, IMG_WIDTH], method='nearest')
    mask = tf.cast(mask, tf.int32)
    
    return img, mask

def build_unet_with_mobilenet(input_shape, num_classes):
    """
    Build U-Net model with MobileNetV2 encoder
    """
    # Load pre-trained MobileNetV2
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Use these layers as skip connections
    layer_names = [
        'block_1_expand_relu',   # 64x64
        'block_3_expand_relu',   # 32x32
        'block_6_expand_relu',   # 16x16
        'block_13_expand_relu',  # 8x8
        'out_relu',              # 4x4
    ]
    
    base_model_outputs = [base_model.get_layer(name).output for name in layer_names]
    
    # Create encoder
    encoder = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)
    encoder.trainable = False  # Freeze encoder weights
    
    # Build decoder
    inputs = tf.keras.Input(shape=input_shape)
    skips = encoder(inputs)
    
    x = skips[-1]  # Start from bottleneck
    
    # Decoder path with skip connections
    for i, skip in enumerate(reversed(skips[:-1])):
        filters = 512 // (2**i)
        x = layers.Conv2DTranspose(filters, 3, strides=2, padding='same')(x)
        x = layers.Concatenate()([x, skip])
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
    
    # Final upsampling to original size
    outputs = layers.Conv2DTranspose(
        num_classes, 3, strides=2, padding='same', activation='softmax'
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def create_datasets(image_paths, mask_paths):
    """
    Create training and validation datasets
    """
    # Split data
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        image_paths, mask_paths, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(train_imgs)}")
    print(f"Validation samples: {len(val_imgs)}")
    
    # Create datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_imgs, train_masks))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_imgs, val_masks))
    
    # Apply preprocessing
    train_dataset = (train_dataset
                    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                    .cache()
                    .shuffle(1000)
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))
    
    val_dataset = (val_dataset
                  .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(BATCH_SIZE)
                  .prefetch(tf.data.AUTOTUNE))
    
    return train_dataset, val_dataset

def visualize_predictions(model, val_dataset, num_samples=3):
    """
    Visualize model predictions
    """
    for images, masks in val_dataset.take(1):
        predictions = model.predict(images)
        pred_masks = tf.argmax(predictions, axis=-1)
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(15, num_samples * 5))
        
        for i in range(min(num_samples, len(images))):
            # Original image
            axes[i, 0].imshow(images[i])
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            # Ground truth mask
            axes[i, 1].imshow(masks[i].numpy().squeeze(), cmap='viridis')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Predicted mask
            axes[i, 2].imshow(pred_masks[i], cmap='viridis')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig('prediction_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        break

def main():
    """
    Main training function
    """
    print("=== Cat and Dog Segmentation Training ===")
    
    # Collect data
    image_paths, mask_paths = collect_image_mask_pairs()
    
    if len(image_paths) == 0:
        print("No image-mask pairs found! Check your dataset structure.")
        return
    
    # Create datasets
    train_dataset, val_dataset = create_datasets(image_paths, mask_paths)
    
    # Build model
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    model = build_unet_with_mobilenet(input_shape, NUM_CLASSES)
    
    print("\nModel Summary:")
    model.summary()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras', save_best_only=True, monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )
    ]
    
    # Train model
    print("\nStarting training...")
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Evaluate model
    print("\nEvaluating model...")
    loss, accuracy = model.evaluate(val_dataset)
    print(f"Validation Loss: {loss:.4f}")
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    # Save final model
    model.save('cat_dog_segmentation_model.keras')
    print("Model saved as 'cat_dog_segmentation_model.keras'")
    
    # Visualize results
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, val_dataset)
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()