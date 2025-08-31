from PIL import Image

import matplotlib.pyplot as plt
import tensorflow as tf

def predictImg(org_img, model):
    
    # Preprocess the image
    img = tf.convert_to_tensor(org_img, dtype=tf.float32)
    img = tf.image.resize(img, model["catAndDogModel"].input_img_size)
    img_normalized = img / 255.0
    img_batch = tf.expand_dims(img_normalized, axis=0)
    
    # Make prediction
    predicted_masks = model["catAndDogModel"].model.predict(img_batch)
    pred_mask = tf.argmax(predicted_masks, axis=-1)
    pred_mask = pred_mask[0]
    
    # Display the results
    showImage(img, pred_mask)

def showImage(img, pred_mask):

    # Display the original image and the predicted mask side by side
    plt.figure(figsize=(12, 6))
    
    # Original Image
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(tf.keras.utils.array_to_img(img))
    plt.axis('off')
    
    # Predicted Mask
    plt.subplot(1, 2, 2)
    plt.title("Predicted Mask")
    plt.imshow(pred_mask)
    plt.axis('off')
    
    plt.show()