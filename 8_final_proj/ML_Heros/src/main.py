from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model


app = FastAPI(title="Image Segmentation API")


# MODEL_PATH = "ML_Heros_model.h5"
MODEL_PATH = "8_final_proj\ML_Heros\ML_Heros_model.keras"

model = load_model(MODEL_PATH)

IMG_HEIGHT, IMG_WIDTH = 128, 128


def model_predict(img_bgr):
    """
    Run segmentation prediction on an image.
    """
    # Resize and normalize
    img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)

    # Model prediction
    pred = model.predict(img_input)[0]  # shape (H, W, C) or (H, W, 1)

    if pred.shape[-1] == 1:  # binary segmentation
        mask = (pred[..., 0] > 0.5).astype(np.uint8)
    else:  # multi-class segmentation
        mask = np.argmax(pred, axis=-1).astype(np.uint8)

    # Resize mask back to original image
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def overlay_mask(img_bgr, mask):
    """
    Overlay segmentation mask on original image.
    """
    # Define colors (BGR format)
    colors = {
        0: [0, 0, 0],       # background = black
        1: [0, 255, 0],     # cat = green
        2: [0, 0, 255],     # dog = red
    }

    # Create color mask
    color_mask = np.zeros_like(img_bgr)
    for class_id, color in colors.items():
        color_mask[mask == class_id] = color

    # Blend with original
    alpha = 0.5
    overlayed = cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)
    return overlayed


# API Endpoints

@app.get("/")
def root():
    return {"message": "Image Segmentation API is running!"}


@app.post("/segment/")
async def segment_image(file: UploadFile = File(...), overlay: bool = True):
    """
    Upload an image -> returns segmentation result.
    - If overlay=True, returns image with overlay mask.
    - If overlay=False, returns mask only.
    """
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Predict mask
    mask = model_predict(img)

    # Select output
    if overlay:
        output_img = overlay_mask(img, mask)
    else:
        # Expand mask into 3 channels for saving as PNG
        output_img = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

    # Encode and stream response
    _, buffer = cv2.imencode(".png", output_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
