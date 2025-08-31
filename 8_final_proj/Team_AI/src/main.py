from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import io
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("ML011_SuYee_model.h5")

IMG_HEIGHT, IMG_WIDTH = 128, 128

def model_predict(img_bgr):
    # Resize to model input size
    img_resized = cv2.resize(img_bgr, (IMG_WIDTH, IMG_HEIGHT)) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)  # add batch

    # Model prediction
    pred = model.predict(img_input)[0]  

    # For binary segmentation: threshold
    if pred.shape[-1] == 1:
        mask = (pred[...,0] > 0.5).astype(np.uint8)  # shape (H,W)
    else:
        # Multi-class: argmax
        mask = np.argmax(pred, axis=-1).astype(np.uint8)  # shape (H,W)

    # Resize mask back to original image size
    mask = cv2.resize(mask, (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask 

def overlay_mask(img_bgr, mask):
    """
    img_bgr: original image (H,W,3)
    mask: predicted mask (H,W) with class IDs
    """
    # Define colors for each class (BGR)
    colors = {
        0: [0, 0, 0],        # background
        1: [0, 255, 0],      # cat = green
        2: [0, 0, 255]       # dog = red
    }

    # Create color mask
    color_mask = np.zeros_like(img_bgr)
    for class_id, color in colors.items():
        color_mask[mask == class_id] = color

    # Overlay with transparency
    alpha = 0.5
    overlayed = cv2.addWeighted(img_bgr, 1-alpha, color_mask, alpha, 0)
    return overlayed

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/segment/")
async def segment_image(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Predict mask
    mask = model_predict(img)

    # Overlay mask on original image
    overlay_img = overlay_mask(img, mask)

    # Encode overlay as PNG
    _, buffer = cv2.imencode(".png", overlay_img)
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
