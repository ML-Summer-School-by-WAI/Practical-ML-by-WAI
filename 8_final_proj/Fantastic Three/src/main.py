# main.py

import os
import io
import numpy as np
import cv2
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException, Response, Query
from PIL import Image
from typing import Dict, Any, Literal
from contextlib import asynccontextmanager

from model_load import load_segmentation_model, segmentation_model

# --- CONFIGURATION ---
IMG_SIZE = (128, 128)
NUM_CLASSES = 3
COLOR_MAP = np.array([
    [0, 0, 0],       # Class 0: Black (Background)
    [255, 0, 0],     # Class 1: Red (Dog)
    [0, 0, 255]      # Class 2: Blue (Cat)
], dtype=np.uint8)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global segmentation_model
    try:
        segmentation_model = load_segmentation_model()
        yield
    except (FileNotFoundError, RuntimeError) as e:
        print(f"Application failed to start due to model loading error: {e}")
        segmentation_model = None
        yield

app = FastAPI(
    title="Dog and Cat Segmentation API",
    description="Upload an image of a dog or cat and get a segmented mask back.",
    version="1.0.0",
    lifespan=lifespan
)

def mask_to_rgb(mask: np.ndarray, color_map: np.ndarray) -> np.ndarray:
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_idx, color in enumerate(color_map):
        rgb_mask[mask == class_idx] = color
    return rgb_mask

async def predict_segmentation(image_bytes: bytes) -> Dict[str, bytes]:
    if segmentation_model is None:
        raise HTTPException(status_code=500, detail="Segmentation model not loaded.")

    try:
        original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        original_image_np = np.array(original_image)
        
        input_image = original_image.resize(IMG_SIZE)
        input_image = np.array(input_image, dtype=np.float32) / 255.0
        input_image = np.expand_dims(input_image, axis=0)
        
        # Use verbose=0 to suppress the progress bar
        predicted_masks = segmentation_model.predict(input_image, verbose=0)
        pred_mask_encoded = np.argmax(predicted_masks, axis=-1)[0]
        
        color_mask = mask_to_rgb(pred_mask_encoded, COLOR_MAP)
        color_mask_resized = cv2.resize(color_mask, (original_image_np.shape[1], original_image_np.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = cv2.addWeighted(original_image_np, 0.6, color_mask_resized, 0.4, 0)

        color_mask_bytes_io = io.BytesIO()
        Image.fromarray(color_mask_resized).save(color_mask_bytes_io, format="PNG")
        color_mask_bytes_io.seek(0)
        
        overlay_bytes_io = io.BytesIO()
        Image.fromarray(overlay).save(overlay_bytes_io, format="PNG")
        overlay_bytes_io.seek(0)
        
        return {
            "color_mask": color_mask_bytes_io.getvalue(),
            "overlay_mask": overlay_bytes_io.getvalue()
        }

    except Exception as e:
        print(f"Error during segmentation prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {e}")


# Post Method
@app.post("/predict-mask", summary="Get a specific mask type", response_description="Returns the requested mask type as a PNG image")
async def get_mask(
    file: UploadFile = File(...),
    mask_type: Literal["overlay_mask", "color_mask"] = Query("overlay_mask", description="Specify the type of mask to return.")
):
    """
    **Upload an image** and get either the **overlay mask** or the **color mask**.
    
    - Use the `mask_type` query parameter to choose which mask to receive.
    - If no parameter is provided, it defaults to the `overlay_mask`.
    
    **Expects**: An image file (`.jpg`, `.png`, etc.)
    **Returns**: A PNG image with the requested segmentation result.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, detail="Invalid file type. Please upload an image."
        )

    try:
        image_data = await file.read()
        segmented_images = await predict_segmentation(image_data)
        
        # Return the requested mask based on the query parameter
        return Response(content=segmented_images[mask_type], media_type="image/png")
    
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

# Get Method 
@app.get("/", tags=["Root"])
async def read_root() -> Dict[str, str]:
    return {"message": "Welcome to the Dog and Cat Segmentation API. Visit /docs for API documentation."}
