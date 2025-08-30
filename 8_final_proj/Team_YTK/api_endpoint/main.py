"""
API module for serving image overlay predictions using FastAPI.
This service loads a trained model once at startup and processes
uploaded images to return overlay visualizations.
"""

import io
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from model_work import SemanticSegmentation

seg_model = {}


@asynccontextmanager
async def startup_lifespan(app: FastAPI):
    """FastAPI lifespan context manager for initializing and
    cleaning up the semantic segmentation model"""
    semantic_seg_model = SemanticSegmentation()
    semantic_seg_model.load_model("cats_dogs_mobilenet.keras")
    seg_model["semantic_seg_model"] = semantic_seg_model
    yield
    seg_model.clear()


app = FastAPI(lifespan=startup_lifespan)


@app.post("/overlay_image")
async def overlay_pred(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Overlay prediction endpoint.

    Args:
        file (UploadFile): An uploaded image file, now it only supports DOG and CAT.

    Returns:
        StreamingResponse: A PNG image with prediction overlay.
    """

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Invalid file type"}
    file_bytes = await file.read()  # read the uploaded file

    overlay_image = seg_model["semantic_seg_model"].predict_and_visualize_color(
        file_bytes
    )  # returns NumPy array

    success, buffer = cv2.imencode(".png", overlay_image)
    if not success:
        return {"error": "Failed to encode image"}

    # Return PNG as StreamingResponse
    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/png")
