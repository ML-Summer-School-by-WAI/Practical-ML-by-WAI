# main.py
from __future__ import annotations

import io
from contextlib import asynccontextmanager
from typing import Dict, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Response, Query, HTTPException
from fastapi.responses import StreamingResponse

# TensorFlow / Keras
from tensorflow.keras.models import load_model


# ==============================
# Config
# ==============================
MODEL_PATH = "final_model.keras"       # your saved Keras model file
DEFAULT_ALPHA = 0.5                    # overlay transparency (0..1)

# Class IDs: 0=background, 1=cat, 2=dog (adjust if your labels differ)
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),      # background (black / transparent)
    1: (0, 255, 0),    # cat (green)
    2: (0, 0, 255),    # dog (red)
}

app = FastAPI(title="Cat/Dog Segmentation API", version="1.0.0")

_state: Dict[str, object] = {
    "model": None,             # will hold the loaded model
    "input_size": (128, 128),  # (H, W), updated after load
}


# ==============================
# Lifespan: load model once
# ==============================
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        model = load_model(MODEL_PATH)
    except Exception as e:
        _state["model"] = None
        print(f"[ERROR] Failed to load model at '{MODEL_PATH}': {e}")
    else:
        _state["model"] = model

        # Try to infer model input size (H, W)
        try:
            _, in_h, in_w, _ = model.input_shape  # (None, H, W, C)
            _state["input_size"] = (int(in_h), int(in_w))
        except Exception:
            _state["input_size"] = (128, 128)  # fallback

        print(f"[INFO] Loaded model from '{MODEL_PATH}' with input size: {_state['input_size']}")

    try:
        yield
    finally:
        _state["model"] = None


app.router.lifespan_context = lifespan


# ==============================
# Helpers
# ==============================
def _ensure_model_available():
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=f"Segmentation model not available. Ensure '{MODEL_PATH}' exists and is valid.",
        )


def predict_mask(img_bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Runs model inference and returns a (H, W) uint8 mask of class IDs.
    Supports binary (1-channel) or multi-class outputs.
    """
    _ensure_model_available()
    model = _state["model"]  # type: ignore
    in_h, in_w = _state["input_size"]  # type: ignore

    # Preprocess
    img_resized = cv2.resize(img_bgr, (in_w, in_h)).astype(np.float32) / 255.0
    img_input = np.expand_dims(img_resized, axis=0)  # (1, H, W, 3)

    # Predict
    pred = model.predict(img_input, verbose=0)[0]  # (H, W, C) or (H, W, 1)

    # Postprocess
    if pred.shape[-1] == 1:
        mask_small = (pred[..., 0] > threshold).astype(np.uint8)
    else:
        mask_small = np.argmax(pred, axis=-1).astype(np.uint8)

    # Resize back to original size
    mask = cv2.resize(
        mask_small,
        (img_bgr.shape[1], img_bgr.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    return mask


def overlay_mask(img_bgr: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    """Overlay a colorized mask onto the original image."""
    color_mask = np.zeros_like(img_bgr, dtype=np.uint8)
    for class_id, bgr in CLASS_COLORS.items():
        color_mask[mask == class_id] = bgr
    return cv2.addWeighted(img_bgr, 1 - alpha, color_mask, alpha, 0)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Convert class mask (H, W) to a color image using CLASS_COLORS."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, bgr in CLASS_COLORS.items():
        color[mask == class_id] = bgr
    return color


def encode_png(img: np.ndarray) -> Optional[bytes]:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else None


def read_image_from_upload(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


# ==============================
# Routes
# ==============================
@app.get("/", summary="Health & model info")
def health():
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_path": MODEL_PATH,
        "model_input_size": _state["input_size"],
        "classes": {0: "background", 1: "cat", 2: "dog"},
    }


@app.post("/segment", response_class=StreamingResponse, summary="Overlay mask on image")
async def segment_image(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    alpha: float = Query(DEFAULT_ALPHA, ge=0.0, le=1.0, description="Overlay transparency"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for binary models"),
):
    """Return original image with segmentation mask overlay."""
    data = await file.read()
    img = read_image_from_upload(data)
    if img is None:
        return Response(content="Invalid image", status_code=400)

    mask = predict_mask(img, threshold=threshold)
    overlay = overlay_mask(img, mask, alpha=alpha)

    png = encode_png(overlay)
    if not png:
        return Response(content="Failed to encode PNG", status_code=500)

    return StreamingResponse(io.BytesIO(png), media_type="image/png")


@app.post("/segment/mask", response_class=StreamingResponse, summary="Return mask only")
async def segment_mask(
    file: UploadFile = File(..., description="Image file (jpg/png)"),
    format: str = Query("color", pattern="^(color|gray)$", description="Mask output: color or gray"),
    threshold: float = Query(0.5, ge=0.0, le=1.0, description="Threshold for binary models"),
):
    """Return segmentation mask only (color or grayscale)."""
    data = await file.read()
    img = read_image_from_upload(data)
    if img is None:
        return Response(content="Invalid image", status_code=400)

    mask = predict_mask(img, threshold=threshold)
    out = colorize_mask(mask) if format == "color" else mask.astype(np.uint8)

    png = encode_png(out)
    if not png:
        return Response(content="Failed to encode PNG", status_code=500)

    return StreamingResponse(io.BytesIO(png), media_type="image/png")
