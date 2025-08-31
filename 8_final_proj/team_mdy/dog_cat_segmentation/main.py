"""
main.py
FastAPI app for Cat & Dog Semantic Segmentation
"""

from fastapi import FastAPI, Body, File, UploadFile, HTTPException
from dataclasses import dataclass, asdict
from typing import List, Union
import time
import io
import base64
from model_work import SemanticSegmentation
from PIL import Image
import numpy as np
import tensorflow as tf
from fastapi.responses import StreamingResponse

# -----------------------------
# Dataclasses for Requests/Responses
# -----------------------------
@dataclass
class ImageRequest:
    image: bytes  # raw image bytes

@dataclass
class ImageSegmentationResult:
    original_size: list[int]          # [height, width]
    mask_rle: str                     # Run-Length Encoded mask
    overlay_image: str                # Base64 PNG overlay
    colored_mask_image: str           # Base64 PNG colored mask
    class_percentages: dict           # pixel percentage per class
    class_labels: dict                # class labels (0: background, 1: dog, 2: cat)
    dominant_class: str               # dominant class label

@dataclass
class SegmentationResponse:
    results: List[ImageSegmentationResult]
    execution_time_ms: int
    model_version: str

# -----------------------------
# App & model setup
# -----------------------------
ml_models = {}
app = FastAPI(title="Cat & Dog Segmentation API", version="1.0")

@app.on_event("startup")
async def load_models():
    try:
        seg_model = SemanticSegmentation()
        # Try different possible model paths
        model_paths = ["model/dog_cat.h5", "dog_cat.h5", "cat_dog.h5", "model/cat_dog.h5"]
        model_loaded = False
        
        for path in model_paths:
            try:
                seg_model.load_model(path)
                print(f"Model loaded successfully from: {path}")
                model_loaded = True
                break
            except FileNotFoundError:
                print(f"Model not found at: {path}")
                continue
        
        if not model_loaded:
            print("Warning: No model file found. Please ensure a model file exists.")
            # Create a dummy model for testing
            seg_model = None
        
        ml_models["seg_model"] = seg_model
    except Exception as e:
        print(f"Error loading model: {e}")
        ml_models["seg_model"] = None

# -----------------------------
# Utilities
# -----------------------------
def mask_to_rle(mask: np.ndarray) -> str:
    """Simple Run-Length Encoding"""
    pixels = mask.flatten()
    rle = []
    count = 0
    last_val = 0
    for val in pixels:
        if val == last_val:
            count += 1
        else:
            rle.append(str(count))
            count = 1
            last_val = val
    rle.append(str(count))
    return " ".join(rle)

def overlay_to_base64(overlay_array: np.ndarray) -> str:
    """Convert overlay array to base64 PNG"""
    img = Image.fromarray(overlay_array)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def validate_image(image_bytes: bytes) -> np.ndarray:
    """Validate and convert image bytes to numpy array"""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return np.array(img)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

def create_colored_mask(mask: np.ndarray) -> np.ndarray:
    """Create a colored mask image with different colors for each class"""
    # Define colors for each class (RGB format)
    colors = {
        0: [0, 0, 0],      # Background - Black
        1: [255, 0, 0],    # Dog - Red
        2: [0, 255, 0]     # Cat - Green
    }
    
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        colored_mask[mask == class_id] = color
    
    return colored_mask

def get_class_labels() -> dict:
    """Get class label mapping"""
    return {
        0: "background",
        1: "dog", 
        2: "cat"
    }

def determine_dominant_class(class_percentages: dict, class_labels: dict) -> str:
    """Determine the dominant class (excluding background)"""
    # Filter out background class
    animal_percentages = {k: v for k, v in class_percentages.items() if k != 0}
    
    if not animal_percentages:
        return "background"
    
    # Find class with highest percentage
    dominant_class_id = max(animal_percentages, key=animal_percentages.get)
    return class_labels.get(dominant_class_id, "unknown")

# -----------------------------
# Endpoints
# -----------------------------
@app.get("/")
def home():
    return {"message": "Welcome to Cat & Dog Segmentation API!"}

@app.get("/info")
def get_info():
    """Get API information and available endpoints"""
    return {
        "title": "Cat & Dog Segmentation API",
        "version": "1.0",
        "description": "API for semantic segmentation of cat and dog images",
        "endpoints": {
            "/segment": "POST - Segment images using JSON body with image bytes",
            "/segment-file": "POST - Segment images using file upload",
            "/segment-mask-stream": "POST - Get colored mask as direct image stream",
            "/segment-overlay-stream": "POST - Get overlay as direct image stream",
            "/docs": "GET - Interactive API documentation (Swagger UI)"
        },
        "class_labels": {
            0: "background",
            1: "dog", 
            2: "cat"
        },
        "colors": {
            "background": "Black (0,0,0)",
            "dog": "Red (255,0,0)",
            "cat": "Green (0,255,0)"
        }
    }

# -----------------------------
# /segment endpoint (JSON body, batch support)
# -----------------------------
@app.post("/segment")
async def segment_images(data: Union[ImageRequest, List[ImageRequest]] = Body(...)):
    start_time = time.time()

    if ml_models["seg_model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    model: SemanticSegmentation = ml_models["seg_model"]
    results = []

    images = data if isinstance(data, list) else [data]

    for img_req in images:
        try:
            # Validate image
            img_array = validate_image(img_req.image)

            # Predict
            img_pil, rgb_mask, overlay, mask = model.predict(img_req.image)

            # Class percentages
            unique, counts = np.unique(mask, return_counts=True)
            total_pixels = mask.size
            class_percentages = {int(u): int(c / total_pixels * 100) for u, c in zip(unique, counts)}

            # Class labels
            detected_classes = []
            for cls in unique:
                if cls in model.class_labels:
                    label_name, color = model.class_labels[cls]
                    detected_classes.append({
                        "class_id": int(cls),
                        "label": label_name,
                        "color": color
                    })

            # Overlay and colored mask as base64
            overlay_base64 = overlay_to_base64(np.array(overlay))
            colored_mask_base64 = overlay_to_base64(rgb_mask)

            results.append({
                "overlay_image": overlay_base64,
                "colored_mask_image": colored_mask_base64,
                "class_percentages": class_percentages,
                "detected_classes": detected_classes
            })
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

    execution_time_ms = int((time.time() - start_time) * 1000)
    return {
        "results": results,
        "execution_time_ms": execution_time_ms,
        "model_version": "1.0"
    }


# -----------------------------
# /segment-file endpoint (single file upload)
# -----------------------------
@app.post("/segment-file")
async def segment_file(file: UploadFile = File(...)):
    start_time = time.time()

    if ml_models["seg_model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    model: SemanticSegmentation = ml_models["seg_model"]

    try:
        image_bytes = await file.read()
        img_array = validate_image(image_bytes)

        img_pil, rgb_mask, overlay, mask = model.predict(image_bytes)

        unique, counts = np.unique(mask, return_counts=True)
        total_pixels = mask.size
        class_percentages = {int(u): int(c / total_pixels * 100) for u, c in zip(unique, counts)}

        detected_classes = []
        for cls in unique:
            if cls in model.class_labels:
                label_name, color = model.class_labels[cls]
                detected_classes.append({
                    "class_id": int(cls),
                    "label": label_name,
                    "color": color
                })

        overlay_base64 = overlay_to_base64(np.array(overlay))
        colored_mask_base64 = overlay_to_base64(rgb_mask)

        execution_time_ms = int((time.time() - start_time) * 1000)

        return {
            "results": [{
                "overlay_image": overlay_base64,
                "colored_mask_image": colored_mask_base64,
                "class_percentages": class_percentages,
                "detected_classes": detected_classes
            }],
            "execution_time_ms": execution_time_ms,
            "model_version": "1.0"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


@app.post("/segment-overlay-upload")
async def segment_overlay_stream(file: UploadFile = File(...)):
    """Segment an image and return the overlay as a direct image stream with labels in headers"""
    start_time = time.time()

    # Check if model is loaded
    if ml_models["seg_model"] is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check server logs.")

    model: SemanticSegmentation = ml_models["seg_model"]

    try:
        # Read file content
        image_bytes = await file.read()

        # Validate image
        img_array = validate_image(image_bytes)

        # Perform segmentation (new API)
        img_pil, rgb_mask, overlay, mask = model.predict(image_bytes)

        # Prepare label info (which classes exist in mask)
        unique_classes = np.unique(mask)
        class_labels = []
        for cls in unique_classes:
            if cls in model.class_labels:  # ensure label exists
                label_name, color = model.class_labels[cls]
                class_labels.append({
                    "class_id": int(cls),
                    "label": label_name,
                    "color": color
                })

        # Convert overlay to PNG bytes
        buf = io.BytesIO()
        overlay.save(buf, format="PNG")
        buf.seek(0)

        # Return overlay + metadata
        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={
                "X-Execution-Time": str(int((time.time() - start_time) * 1000)),
                "X-Model-Version": "1.0",
                "X-Classes-Detected": str(class_labels)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
