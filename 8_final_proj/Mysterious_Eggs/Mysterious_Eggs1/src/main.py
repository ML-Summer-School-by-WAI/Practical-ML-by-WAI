from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
import cv2
import io
from semantic_segmentation import SemanticSegmentation


# Global variable to store the model
seg_model = {}


@asynccontextmanager
async def startup_lifespan(app: FastAPI):
    """FastAPI lifespan context manager for initializing and
    cleaning up the semantic segmentation model"""
    semantic_seg_model = SemanticSegmentation()
    semantic_seg_model.load_model("dataset/mysterious_eggs_model.h5")
    seg_model["semantic_seg_model"] = semantic_seg_model
    yield
    seg_model.clear()


app = FastAPI(
    title="Semantic Segmentation API",
    description="API for semantic segmentation of cats and dogs using U-Net",
    version="1.0.0",
    lifespan=startup_lifespan
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Semantic Segmentation API",
        "description": "Upload an image to get semantic segmentation overlay for cats and dogs",
        "endpoints": {
            "/overlay_image": "POST - Upload an image file to get segmentation overlay"
        }
    }


@app.post("/overlay_image")
async def overlay_pred(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Overlay prediction endpoint.

    Args:
        file (UploadFile): An uploaded image file, now it only supports DOG and CAT.

    Returns:
        StreamingResponse: A PNG image with prediction overlay.
    """
    
    # Validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Invalid file type. Please upload PNG, JPEG, or JPG images only."}
    
    try:
        # Read the uploaded file
        file_bytes = await file.read()
        
        # Get the overlay image from the model
        overlay_image = seg_model["semantic_seg_model"].predict_and_visualize_color(
            file_bytes
        )  # returns NumPy array in BGR format
        
        # Encode the image to PNG format
        success, buffer = cv2.imencode(".png", overlay_image)
        if not success:
            return {"error": "Failed to encode image"}
        
        # Return PNG as StreamingResponse
        return StreamingResponse(
            io.BytesIO(buffer.tobytes()), 
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=segmentation_overlay.png"}
        )
        
    except Exception as e:
        return {"error": f"Failed to process image: {str(e)}"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    model_loaded = "semantic_seg_model" in seg_model and seg_model["semantic_seg_model"].model is not None
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }
