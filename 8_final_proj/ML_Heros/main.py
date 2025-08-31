from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from contextlib import asynccontextmanager
from models import SegmentationModel
import cv2
import numpy as np


ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    print("Application is starting up.")
    yield
    # Shutdown code
    print("Application is shutting down.")

app = FastAPI(lifespan=lifespan)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
   
    # Check if the model is loaded before proceeding
    if "segmentation_model" not in ml_models or ml_models["segmentation_model"].model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Please check server logs.")

    try:
        # Read the uploaded image bytes
        image_bytes = await file.read()
        
        # Get the segmentation model from our global dictionary
        model = ml_models["segmentation_model"]
        
        # Call the predict_image method to get the output mask
        output_mask = model.predict_image(image_bytes)

        # Check if the prediction was successful (i.e., not None)
        if output_mask is None:
            raise HTTPException(status_code=400, detail="Failed to process image. Make sure the file is a valid image.")

        _, encoded_image = cv2.imencode('.png', output_mask)
        
        # Convert the buffer to bytes
        encoded_image_bytes = encoded_image.tobytes()
        
        # Return the byte-encoded image with the correct media type
        return Response(content=encoded_image_bytes, media_type="image/png")

    except Exception as e:
        # Catch any other exceptions and return a 500 status code
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")
