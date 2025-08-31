from fastapi import FastAPI, UploadFile, File, HTTPException
from contextlib import asynccontextmanager
from PIL import Image
from .utils import predictImg
from .model_work import CatAndDogModel

import uvicorn
import io
import numpy as np
    
ml_models ={}

@asynccontextmanager
async def startup_lifespan(app : FastAPI):
    # Initialize and load Cat and Dog model
    catAndDogModel = CatAndDogModel()
    catAndDogModel.load_model()
    ml_models["catAndDogModel"] = catAndDogModel

    yield
    ml_models.clear()


app = FastAPI(lifespan=startup_lifespan)

@app.get("/")
def home():
    return "Hello, World!"

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic validation
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Unsupported file type. Send an image.")

    try:
        # Read and process the image
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
        img = img.convert("RGB")
        img = np.array(img)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image.")
    
    try:
        predictImg(img, ml_models)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

if __name__ == "__main__":
    uvicorn.run("main:app", host='0.0.0.0', port=8888, reload=True)
