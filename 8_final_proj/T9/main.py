from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import RedirectResponse
import uvicorn
import numpy as np
import cv2
from contextlib import asynccontextmanager
from model_work import ImageModel

ml_models = {}

@asynccontextmanager
async def startup_lifespan(app: FastAPI):
    model = ImageModel()
    model.load_model()
    ml_models["imageModel"] = model
    yield
    ml_models.clear()

app = FastAPI(lifespan=startup_lifespan)

@app.get("/")
def home():
    return RedirectResponse(url="/docs")

@app.post("/predict", responses={200: {"content": {"image/png": {}}}}, response_class=Response)
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    overlayed_img = ml_models["imageModel"].predict_and_overlay(img=image_bytes)
    success, buffer = cv2.imencode('.png', overlayed_img)
    if not success:
        return Response(content=b"", media_type="image/png", status_code=500)
    return Response(content=buffer.tobytes(), media_type="image/png")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)