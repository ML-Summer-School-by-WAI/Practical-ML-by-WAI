from fastapi import FastAPI, UploadFile, File
from fastapi.responses import Response
from contextlib import asynccontextmanager
from app.models import MLHerosSegmentation
import cv2

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model = MLHerosSegmentation()
    await model.load_model()
    ml_models["segmentation_model"] = model
    yield
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

@app.get("/")
def root():
    return {"message": "ML_Heros API running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    output_mask = ml_models["segmentation_model"].predict_image(image_bytes)
    encoded_image = cv2.imencode('.png', output_mask)[1].tobytes()
    return Response(content=encoded_image, media_type="image/png")
