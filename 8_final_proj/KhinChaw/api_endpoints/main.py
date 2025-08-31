import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response
from typing import Annotated
from model_work import SegmentationModel
from contextlib import asynccontextmanager
import cv2

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):

    Model = SegmentationModel()
    await Model.load_model() # type: ignore

    ml_models["segmentation_model"] = Model
    
    yield ml_models

    ml_models.clear()

app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read the image content
    image_bytes = await file.read()
    
    # Make prediction
    output = ml_models["segmentation_model"].predict_image(image_bytes)
    print("<--- Prediction done --->")
    
    
    encoded_image = cv2.imencode('.png', output)[-1].tobytes()
    


    # print(type(output))  # <class 'tensorflow.python.framework.ops.EagerTensor'>
    # Return the image with the same content and type
    return  Response(content=encoded_image, media_type="image/png")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)