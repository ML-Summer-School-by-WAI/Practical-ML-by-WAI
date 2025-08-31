from fastapi import FastAPI, UploadFile, File
import uvicorn
import tensorflow as tf
import numpy as np
import cv2
import os
from datetime import datetime
from model import CatAndDogModel
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
import time


ml_models = {}

OUTPUT_DIR = "static"
os.makedirs(OUTPUT_DIR, exist_ok=True)

color_map = {
    0: [0, 0, 0],
    1: [255, 0, 0],
    2: [0, 255, 0]
}

def mask_to_rgb(mask, color_map):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, v in color_map.items():
        rgb[mask == k] = v
    return rgb

def cleanup_static_folder(folder="static", max_age_seconds=600):
    now = time.time()
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        if os.path.isfile(file_path):
            file_age = now - os.path.getmtime(file_path)
            if file_age > max_age_seconds:
                os.remove(file_path)
      

@asynccontextmanager
async def startup_lifespan(app: FastAPI):
    catAndDogModel = CatAndDogModel()
    catAndDogModel.load_model()
    ml_models["catAndDogModel"] = catAndDogModel
    yield

app = FastAPI(lifespan=startup_lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    cleanup_static_folder()
    catAndDogModel = ml_models["catAndDogModel"]
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    temp_path = os.path.join(OUTPUT_DIR, f"temp_{ts}.png")
    with open(temp_path, "wb") as f:
        f.write(await file.read())
    img,img_batch = catAndDogModel.preprocess_image(temp_path)
    predicted_masks = catAndDogModel.predict(img_batch)
    pred_mask = tf.argmax(predicted_masks, axis=-1)[0].numpy()
    color_mask = mask_to_rgb(pred_mask, color_map)
    input_image_numpy = tf.keras.utils.img_to_array(img, dtype=np.uint8)

    overlay = cv2.addWeighted(input_image_numpy, 0.6, color_mask, 0.4, 0)

    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    input_filename = f"image_{ts}.png"
    mask_filename = f"mask_{ts}.png"
    overlay_filename = f"overlay_{ts}.png"

    input_path = os.path.join(OUTPUT_DIR, input_filename)
    mask_path = os.path.join(OUTPUT_DIR, mask_filename)
    overlay_path = os.path.join(OUTPUT_DIR, overlay_filename)

    cv2.imwrite(input_path, input_image_numpy)
    cv2.imwrite(mask_path, color_mask)
    cv2.imwrite(overlay_path, overlay)

    base_url = "http://localhost:8888/static"

    return {
        "input_image": f"{base_url}/{input_filename}",
        "mask": f"{base_url}/{mask_filename}",
        "overlay": f"{base_url}/{overlay_filename}"
    }

          
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)