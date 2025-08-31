from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import numpy as np
from PIL import Image
import io
import os
import base64

from model_work import CatDogModel
from models.schema import PredictionResponse

app = FastAPI()

MODEL_PATH = os.path.join(os.getcwd(), "ml_models", "unet_model_ml020.h5")
CLASS_NAMES_PATH = os.path.join(os.getcwd(), "ml_models", "class_names.json")

cat_dog_model = CatDogModel(MODEL_PATH, CLASS_NAMES_PATH)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Keep track of last uploaded file
last_uploaded_file = None

@app.get("/")
def root():
    return {"message": "Cat vs Dog API is running!"}

"""
Uploads an image, performs cat/dog segmentation, and returns prediction results.

"""
@app.post("/predict_image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    global last_uploaded_file, last_segmentation_file
    try:
        input_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        last_uploaded_file = input_path

        img = Image.open(input_path).convert("RGB")
        img_resized = img.resize((128, 128))
        img_array = np.array(img_resized) / 255.0

        preds = cat_dog_model.model.predict(np.expand_dims(img_array, axis=0))[0]  
        mask_model = np.argmax(preds, axis=-1)  # raw mask from model

        mask = np.zeros_like(mask_model)
        mask[mask_model == 0] = 0  # background
        mask[mask_model == 1] = 2  # dog → 2
        mask[mask_model == 2] = 1  # cat → 1

        colors = {
            0: (0, 0, 0),       # black = background
            1: (0, 255, 0),     # green = cat
            2: (0, 0, 255)      # blue = dog
        }
        mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for k, color in colors.items():
            mask_rgb[mask == k] = color

        mask_img = Image.fromarray(mask_rgb)
        mask_filename = f"mask_{file.filename}"
        mask_path = os.path.join(UPLOAD_DIR, mask_filename)
        mask_img.save(mask_path)
        last_segmentation_file = mask_path

        mask_no_bg = mask[mask != 0]
        if len(mask_no_bg) > 0:
            unique, counts = np.unique(mask_no_bg, return_counts=True)
            main_class_index = unique[np.argmax(counts)]
        else:
            main_class_index = 0  # fallback to background

        # Correct class_names order
        cat_dog_model.class_names = ["background", "cat", "dog"]
        main_class = cat_dog_model.class_names[main_class_index]

        confidence = float((mask == main_class_index).sum() / mask.size)

        with open(input_path, "rb") as f:
            img_bytes = f.read()
        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        return PredictionResponse(
            class_name=main_class,
            confidence=confidence,
            image_base64=img_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
Generates a side-by-side comparison of the original image, mask, and overlay.

"""
@app.get("/compare")
async def compare_images(alpha: float = 0.5):
    global last_uploaded_file, last_segmentation_file

    if not last_uploaded_file or not os.path.exists(last_uploaded_file):
        raise HTTPException(status_code=404, detail="No original image available. Upload an image first.")
    if not last_segmentation_file or not os.path.exists(last_segmentation_file):
        raise HTTPException(status_code=404, detail="No segmentation mask available. Upload an image first.")

    try:
        original = Image.open(last_uploaded_file).convert("RGB")
        mask = Image.open(last_segmentation_file).convert("RGBA").resize(original.size)

        mask.putalpha(int(alpha * 255))
        overlay_img = Image.alpha_composite(original.convert("RGBA"), mask)

        width, height = original.size
        combined = Image.new("RGB", (width * 3, height))
        combined.paste(original, (0, 0))
        combined.paste(mask.convert("RGB"), (width, 0))
        combined.paste(overlay_img.convert("RGB"), (width * 2, 0))

        compare_path = os.path.join(
            UPLOAD_DIR, f"compare_{os.path.splitext(os.path.basename(last_uploaded_file))[0]}.png"
        )
        combined.save(compare_path, format="PNG")

        return FileResponse(compare_path, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
