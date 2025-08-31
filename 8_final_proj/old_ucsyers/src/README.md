# 🐾 Cat & Dog Segmentation API

This project provides a **FastAPI service** for semantic segmentation of cat and dog images using a trained TensorFlow/Keras model (`final_model.keras`).  

It exposes endpoints to return either:
- **Overlay image**: original image + transparent segmentation mask
- **Mask only**: color-coded or grayscale class mask

---

## 📦 Setup


## 📂 Place Your Model

Ensure the trained model file is in the **same folder as `main.py`**:

```
src/
 ├── main.py
 └── final_model.keras
```

- File name must be: **`final_model.keras`**

---

## ▶️ Run the API

If you are **inside the folder** that contains `main.py`:

```bash
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

You should see logs like:

```
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
[INFO] Loaded model from 'final_model.keras' with input size: (128, 128)
```

---

## 🔍 Test the API in Swagger UI

Once running, open:

- Swagger UI → http://localhost:8888/docs  
- ReDoc → http://localhost:8888/redoc  

### A) Health check
1. Expand **`GET /`**
2. Click **Try it out** → **Execute**
3. You should see JSON with `"model_loaded": true`

### B) Overlay segmentation
1. Expand **`POST /segment`** → **Try it out**
2. **Choose File** → select a JPG/PNG of a cat or dog
3. (Optional) set `alpha=0.5`, leave `threshold=0.5`
4. **Execute**
5. Download the file → save as `overlay.png`

### C) Mask only
1. Expand **`POST /segment/mask`** → **Try it out**
2. **Choose File**
3. Set `format=color` (or `gray`)
4. **Execute**
5. Download the file → save as `mask.png`

---

## 🧪 Test via cURL

```bash
# Overlay image
curl -X POST -F "file=@cat.jpg" "http://localhost:8888/segment" --output overlay.png

# Mask (color)
curl -X POST -F "file=@cat.jpg" "http://localhost:8888/segment/mask?format=color" --output mask_color.png

# Mask (gray)
curl -X POST -F "file=@cat.jpg" "http://localhost:8888/segment/mask?format=gray" --output mask_gray.png
```

---

## ⚙️ API Endpoints

### `GET /`
- Returns health & model info

### `POST /segment`
- Input: image file
- Output: PNG of original image with mask overlay
- Query params:
  - `alpha` (0.0–1.0, default 0.5) — overlay transparency
  - `threshold` (binary models only, default 0.5)

### `POST /segment/mask`
- Input: image file
- Output: PNG of mask only
- Query params:
  - `format=color|gray` (default color)
  - `threshold` (binary models only, default 0.5)

---
## 🐳 Run with Docker

If you want to containerize the API:

```bash
# stop/remove old container if any
docker rm -f seg-api-container 2>/dev/null || true

# rebuild WITHOUT cache so the new Dockerfile takes effect
docker build --no-cache -t seg-api .

# run
docker run -p 8888:8888 --name seg-api-container seg-api
```

## 📖 API Docs

- Swagger: http://localhost:8888/docs  
- ReDoc: http://localhost:8888/redoc  
