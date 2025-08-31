# Cat and Dog Image Segmentation Project — Frontend

A lightweight web UI for uploading an image, sending it to the FastAPI backend for **cat/dog segmentation**, and visualizing the predicted mask over the original image.

## Features

- Upload local image (PNG/JPG)
- Call backend inference API
- Show original, mask, and overlay (adjustable opacity)
- Basic error handling & loading states
- CORS-friendly fetch

## Tech Stack

- **React** (works with Vite or Create React App)
- **TypeScript** (optional—remove notes if you’re on JS)
- **Fetch API** for HTTP calls
- Minimal CSS (Tailwind or plain CSS—up to your project)

> Your repo uses Node 16.x in other parts. Keep Node ≥16.x here too.

---

## Getting Started

### 1) Install

```bash
# choose one
npm install
# or
yarn
```

### 2) Environment variables

Create a `.env` file in this `frontend` folder and set your backend URL.

For **Vite**:
```
VITE_API_BASE_URL=http://localhost:8888
```

For **Create React App**:
```
REACT_APP_API_BASE_URL=http://localhost:8888
```

> If you deploy, change this to your public backend URL.

### 3) Run the app

**Vite:**
```bash
npm run dev
# or
yarn dev
```

**Create React App:**
```bash
npm start
# or
yarn start
```

App will start on something like `http://localhost:5173` (Vite) or `http://localhost:3000` (CRA).

---

## Scripts

Common scripts (adjust to your package.json):

```bash
# development
npm start

```

---

## CORS

On the FastAPI side, enable CORS for local dev:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## UI Flow

1. **Upload** an image (cats/dogs).
2. Click **Predict**.
3. See:
   - Original image
   - Predicted mask
   - **Overlay** with adjustable opacity slider

---

