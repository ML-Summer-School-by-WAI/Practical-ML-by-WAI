# Semantic Segmentation API

A FastAPI-based web service for semantic segmentation of cats and dogs using a U-Net model.

## Features

- **Semantic Segmentation**: Upload images containing cats and dogs to get pixel-level segmentation
- **Color Overlay**: Returns images with colored overlays (green for cats, red for dogs, black for background)
- **RESTful API**: Simple HTTP endpoints for easy integration
- **Model Management**: Automatic model loading and lifecycle management

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure the model file exists at `dataset/mysterious_eggs_model.h5`

## Running the Server

### Option 1: Using the run script
```bash
python run_server.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The server will start at `http://localhost:8000`

## API Endpoints

### POST /overlay_image
Upload an image file to get semantic segmentation with colored overlay.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: Image file (PNG, JPEG, JPG)

**Response:**
- Content-Type: image/png
- Body: PNG image with segmentation overlay

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/overlay_image" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg" \
     --output result.png
```

### GET /
Root endpoint with API information.

### GET /health
Health check endpoint to verify model status.

## Color Coding

- **Green**: Cat pixels
- **Red**: Dog pixels  
- **Black**: Background pixels

## Model Details

- **Architecture**: U-Net for semantic segmentation
- **Input Size**: 128x128 pixels
- **Classes**: 3 (background, cat, dog)
- **Model File**: `dataset/mysterious_eggs_model.h5`

## Error Handling

The API returns appropriate error messages for:
- Invalid file types
- Model loading failures
- Image processing errors
