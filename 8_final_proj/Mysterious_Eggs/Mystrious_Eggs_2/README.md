# 🐱🐶 Dogs and Cats Segmentation API

A deep learning-powered Flask API that performs semantic segmentation on cat and dog images, built with TensorFlow and U-Net architecture.

## 🌟 Features

- **Real-time Image Segmentation**: Upload cat/dog images and get pixel-level segmentation masks
- **RESTful API**: Easy-to-use endpoints for integration
- **Model Auto-loading**: Automatically loads the best available trained model
- **Docker Support**: Containerized deployment for easy scaling
- **Confidence Scores**: Get prediction confidence for each pixel class
- **Visual Output**: Base64-encoded mask images for visualization

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- pipenv (recommended) or pip
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd dogs_and_cat_filter
```

2. **Install dependencies using pipenv**
```bash
pipenv install
pipenv shell
```

*Or using pip:*
```bash
pip install -r requirements.txt
```

3. **Run the Flask application**
```bash
python app.py
```

The API will be available at `http://localhost:5001`

## 📋 API Endpoints

### Health Check
```http
GET /health
```
Returns API status and model information.

**Response:**
```json
{
  "status": "healthy",
  "model_status": "Model loaded from cat_dog_segmentation_model.keras",
  "tensorflow_version": "2.20.0",
  "endpoints": {
    "/": "Health check",
    "/segment": "POST - Image segmentation",
    "/model/status": "GET - Model information",
    "/model/load": "POST - Load specific model",
    "/train/status": "GET - Training status"
  }
}
```

### Image Segmentation
```http
POST /segment
```
Upload an image and get segmentation results.
![prediction_results.png]
**Request:**
- Content-Type: `multipart/form-data`
- Body: Form data with key `image` (file upload)

**Response:**
```json
{
  "success": true,
  "confidence_scores": {
    "background": 0.7047361135482788,
    "pet": 0.2952638864517212
  },
  "mask_base64": "iVBORw0KGgoAAAANSUhEUgAAAIA...",
  "input_shape": [1, 128, 128, 3],
  "mask_shape": [128, 128, 3]
}
```

### Model Status
```http
GET /model/status
```
Get detailed information about the loaded model.

### Training Status
```http
GET /train/status
```
Check available models and training artifacts.

## 🧪 Testing with Postman

1. **Health Check:**
   - Method: `GET`
   - URL: `http://localhost:5001/health`

2. **Image Segmentation:**
   - Method: `POST`
   - URL: `http://localhost:5001/segment`
   - Body: `form-data` with key `image` (select a cat/dog image file)

3. **Expected Results:**
   - Background confidence: Percentage of image that's background
   - Pet confidence: Percentage of image that contains pet
   - Base64 mask: Visual segmentation mask (decode to see the result)

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the Docker image
docker build -t dogs-cats-segmentation .

# Run the container
docker run -p 5001:5001 dogs-cats-segmentation
```

### Using Docker Compose

```bash
# Start the service
docker-compose up

# Run in background
docker-compose up -d

# Stop the service
docker-compose down
```


## 🔧 Model Information

- **Architecture**: U-Net with MobileNetV2 backbone
- **Input Size**: 128x128x3 (RGB images)
- **Output Classes**: 2 (Background, Pet)
- **Framework**: TensorFlow/Keras
- **Model Format**: `.keras` files

### Available Models

The API automatically loads the best available model from:
- `cat_dog_segmentation_model.keras`
- `best_model.keras` 
- `saved_models/*.keras`

## 📁 Project Structure

```
dogs_and_cat_filter/
├── app.py                          # Main Flask API application
├── segmentation_model.py           # U-Net model definition
├── local_data_pipeline.py          # Data loading and preprocessing
├── train.py                        # Model training script
├── evaluate.py                     # Model evaluation
├── Dockerfile                      # Docker configuration
├── docker-compose.yml             # Docker Compose setup
├── Pipfile                         # Python dependencies
├── .gitignore                      # Git ignore rules
├── README.md                       # This file
├── dataset/                        # Training data (ignored by Git)
├── saved_models/                   # Trained models (ignored by Git)
└── test_results/                   # Test outputs (ignored by Git)
```

## 🎯 Training Your Own Model

1. **Prepare your dataset**
   - Place images in `dataset/cat_and_dog_dataset/`
   - Ensure you have corresponding segmentation masks

2. **Run training**
```bash
python train_cat_dog_segmentation.py
```

3. **Evaluate the model**
```bash
python evaluate.py
```

## 🛠️ Development

### Dependencies

- **TensorFlow** ≥2.13.0 - Deep learning framework
- **Flask** - Web framework for API
- **OpenCV** ≥4.8.0 - Image processing
- **Pillow** ≥10.0.0 - Image handling
- **NumPy** ≥1.24.0 - Numerical computing
- **Scikit-learn** ≥1.3.0 - Machine learning utilities
- **Gunicorn** ≥21.0.0 - WSGI server for production




