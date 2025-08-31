## 👩‍💻 Team Members

- [Wai Yan Myo](https://github.com/waiyanmyodev)
- [Phyo Myat Oo](https://github.com/Phyo-Myat-Oo) 
- [Thant Sin Tun](https://github.com/ThantSinTun009)
- [May Thazin Htun](https://github.com/maythazinhtunn)


# Mysterious Eggs - Semantic Segmentation API

A FastAPI-based semantic segmentation service for cats and dogs using a U-Net deep learning model. This application provides real-time image segmentation with color-coded overlay visualization.

## 🎯 Features

- **Semantic Segmentation**: Identifies and segments cats and dogs in images
- **Real-time API**: Fast inference through FastAPI endpoints
- **Color-coded Visualization**: 
  - Green overlay for cats
  - Red overlay for dogs
  - Black for background
- **Health Monitoring**: Built-in health check endpoints
- **Docker Support**: Containerized deployment ready

## 🏗️ Architecture

- **Backend**: FastAPI with Python 3.11
- **ML Model**: U-Net architecture trained on cat and dog dataset
- **Image Processing**: OpenCV and PIL for image manipulation
- **Model Format**: TensorFlow/Keras (.h5 format)

## 📋 Requirements

### System Requirements
- Python 3.11+
- Docker (optional, for containerized deployment)

### Python Dependencies
All dependencies are managed through Pipfile. Key packages include:
- FastAPI
- TensorFlow
- OpenCV
- Pillow
- Uvicorn
- Python Multipart
- Pydantic

See `Pipfile` for complete dependency list.

## 🚀 Quick Start

### Local Development

1. **Clone the repository and navigate to the project directory**
   ```bash
   cd Mysterious_Eggs
   ```

2. **Install dependencies using pipenv**
   ```bash
   # Install pipenv if you haven't already
   pip install pipenv
   
   # Install project dependencies
   pipenv install
   
   # Activate the virtual environment
   pipenv shell
   ```

3. **Ensure the model file is present**
   - The trained model should be located at `dataset/mysterious_eggs_model.h5`
   - If missing, train the model using the provided Jupyter notebooks

4. **Run the server**
   ```bash
   # If you're in the pipenv shell
   python run_server.py
   
   # Or run directly with pipenv
   pipenv run python run_server.py
   ```

5. **Access the API**
   - Server runs on: `http://localhost:8000`
   - Interactive docs: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/health`

### Docker Deployment

1. **Build the Docker image**
   ```bash
   docker build -t mysterious-eggs-api .
   ```

2. **Run the container**
   ```bash
   docker run -p 8000:8000 mysterious-eggs-api
   ```

3. **Access the application**
   - API: `http://localhost:8000`
   - Health check: `http://localhost:8000/health`

## 📡 API Endpoints

### `GET /`
Returns API information and available endpoints.

**Response:**
```json
{
  "message": "Semantic Segmentation API",
  "description": "Upload an image to get semantic segmentation overlay for cats and dogs",
  "endpoints": {
    "/overlay_image": "POST - Upload an image file to get segmentation overlay"
  }
}
```

### `POST /overlay_image`
Upload an image file to get semantic segmentation with color overlay.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file (PNG, JPEG, JPG)

**Response:**
- Content-Type: `image/png`
- Body: Processed image with segmentation overlay

**Example using curl:**
```bash
curl -X POST "http://localhost:8000/overlay_image" \
  -H "accept: image/png" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@your_image.jpg" \
  --output result.png
```

### `GET /health`
Health check endpoint to verify model loading status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🎨 Color Coding

The segmentation overlay uses the following color scheme:
- **Black (0, 0, 0)**: Background
- **Green (0, 255, 0)**: Cat
- **Red (255, 0, 0)**: Dog

## 🔧 Development

### Project Structure
```
Mysterious_Eggs/
├── src/
│   ├── __init__.py
│   ├── main.py                    # FastAPI application
│   └── semantic_segmentation.py   # ML model wrapper
├── dataset/
│   ├── mysterious_eggs_model.h5   # Trained U-Net model
│   └── cat_and_dog_dataset/       # Training dataset
├── run_server.py                  # Server startup script
├── Pipfile                        # Pipenv configuration
├── Dockerfile                     # Docker configuration
└── README.md                      # This file
```

### Model Training
The project includes Jupyter notebooks for model training:
- `u_net.ipynb`: Main U-Net training notebook
- `u_net_transfer_learning.ipynb`: Transfer learning approach
- `image_encoding.ipynb`: Data preprocessing and encoding
- `find_numClass.ipynb`: Dataset analysis

### Data Processing
- `labelme2segmentationclass.py`: Convert LabelMe annotations to segmentation masks
- `image_encoding.py`: Image preprocessing utilities
- `labels.txt`: Class labels definition

## 🐛 Troubleshooting

### Common Issues

1. **Model not found error**
   - Ensure `dataset/mysterious_eggs_model.h5` exists
   - Check file permissions

2. **Import errors**
   - Verify all dependencies are installed: `pipenv install`
   - Ensure you're in the pipenv environment: `pipenv shell`
   - Check Python version (requires 3.11+)

3. **OpenCV issues on Linux**
   - Install system dependencies: `apt-get install libgl1-mesa-glx`

4. **Memory issues**
   - The model requires sufficient RAM for TensorFlow operations
   - Consider reducing batch size or image resolution

### Docker Issues

1. **Build failures**
   - Ensure Docker has sufficient memory allocated
   - Check internet connection for package downloads

2. **Runtime errors**
   - Verify the model file is included in the Docker image
   - Check container logs: `docker logs <container_id>`

## 📊 Model Information

- **Architecture**: U-Net with encoder-decoder structure
- **Input Size**: 128x128 pixels
- **Classes**: 3 (background, cat, dog)
- **Framework**: TensorFlow/Keras
- **Training Dataset**: Custom cat and dog images with segmentation masks

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Practical ML course materials.

## 🙏 Acknowledgments

- Dataset preparation using LabelMe annotation tool
- U-Net architecture implementation based on TensorFlow/Keras
- FastAPI framework for REST API development
