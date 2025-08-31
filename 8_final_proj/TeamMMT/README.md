# Cat-Dog Image Segmentation API (TeamMMT Project)

📌 **Project Overview**

This project was developed as part of the Women in AI Myanmar Summer Class by TeamMMT.

The goal is to perform semantic segmentation of cats and dogs using transfer learning with U-Net + MobileNet (as encoder) and serve it through a FastAPI web service.

## Team Members

- **Myat Kyay Mone** (ML007) - [GitHub](https://github.com/Kyaymone)
- **Myat Mon Kyaw** (ML009) - [GitHub](https://github.com/myatmon123) 
- **Thu Zar Lin** (ML018) - [GitHub](https://github.com/ThuZarLin)

## Pipeline Summary

### 1. Data Preprocessing
- Annotated images with LabelMe
- Converted annotations into Pascal VOC format using `labelme2voc.py`
- Prepared datasets for training with proper image encoding

### 2. Model Training
- Built U-Net with MobileNet encoder for transfer learning
- Trained on the preprocessed cat and dog dataset
- Model saved as `cats_and_dogs_final_model.keras`

### 3. API Development & Deployment
- Developed a FastAPI service to serve the trained model
- Created REST API endpoint for image segmentation
- Dockerized the application for containerized deployment

## 🛠️ Project Structure

```
TeamMMT/
├── api_endpoint/              # FastAPI application
│   ├── main.py               # FastAPI main application
│   ├── model_work.py         # Model loading and prediction logic
│   └── __init__.py           # Package initialization
├── data_annotated/           # Annotated training data
├── data_dataset_voc/         # Pascal VOC format dataset
├── cats_and_dogs_filtered/   # Filtered training images
├── cats_and_dogs_final_model.keras  # Trained model file
├── labelme2voc.py           # LabelMe to Pascal VOC converter
├── requirements.txt         # Python dependencies
├── Pipfile                  # Pipenv configuration
├── Dockerfile              # Docker configuration
└── README.md               # This file
```

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)
- Pipenv (optional, for local development)

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd TeamMMT
   ```

2. **Build the Docker image**
   ```bash
   docker build -t catdog-seg .
   ```

3. **Run the container**
   ```bash
   # Run in foreground (to see logs)
   docker run -p 8000:8000 catdog-seg
   
   # Or run in background
   docker run -d -p 8000:8000 --name catdog-api catdog-seg
   ```

4. **Access the API**
   - Open your browser and go to: http://localhost:8000/docs
   - Use the interactive API documentation to test image uploads

### Option 2: Local Development

1. **Install dependencies with Pipenv**
   ```bash
   pipenv install
   pipenv shell
   ```

2. **Run the API locally**
   ```bash
   python -m uvicorn api_endpoint.main:app --host 0.0.0.0 --port 8000 --reload
   ```

3. **Access the API**
   - Open: http://127.0.0.1:8000/docs

### Option 3: Using pip and requirements.txt

1. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python -m uvicorn api_endpoint.main:app --host 0.0.0.0 --port 8000
   ```

## 🧹 Data Preparation (For Training)

If you want to retrain the model or prepare new data:

1. **Convert LabelMe annotations to Pascal VOC**
   ```bash
   python labelme2voc.py
   ```

2. **Prepare images for training**
   ```bash
   python prepare_img.ipynb  # Run the Jupyter notebook
   ```

## 📡 API Usage

### Endpoint: `/overlay_image`

**Method:** POST  
**Content-Type:** multipart/form-data  
**Parameters:**
- `file`: Image file (PNG, JPEG, JPG)

**Response:** PNG image with segmentation overlay

### Example using curl:
```bash
curl -X POST "http://localhost:8000/overlay_image" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_cat_or_dog_image.jpg"
```

### Example using Python requests:
```python
import requests

url = "http://localhost:8000/overlay_image"
files = {"file": open("cat_image.jpg", "rb")}
response = requests.post(url, files=files)

# Save the segmented image
with open("segmented_output.png", "wb") as f:
    f.write(response.content)
```

## 🔧 Model Details

- **Architecture:** U-Net with MobileNet encoder
- **Task:** Semantic segmentation (Cat vs Dog vs Background)
- **Input:** RGB images (automatically resized to 512x512)
- **Output:** Color-coded segmentation mask
  - Black: Background
  - Red: Cat
  - Green: Dog

## 🐳 Docker Commands

```bash
# Build image
docker build -t catdog-seg .

# Run container
docker run -p 8000:8000 catdog-seg

# Run in background
docker run -d -p 8000:8000 --name catdog-api catdog-seg

# View logs
docker logs catdog-api

# Stop container
docker stop catdog-api

# Remove container
docker rm catdog-api
```

## 📦 Dependencies

Key packages used in this project:
- `fastapi` - Web framework for building APIs
- `uvicorn` - ASGI server for running FastAPI
- `tensorflow` - Deep learning framework
- `opencv-python-headless` - Computer vision library (headless for Docker)
- `pillow` - Image processing
- `numpy` - Numerical computations
- `python-multipart` - For handling file uploads

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is part of the Women in AI Myanmar Summer Class.

## 👥 Team

**TeamMMT** - Women in AI Myanmar Summer Class Participants

---

For questions or issues, please contact the team members.
