# Segmentation Project

This Project is a deep learning-based image segmentation tool built with Keras and MobileNetV2 as the pretrained encoder. It is designed for easy deployment using Docker and provides a simple API and interactive interface for image segmentation tasks.

**Key Features:**

- Pretrained MobileNetV2 encoder
- Keras model for training and inference
- Dockerized for portability
- REST API and Swagger UI for predictions
- Automatic cleanup of temporary files


## Project Structure

```
segmentation_project/
├── Dockerfile
├── main.py
├── model.py
├── Pipfile
├── Pipfile.lock
├── schemas.py
├── train_notebook.ipynb
├── models/
│   ├── class_names.json
│   └── dog_cat_classification.keras
├── datasets/
└── static/
	├── image_*.png
	├── mask_*.png
	├── overlay_*.png
	└── temp_*.png
```

- **main.py**: Entry point for inference and serving the model
- **model.py**: Model architecture and loading logic
- **schemas.py**: Data schemas for API requests
- **train_notebook.ipynb**: Jupyter notebook for training and experiments
- **models/**: Trained model files and class names
- **Dockerfile**: Containerization for deployment
- **Pipfile & Pipfile.lock**: Python dependencies (pipenv)

## Setup & Usage

### 1. Build Docker Image

Navigate to the project folder and run:

```bash
docker build . -t segmentation_project
```

### 2. Run the API Server

```bash
docker run -p 8000:8000 --env PORT=8000 segmentation_project
```

Or use Docker Desktop to run the container.

### 3. Make Predictions

You can predict segmentation results by:

- Sending a request to the API endpoint
- Using the interactive Swagger UI at [`/docs`](http://localhost:8000/docs) and clicking **Upload Image**

**Outputs:**

1. Original image
2. Segmentation mask
3. Overlay (image + mask)

All outputs are saved in the `static/` folder. This folder is automatically cleaned after each inference or every 10 minutes to avoid storing unnecessary files.

#### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" -F "file=@your_image.png"
```

You can predict segmentation results by either:

- Sending a request directly to the API endpoint, or
- Using the interactive interface available at /docs (Swagger UI) with the Upload Image button.
  Upload an image to get the following outputs: 1. Original image 2. Segmentation mask 3. Overlay (image with mask applied) with links.

These outputs are saved in a static/ folder. The folder is automatically cleaned when a new inference is run or after 10 minutes to avoid storing unnecessary files. 

## Model Files

- `models/dog_cat_classification.keras`: Trained Keras model
- `models/class_names.json`: Class index-to-name mapping

## Dataset

The dataset is annotated by all team members and is available publicly on Kaggle: [Project Dataset on Kaggle](https://www.kaggle.com/datasets/kukulauren/project-dataset/data)

## Model Training Notebook

See `train_notebook.ipynb` for training code and experiments.

## Team Members

- [Myat Phoo Pwint](https://github.com/myatphoopwint926)
- [Htwe Myat Cho](https://github.com/h-myatcho)
- [Khin Myat Noe](https://github.com/kukulauren)

---

For questions or support, please contact any team member via GitHub.
