# 🐱🐶 Cat-Dog Image Segmentation (WAI Summer Class Project by ML_Heros)

## 📌 Project Overview
This project is the **final project of the WAI Practical Machine Learning course**.  

The goal is to perform **semantic segmentation of cats and dogs** using a **U-Net model with MobileNet backbone**.  
A **FastAPI service** allows users to upload images and get segmentation masks overlaid on the original images.

---

## 🚀 Features
- **Semantic segmentation** of cats and dogs  
- **U-Net architecture** with MobileNet encoder  
- Dataset annotated using [LabelMe](https://github.com/wkentaro/labelme)  
- **FastAPI REST API** for serving predictions  
- Outputs an **overlay of the segmentation mask** on the original image  

---

## 🧩 Pipeline Summary
1. **Data Annotation** – Label images of cats and dogs using LabelMe.  
2. **Data Preprocessing** – Convert annotations to VOC format and encode images for model training.  
3. **Model Training** – Train a U-Net model with MobileNet backbone on the preprocessed dataset.  
4. **Prediction** – Input images are passed to the trained model to generate segmentation masks.  
5. **Overlay & Output** – Segmentation masks are overlaid on the original images for visualization.  
6. **Serving** – FastAPI REST API provides endpoints for uploading images and receiving overlaid predictions.  
7. **Deployment** – Run locally or deploy via Docker for reproducible and scalable usage.  

---

## 👥 Team Members

| Avatar | Name | GitHub |
|--------|------|--------|
| <img src="https://github.com/mrmyothet.png" width="50" height="50" style="border-radius:50%;"> | **Myo Thet** | [@mrmyothet](https://github.com/mrmyothet) |
| <img src="https://github.com/Htet-Khant-Linn.png" width="50" height="50" style="border-radius:50%;"> | **Htet Khant Linn** | [@Htet-Khant-Linn](https://github.com/Htet-Khant-Linn) | 
| <img src="https://github.com/EriThinMyat.png" width="50" height="50" style="border-radius:50%;"> | **Aye Thin Myat** | [@EriThinMyat](https://github.com/EriThinMyat) |

---

## 🛠️ Setup

### 1. Clone the repository
    ```bash
    git clone https://github.com/WAI-Practical-ML-Heros/Practical-ML-by-WAI.git
    cd Practical-ML-by-WAI/8_final_proj/ML_Heros
    ```

2. **Install dependencies**
   ```bash
   pipenv install
   pipenv shell
   ```

3. **Run the API locally**
   ```bash
   cd api_endpoint
   uvicorn main:app --reload --host 0.0.0.0 --port 8000
   ```

### Docker Deployment

#### Option 1: Pull from Docker Hub
```bash
    docker pull tenyain/catsdogs-api
    docker run -p 8000:8000 tenyain/catsdogs-api
```

#### Option 2: Build and run locally
```bash
    docker build -t catdog-seg .
    docker run -p 8000:8000 catdog-seg
```