# 🐱🐶 Cat-Dog Image Segmentation (WAI Summer Class Project)



## 📌 Project Overview  
This project was developed as part of the **Women in AI Myanmar Summer Class**.  

The goal is to perform **semantic segmentation** of cats and dogs using **transfer learning with U-Net + MobileNet (as encoder)**.  

**Pipeline Summary**  
1. **Data Preprocessing**  
   - Annotated images with **LabelMe**  
   - Converted annotations into **Pascal VOC format**  
   - Encoded masks to reduce file size  
2. **Model Training**  
   - Built **U-Net with MobileNet encoder** for transfer learning  
   - Trained on the preprocessed dataset  
3. **Deployment**  
   - Developed a **FastAPI** service to serve the trained model  
   - Dockerized the application for containerized deployment

---
### 🛠️ Setup

1. Clone repository
```
git clone https://github.com/yourusername/cat_dog_segmentation.git
cd cat_dog_segmentation
```
2. Install dependencies (Pipenv)
```
pipenv install
pipenv shell
```

### 🧹 Data Preparation

Run preprocessing before training:
```
python preprocessing/labelme2voc.py
python preprocessing/image_encoding.py
```


### 🚀 Serving with FastAPI

Run API locally:
```
uvicorn src.main:app --reload
```
Then open in browser: http://127.0.0.1:8000/docs


### 🐳 Docker Deployment

Build and run the container:
```
docker build -t catdog-seg .
docker run -p 8000:8000 catdog-seg
```

