# 🐶🐱 Dog & Cat Image Semantic Segmentation Project

Hi there! Welcome to our **Dog & Cat Image Segmentation Project**! ✨

We built a system that can **magically separate dogs and cats in images** using deep learning. You can even send it an image via an **API** and get back a segmented picture! All packaged neatly in **Docker** so it runs anywhere. 🚀

---

## 👩‍💻 **Team Members**

- [**Khin Chaw Lai Lai Tun**](https://github.com/KhinChaw)
- [**Myat ThinZar Hlaing**](https://github.com/MyatThinzar1259)
- [**Min Thiha Tun**](https://github.com/MinThihaTun3012)

## 💡 **Project Goal**

The goal of this project is to:

- **Segment dogs and cats** in images using a U-Net model.
- Use **transfer learning** to improve segmentation accuracy and training speed.
- Serve the model via **FastAPI** to easily get segmented output images.
- Make the project **Docker-ready** for easy deployment and sharing.

---

## 🐾 **Pipeline (How it Works)**

1. **Data Preparation**

   - Load dog & cat images.
   - Encode masks for segmentation.
   - Split into training & validation sets.

2. **Model Training**

   - U-Net architecture + pretrained encoder for **transfer learning**.
   - Train the model to segment dogs & cats.

3. **API Development**

   - FastAPI receives images.
   - Model predicts segmentation.
   - Returns segmented output images!

4. **Docker Deployment**
   - Everything runs in a container 🐳  
     !

---

## 📁 **Project Structure**

Based on the folder structure in the image, here is a breakdown you can use for your README file.

Team-KMM/
├── 📂 api_endpoints/ # FastAPI application and model serving logic
│ ├── 📂 pycache/ # Python cache files
│ ├── 🐍 main.py # Main FastAPI entry point
│ └── 🐍 model_work.py # Code for model inference
│
├── 📂 models/ # Directory for trained models
│ └── 🧠 cat_and_dog_unet.keras # Trained UNet model
│
├── 📂 notebooks/ # Jupyter notebooks for data preparation & training
│ ├── 📓 image_encoding.ipynb
│ ├── 🐍 labelme2voc.py
│ ├── 🖼️ model.png
│ ├── 📓 prepare_img.ipynb
│ ├── 📓 read_pascal_voc.ipynb
│ ├── 📄 README.md # Notebook-specific README
│ ├── 🖼️ tf_lr_model_architecture.png
│ └── 📓 u_net_transfer_learning.ipynb
│
├── 🐳 Dockerfile # Instructions to build the Docker image
├── 📦 cat_and_dog_dataset.zip # Dataset archive
├── 📄 Pipfile # Project dependencies (primary)
├── 📄 Pipfile.lock # Locked dependencies
├── 📄 README.md # Main project README
└── 📄 requirements.txt # Dependency list

## 🛠 **Installation**

### Clone Our Cutie Repo

```bash
git clone https://github.com/ML-Summer-School-by-WAI/Practical-ML-by-WAI/tree/main/8_final_proj






```
