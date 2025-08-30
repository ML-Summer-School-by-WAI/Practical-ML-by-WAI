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
├── api_endpoints/ # Contains the FastAPI application and model serving logic
│ ├── **pycache**/
│ ├── main.py # Main FastAPI entry point
│ └── model_work.py # Code for model inference
├── models/ # Directory for storing trained models
│ ├── cat_and_dog_unet.keras# The trained UNet model
├── notebooks/ # Jupyter notebooks for data preparation and model training
│ ├── image_encoding.ipynb
│ ├── labelme2voc.py
│ ├── model.png
│ ├── prepare_img.ipynb
│ ├── read_pascal_voc.ipynb
│ ├── README.md # Specific README for the notebooks directory
│ ├── tf_lr_model_architecture.png
│ └── u_net_transfer_learning.ipynb
├── Dockerfile # Instructions to build the Docker image
└── cat_and_dog_dataset.zip. # Datasets Folder
├── Pipfile # Project dependencies (primary)
├── Pipfile.lock # Project dependencies (locked versions)
├── README.md # Main project README
└── requirements.txt # A list of project dependencies

## 🛠 **Installation**

### Clone Our Cutie Repo

```bash
git clone https://github.com/ML-Summer-School-by-WAI/Practical-ML-by-WAI/tree/main/8_final_proj






```
