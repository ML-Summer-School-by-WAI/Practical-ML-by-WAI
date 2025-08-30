## Cat & Dog Sematic Segmentation with U-Net

Contributors:
1. Linn Lat Tun - https://github.com/sigmaava
2. Hsu Pyae Thu - https://github.com/hsupyae04-95
3. May Me Hlwan Myo - https://github.com/kaxlon33

## Project Overview

  This project provides an end-to-end pipeline for image annotation, segmentation, encoding, and classification of cat and dog images using deep learning. The solution leverages U-Net architecture with transfer learning for robust image segmentation and classification. A FastAPI backend serves predictions and segmentation results via RESTful endpoints, and the entire application is containerized with Docker for easy deployment.
  
## Workflow

1.  Image Annotation:
     - Images of cats and dogs are manually annotated to create segmentation masks, enabling precise identification of regions of interest.

2.  Image Segmentation & Encoding:
     - Annotated images are processed and segmented. The segmentation masks are encoded to prepare the dataset for model training.

3.  Model Training:
     - The U-Net model, enhanced with transfer learning, is trained on the prepared dataset to perform accurate image segmentation and classification.

4.  API Development:
     - A FastAPI application is built to provide endpoints for image classification and segmentation. Users can upload images and receive predictions and segmentation masks in real time.
        
5.  Containerization:
     - The project is packaged with Docker, allowing seamless deployment and scalability. The Docker container exposes the FastAPI service for easy integration.

##  Getting Started

1.  Install Dependencies
       - pip install -r requirement.txt (or) pipenv install

2.  Prepare the Dataset
       - Place your annotated images and segmentation masks in the dataset/cat_and_dog_dataset/images and dataset/cat_and_dog_dataset/masks directories.
        
3.  Train the Model (Optional)
      - use the provided pre-trained weights in ml_models/unet_model_ml020.h5.

4.  Run the FastAPI Server
     - Start the API server to serve predictions and segmentation results: pipenv run python main.py

5.  Use Docker for Deployment
        To build and run the application in a Docker container:
    - docker build -t cat-dog-segmentation 
    - docker run -p 8000:8000 cat-dog-segmentation

6.  Access the API
     - Use the endpoints to upload images and receive classification and segmentation results.
