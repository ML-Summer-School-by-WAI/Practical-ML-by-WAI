# Cat and Dog Image Segmentation Project

## Team Members
- **[Nwe Ni Oo Wai](https://github.com/nwenioowai)**
- **[Hein Htet Nyi](https://github.com/HeinHtetNyi)**
- **[Ngwe Sin Linn Latt](https://github.com/NgweSin16)**
- **[Aye Chan Htun Naing](https://github.com/ayechanhtunnaing)**

## Project Overview
This project focuses on image segmentation of cats and dogs using deep learning techniques. We utilize a dataset of cat and dog images to train a model that can accurately segment these animals from the background. The project is implemented using U-Net architecture with transfer learning.

---

## Project Structure

```
old_ucsyers/
│
├── README.md
├── requirements.txt
├── Pipfile
├── Dockerfile
│
├── data/                        # Raw images and segmentation masks
│   ├── images/                  # Cat and dog images
│   └── masks/                   # Corresponding segmentation masks
│
├── notebooks/                   # Jupyter notebooks and scripts
│   ├── prepare_img.ipynb        # Data preparation and visualization
│   ├── labelme2voc.py           # Script to convert LabelMe annotations
│   ├── image_encoding.ipynb     # Mask encoding and processing
│   └── read_parcal_voc.ipynb    # VOC reading utilities
│
└── src/                         # Source code and API
    ├── README.md                # API usage and setup instructions
    ├── main.py                  # FastAPI app for segmentation inference
    ├── final_model.keras        # Trained segmentation model (download separately)
```

---

## Data Documentation

- **data/images/**: Contains the original cat and dog images used for training and evaluation.
- **data/masks/**: Contains the corresponding segmentation masks for each image. Each mask is a per-pixel annotation where each pixel value represents a class (e.g., cat, dog, background).
- Data can be prepared or converted using the `notebooks/labelme2voc.py` script.

---

## Model Documentation

- **src/models/**: Contains the implementation of the U-Net architecture and any other model variants.
- **src/final_model.keras**: The trained U-Net model weights.  
  Download from [Google Drive](https://drive.google.com/file/d/1lbS_cL5HbV-P6PfOvGHTuAFs88H3w-ty/view?usp=sharing) and place in the `src/` directory.
- **src/main.py**: FastAPI application for serving the segmentation model as an API.

---

## Notebook Documentation

- **notebooks/prepare_img.ipynb**:  
  Jupyter notebook for data exploration, visualization, and image preparation.
- **notebooks/labelme2voc.py**:  
  Script to convert LabelMe JSON annotations to VOC format for training.
- **notebooks/image_encoding.ipynb**:  
  Notebook for mask encoding and processing.
- **notebooks/read_parcal_voc.ipynb**:  
  Notebook for reading and handling VOC format data.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/ML-Final-Project.git
   cd ML-Final-Project/8_final_proj/old_ucsyers
   ```

2. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

3. **Download the trained model:**
   - Download `final_model.keras` from [Google Drive](https://drive.google.com/file/d/1lbS_cL5HbV-P6PfOvGHTuAFs88H3w-ty/view?usp=sharing)
   - Place it in the `src/` directory.

---

## Running the API

1. **Start the FastAPI server:**
   ```sh
   cd src
   uvicorn main:app --host 0.0.0.0 --port 8888 --reload
   ```
   The API will be available at [http://localhost:8888](http://localhost:8888).

2. **Example API Usage:**
   - Overlay output:
     ```sh
     curl -X POST -F "file=@cat.jpg" "http://localhost:8888/segment" --output overlay.png
     ```
   - Mask (color):
     ```sh
     curl -X POST -F "file=@cat.jpg" "http://localhost:8888/segment/mask?format=color" --output mask_color.png
     ```
   - Mask (gray):
     ```sh
     curl -X POST -F "file=@cat.jpg" "http://localhost:8888/segment/mask?format=gray" --output mask_gray.png
     ```

   See [`src/README.md`](src/README.md) for full API documentation.

---