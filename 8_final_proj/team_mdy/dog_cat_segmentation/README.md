# Dog & Cat Segmentation System

This project provides a complete pipeline for semantic segmentation of dog and cat images, including dataset preparation, model training using TensorFlow with transfer learning, and a FastAPI-based inference API.

---

## Project Members

- [Zwe Yaung Ni Tun](https://github.com/zweyaungnitun)
- [Myint Myat Aung](https://github.com/MrMyintMyatAung)
- [Nwe Ni Htoo](https://github.com/nwenihtoo)

---

## 1. Dataset Preparation

### Annotation Conversion

- Annotate images using [LabelMe](https://github.com/wkentaro/labelme).
- Convert LabelMe JSON annotations to VOC format using `labelme2voc.py`.

**Command:**
```bash
python labelme2voc.py <input_labelme_dir> <output_voc_dir> --labels labels.txt
```

**Outputs:**
- `JPEGImages`: Original images
- `SegmentationClass`: PNG masks with class IDs
- `SegmentationClassEncoded`: PNG images with gray-scale encoded
- `SegmentationClassVisualization`: Visualized masks
- `class_names.txt`: List of class names

---

## 2. Model Training

### Architecture

- **U-Net** segmentation model
- **Encoder:** MobileNetV2 (pre-trained on ImageNet)
- **Decoder:** Custom layers for upsampling and segmentation

### Transfer Learning

- The encoder (MobileNetV2) is frozen during initial training.
- Optionally, unfreeze some layers for fine-tuning.


### Training Steps

1. Prepare VOC-format images and masks.
2. Build and compile the U-Net model.
3. Train the model using TensorFlow/Keras.
4. Save the trained model for inference.

---

## 3. FastAPI Inference API

### Features

- Batch and single-image segmentation endpoints
- Overlay and mask visualization
- Returns class percentages and detected classes

### Running the API

#### With Uvicorn

```bash
uvicorn main:app --host 0.0.0.0 --port 8888 --reload
```

#### With Docker

Build the Docker image:
```bash
docker build -t dogcat-segmentation .
```
Run the container:
```bash
docker run -p 8888:8888 dogcat-segmentation
```
---

## License

Educational Use for Women in AI(Myanmar)

---

For details on annotation conversion, see [`labelme2voc.py`](../labelme2voc.py).
For model code, see training scripts and notebooks in this repository.