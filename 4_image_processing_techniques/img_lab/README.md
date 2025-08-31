```bash
cd data_annotated

labelme data_annotated/ --labels labels.txt
# OR
labelme
```

### types of image dataset
classification dataset
object detection dataset
segmentation dataset
 -  instance segmentation
 -  semantic segmentation




### install labelme
```bash
pip install labelme
```




##### Convert to VOC-format Dataset
- Masks are indexed PNGs (e.g., class "cat" → ID 1).
- Optional: .npy versions, visualization images, instance masks.
```bash

python labelme2voc.py data_annotated data_dataset_voc --labels data_annotated/labels.txt


output_dir/
├── JPEGImages/
│   └── image1.jpg
├── SegmentationClass/
│   └── image1.png         # mask per pixel (each pixel = class ID)
├── class_names.txt



```
### Convert to COCO-format Dataset
- Single .json contains:
    - images
    - annotations
    - categories
- Each object has bbox, segmentation, category_id, and id.

```bash
pip install pycocotools
python labelme2coco.py data_annotated data_dataset_coco --labels data_annotated/labels.txt

output_dir/
├── images/
│   └── image1.jpg
├── annotations/
│   └── instances_train2017.json

```





|               | VOC-style from LabelMe    | COCO-style from LabelMe          |
| ------------- | ------------------------- | -------------------------------- |
| Output Format | PNG masks, optional XML   | COCO-style JSON                  |
| Use Case      | Semantic Segmentation     | Instance Segmentation, Detection |
| Conversion    | `labelme_json_to_dataset` | `labelme2coco.py`                |
