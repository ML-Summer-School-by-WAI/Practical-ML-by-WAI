#!/usr/bin/env python

from __future__ import print_function

import argparse
import glob
import os
import os.path as osp
import sys

import imgviz
import numpy as np

import labelme


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("input_dir", help="Input annotated directory")
    parser.add_argument("output_dir", help="Output dataset directory")
    parser.add_argument(
        "--labels", help="Labels file or comma separated text", required=True
    )
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    print("Creating dataset:", args.output_dir)

    if osp.exists(args.labels):
        with open(args.labels) as f:
            labels = [label.strip() for label in f if label]
    else:
        labels = [label.strip() for label in args.labels.split(",")]

    class_names = []
    class_name_to_id = {}
    for i, label in enumerate(labels):
        class_id = i - 1  # starts with -1
        class_name = label.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        elif class_id == 0:
            assert class_name == "_background_"
        class_names.append(class_name)
    class_names = tuple(class_names)
    print("class_names:", class_names)
    out_class_names_file = osp.join(args.output_dir, "class_names.txt")
    with open(out_class_names_file, "w") as f:
        f.writelines("\n".join(class_names))
    print("Saved class_names:", out_class_names_file)

    # Find all JSON files recursively
    json_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    
    json_files = sorted(json_files)
    print(f"Found {len(json_files)} JSON files")

    for filename in json_files:
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_clsp_file = osp.join(args.output_dir, base + ".png")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        cls, ins = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        ins[cls == -1] = 0  # ignore it.

        # Save only class label as PNG
        labelme.utils.lblsave(out_clsp_file, cls)
        print(f"Saved segmentation mask: {out_clsp_file}")

    print(f"Completed! Generated {len(json_files)} segmentation masks in {args.output_dir}")


if __name__ == "__main__":
    main()


# python labelme2segmentationclass.py "dataset/cat_and_dog_dataset/data/cat and dog" "dataset/cat_and_dog_dataset/SegmentationClass" --labels labels.txt