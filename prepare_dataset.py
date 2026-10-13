"""
Step 4 — Convert CSV annotations to YOLO format + generate data.yaml.

    python prepare_dataset.py
"""

import os
import glob
import shutil
import yaml
import pandas as pd
from pathlib import Path
from config import (
    DATASET_ROOT, OUTPUT_ROOT, SPLITS,
    CSV_CANDIDATES, COLUMN_MAPPING,
)


def find_csv(split_dir: str) -> str:
    for name in CSV_CANDIDATES:
        candidate = os.path.join(split_dir, name)
        if os.path.exists(candidate):
            return candidate
    csvs = glob.glob(os.path.join(split_dir, "*.csv"))
    if csvs:
        return csvs[0]
    raise FileNotFoundError(f"No annotation CSV found in {split_dir}")


def build_class_list(all_csvs: list) -> list:
    classes = set()
    for csv_path in all_csvs:
        df = pd.read_csv(csv_path)
        classes.update(df[COLUMN_MAPPING["class"]].unique().tolist())
    return sorted(classes)


def convert_split(split: str, class_to_id: dict):
    split_dir = os.path.join(DATASET_ROOT, split)
    csv_path = find_csv(split_dir)
    df = pd.read_csv(csv_path)

    out_img_dir = os.path.join(OUTPUT_ROOT, split, "images")
    out_lbl_dir = os.path.join(OUTPUT_ROOT, split, "labels")
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_lbl_dir, exist_ok=True)

    grouped = df.groupby(COLUMN_MAPPING["filename"])
    n_images, n_boxes = 0, 0

    for filename, group in grouped:
        src_img = os.path.join(split_dir, filename)
        if not os.path.exists(src_img):
            print(f"[WARN] missing image, skipped: {src_img}")
            continue

        dst_img = os.path.join(out_img_dir, filename)
        if not os.path.exists(dst_img):
            shutil.copy2(src_img, dst_img)

        lines = []
        for _, row in group.iterrows():
            w = float(row[COLUMN_MAPPING["width"]])
            h = float(row[COLUMN_MAPPING["height"]])
            xmin = float(row[COLUMN_MAPPING["xmin"]])
            ymin = float(row[COLUMN_MAPPING["ymin"]])
            xmax = float(row[COLUMN_MAPPING["xmax"]])
            ymax = float(row[COLUMN_MAPPING["ymax"]])
            cls_name = row[COLUMN_MAPPING["class"]]
            cls_id = class_to_id[cls_name]

            x_center = min(max(((xmin + xmax) / 2.0) / w, 0), 1)
            y_center = min(max(((ymin + ymax) / 2.0) / h, 0), 1)
            bw = min(max((xmax - xmin) / w, 0), 1)
            bh = min(max((ymax - ymin) / h, 0), 1)

            lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {bw:.6f} {bh:.6f}")
            n_boxes += 1

        label_path = os.path.join(out_lbl_dir, Path(filename).stem + ".txt")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        n_images += 1

    print(f"[{split}] {n_images} images converted, {n_boxes} bounding boxes.")


def main():
    split_dirs = [os.path.join(DATASET_ROOT, s) for s in SPLITS]
    csvs = [find_csv(d) for d in split_dirs]

    classes = build_class_list(csvs)
    class_to_id = {c: i for i, c in enumerate(classes)}
    print(f"Classes detected ({len(classes)}): {classes}")

    for split in SPLITS:
        convert_split(split, class_to_id)

    data_yaml = {
        "path": OUTPUT_ROOT,
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(classes),
        "names": classes,
    }
    yaml_path = os.path.join(OUTPUT_ROOT, "data.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"\ndata.yaml written to {yaml_path}")
    print("Done. You can now run train_yolov8_baseline.py")


if __name__ == "__main__":
    main()
