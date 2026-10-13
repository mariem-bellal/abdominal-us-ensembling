#!/usr/bin/env python3
"""
count_yolo.py - Count images, label files and boxes in abdomen_mri_yolo (read only).

Usage (from the project folder):  python count_yolo.py
"""
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "abdomen_mri_yolo")
IMG = (".jpg", ".jpeg", ".png", ".bmp")

total = 0
for split in ("train", "valid", "test"):
    idir, ldir = os.path.join(ROOT, split, "images"), os.path.join(ROOT, split, "labels")
    imgs = {os.path.splitext(f)[0] for f in os.listdir(idir) if f.lower().endswith(IMG)}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(ldir) if f.endswith(".txt")}
    boxes = sum(sum(1 for line in open(os.path.join(ldir, l + ".txt")) if line.strip()) for l in lbls)
    empty = [l for l in lbls if not any(line.strip() for line in open(os.path.join(ldir, l + ".txt")))]
    total += len(imgs)
    print(f"[{split}] {len(imgs)} images | {len(lbls)} label files | {boxes} boxes | "
          f"images without label: {len(imgs - lbls)} | labels without image: {len(lbls - imgs)} | "
          f"empty label files: {len(empty)}")
print(f"Total images: {total}")
