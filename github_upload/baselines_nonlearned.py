#!/usr/bin/env python3
"""
baselines_nonlearned.py - Non-learned baselines on the test set (R3.W4). CPU only.

  1. Otsu: grayscale, Gaussian blur, Otsu threshold, morphological opening,
     bounding box of the largest connected component (score 1.0)
  2. Fixed box: the mean training box (centre and size) placed on every test image
     (shows how much of mAP@50 a position prior alone gives)

Same metrics as the published scripts: mAP@50 and mAP@50-95 with
mean_average_precision; precision / recall / F1 with greedy matching at IoU >= 0.5.

Usage (project folder):  python baselines_nonlearned.py
Output: results/nonlearned_baselines.json (+ printed table)
"""

import glob
import json
import os

import cv2
import numpy as np
from mean_average_precision import MetricBuilder

from config import IOU_MATCH, OUTPUT_ROOT, RESULTS_DIR


def read_boxes(label_path, w, h):
    out = []
    if os.path.exists(label_path):
        for line in open(label_path):
            if line.strip():
                c, xc, yc, bw, bh = map(float, line.split())
                out.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h, int(c)])
    return np.array(out) if out else np.zeros((0, 5))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def otsu_box(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h = stats[k, :4]
    return [float(x), float(y), float(x + w), float(y + h)]


def evaluate(name, preds, gts):
    m = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=1)
    tp = fp = fn = 0
    for p, g in zip(preds, gts):
        m.add(p, np.hstack([g, np.zeros((len(g), 2))]) if len(g) else np.zeros((0, 7)))
        matched = set()
        for box in p:
            best, bj = 0.0, -1
            for j, gg in enumerate(g):
                if j not in matched and iou(box, gg) > best:
                    best, bj = iou(box, gg), j
            if best >= IOU_MATCH:
                tp += 1; matched.add(bj)
            else:
                fp += 1
        fn += len(g) - len(matched)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    return {"baseline": name, "mAP50": float(m.value(iou_thresholds=0.5)["mAP"]) * 100,
            "mAP50-95": float(m.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), mpolicy="soft")["mAP"]) * 100,
            "precision": P * 100, "recall": R * 100, "f1": (2 * P * R / (P + R) * 100) if P + R else 0.0,
            "TP": tp, "FP": fp, "FN": fn}


def main():
    # mean training box (normalised centre and size)
    tr = []
    for lp in glob.glob(os.path.join(OUTPUT_ROOT, "train", "labels", "*.txt")):
        tr += [list(map(float, l.split()))[1:] for l in open(lp) if l.strip()]
    xc, yc, bw, bh = np.mean(np.array(tr), axis=0)

    imgs = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "test", "images", "*.*")))
    gts, p_otsu, p_fixed = [], [], []
    for ip in imgs:
        gray = cv2.imread(ip, cv2.IMREAD_GRAYSCALE)
        h, w = gray.shape
        stem = os.path.splitext(os.path.basename(ip))[0]
        gts.append(read_boxes(os.path.join(OUTPUT_ROOT, "test", "labels", stem + ".txt"), w, h))
        b = otsu_box(gray)
        p_otsu.append(np.array([b + [0, 1.0]]) if b else np.zeros((0, 6)))
        p_fixed.append(np.array([[(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h, 0, 1.0]]))

    res = [evaluate("Otsu + largest component", p_otsu, gts),
           evaluate("Fixed mean training box", p_fixed, gts)]
    out = {"n_test_images": len(imgs), "n_test_boxes": int(sum(len(g) for g in gts)),
           "mean_train_box_normalised": {"xc": xc, "yc": yc, "w": bw, "h": bh}, "results": res}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "nonlearned_baselines.json"), "w") as f:
        json.dump(out, f, indent=2, default=float)
    for r in res:
        print(f"{r['baseline']:28s} mAP50={r['mAP50']:6.2f} mAP50-95={r['mAP50-95']:6.2f} "
              f"P={r['precision']:6.2f} R={r['recall']:6.2f} F1={r['f1']:6.2f} TP/FP/FN={r['TP']}/{r['FP']}/{r['FN']}")


if __name__ == "__main__":
    main()
