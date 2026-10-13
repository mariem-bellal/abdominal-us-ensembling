#!/usr/bin/env python3
"""
saturation_all.py - Colour-Doppler check on ALL test detections (R4.4). CPU only, no GPU.

The paper checked colour (Doppler) overlap on 8 detections (4 images x 2 models).
This script repeats the check on every test image and every detection of the
6 single models (3 YOLOv8s seeds, 3 RT-DETR-l seeds), using the predictions
already cached by eval_unified.py.

Why "colour inside the box" and not an inside/outside ratio: many frames carry
coloured elements OUTSIDE the ultrasound sector (colour scale bar, coloured
on-screen buttons, body-marker icons). They inflate any out-of-box average, so
we only measure colour inside boxes, which lie within the sector.

Definitions
  colour pixel     : HSV saturation > 80 and value > 50 (0-255); greyscale B-mode has
                     saturation near 0, colour Doppler does not
  in-box colour    : fraction of the box's pixels that are colour pixels
  "on colour"      : in-box colour > 5%
  kept detection   : confidence >= the model's validation-tuned threshold
                     (results/unified_eval/instances.csv)
  TP / FP          : greedy one-to-one matching, IoU >= 0.5 with a ground-truth box
  Doppler target   : ground-truth box with in-box colour > 1%

Shortcut reading: if the detectors were drawn to colour Doppler, false positives
would sit on colour more often than ground-truth boxes do, and recall would be
clearly higher for Doppler targets than for the others.

Usage (project folder, after eval_unified.py):  python saturation_all.py
Output: results/saturation_all.json (+ per-detection and per-box CSV files)
"""

import glob
import json
import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image

from config import OUTPUT_ROOT, RESULTS_DIR

UE = os.path.join(RESULTS_DIR, "unified_eval")
S_MIN, V_MIN = 80, 50
ON_COLOUR, DOPPLER_GT = 0.05, 0.01
MODELS = ["Y0", "Y1", "Y2", "R0", "R1", "R2"]
ARCH = {"Y": "YOLOv8s", "R": "RT-DETR-l"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def gt_boxes(stem):
    p = os.path.join(OUTPUT_ROOT, "test", "labels", stem + ".txt")
    out = []
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                _, xc, yc, w, h = map(float, line.split())
                out.append([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])
    return out


def colour_share(mask, b):
    h, w = mask.shape
    x0, y0 = max(int(b[0] * w), 0), max(int(b[1] * h), 0)
    x1, y1 = min(max(int(np.ceil(b[2] * w)), x0 + 1), w), min(max(int(np.ceil(b[3] * h)), y0 + 1), h)
    return float(mask[y0:y1, x0:x1].mean())


def q(s):
    return {"n": int(len(s)), "median": float(s.median()) if len(s) else None,
            "p90": float(s.quantile(.9)) if len(s) else None,
            "share_on_colour": float((s > ON_COLOUR).mean()) if len(s) else None,
            "count_on_colour": int((s > ON_COLOUR).sum())}


def main():
    inst = pd.read_csv(os.path.join(UE, "instances.csv"))
    single = inst[inst.config.isin(["YOLOv8s (single)", "RT-DETR-l (single)"])]
    thr = dict(zip(single.members, single.val_threshold))
    preds = {}
    for m in MODELS:
        with open(os.path.join(UE, f"preds_{m}_test_notta.pkl"), "rb") as f:
            preds[m] = pickle.load(f)

    det_rows, gt_rows = [], []
    for p in sorted(glob.glob(os.path.join(OUTPUT_ROOT, "test", "images", "*.*"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        hsv = np.asarray(Image.open(p).convert("RGB").convert("HSV")).astype(float)
        mask = (hsv[:, :, 1] > S_MIN) & (hsv[:, :, 2] > V_MIN)
        gts = gt_boxes(stem)
        found = {m: set() for m in MODELS}
        for m in MODELS:
            d = preds[m].get(stem, {"boxes": [], "scores": []})
            for b, s in sorted(zip(d["boxes"], d["scores"]), key=lambda x: -x[1]):
                if s < thr[m]:
                    continue
                best, bj = 0.0, -1
                for j, g in enumerate(gts):
                    if j not in found[m] and iou(b, g) > best:
                        best, bj = iou(b, g), j
                tp = best >= 0.5
                if tp:
                    found[m].add(bj)
                det_rows.append({"image": stem, "model": m, "arch": ARCH[m[0]], "score": s,
                                 "TP": tp, "in_box_colour": colour_share(mask, b)})
        for j, g in enumerate(gts):
            row = {"image": stem, "box": j, "in_box_colour": colour_share(mask, g),
                   "image_colour_fraction": float(mask.mean())}
            row.update({f"found_{m}": j in found[m] for m in MODELS})
            gt_rows.append(row)

    dets, gtd = pd.DataFrame(det_rows), pd.DataFrame(gt_rows)
    gtd["doppler_target"] = gtd.in_box_colour > DOPPLER_GT
    res = {"n_test_images": int(gtd.image.nunique()), "n_gt_boxes": int(len(gtd)),
           "n_kept_detections": int(len(dets)),
           "definition": {"S_min": S_MIN, "V_min": V_MIN, "on_colour_if_in_box_colour_above": ON_COLOUR,
                          "doppler_target_if_gt_in_box_colour_above": DOPPLER_GT},
           "gt_boxes_in_box_colour": q(gtd.in_box_colour),
           "n_doppler_targets": int(gtd.doppler_target.sum()), "per_arch": {}}
    for code, arch in ARCH.items():
        g = dets[dets.arch == arch]
        ms = [m for m in MODELS if m[0] == code]
        rec = {}
        for name, sub in (("doppler_targets", gtd[gtd.doppler_target]), ("other_targets", gtd[~gtd.doppler_target])):
            r = [float(sub[f"found_{m}"].mean()) if len(sub) else None for m in ms]
            rec[name] = {"boxes": int(len(sub)), "recall_per_seed": r,
                         "recall_mean": float(np.mean(r)) if len(sub) else None}
        res["per_arch"][arch] = {"TP_in_box_colour": q(g[g.TP].in_box_colour),
                                 "FP_in_box_colour": q(g[~g.TP].in_box_colour), "recall": rec}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "saturation_all.json"), "w") as f:
        json.dump(res, f, indent=2)
    dets.to_csv(os.path.join(RESULTS_DIR, "saturation_all_detections.csv"), index=False)
    gtd.to_csv(os.path.join(RESULTS_DIR, "saturation_all_gtboxes.csv"), index=False)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
