#!/usr/bin/env python3
"""
wa_check.py - Checks the weight-averaging ("SWA") failure reported in the paper
(R3.W6, R3.Q5, R4.9). No training; uses the 3 trained YOLOv8s checkpoints.

What it does
  0. describes each checkpoint: pretrained start point (train_args.model),
     whether EMA weights are stored, precision, Ultralytics version
  1. re-evaluates the saved averaged model (yolov8_swa/weights/best.pt)
  2. endpoint check: the "average" with alpha = 0 and alpha = 1 must give the same
     mAP as seed 42 and seed 123 themselves; if not, the averaging code is wrong
  3. linear path seed 42 -> seed 123 (alpha = 0, .25, .5, .75, 1), with and without
     re-estimating the BatchNorm statistics on training images
  4. uniform average of the 3 seeds, with and without BatchNorm re-estimation,
     evaluated on validation and test

Usage (project folder, environment with ultralytics, GPU recommended):
    python wa_check.py
Outputs: results/wa_check/wa_check.csv, results/wa_check/checkpoints.json
"""

import argparse
import copy
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

from config import BASELINE_DIR, DATA_YAML, OUTPUT_ROOT, RESULTS_DIR, SEEDS, yolo_weights

OUT = os.path.join(RESULTS_DIR, "wa_check")
DEVICE = 0 if torch.cuda.is_available() else "cpu"


def load_ckpt(i):
    return torch.load(yolo_weights(i), map_location="cpu", weights_only=False)


def describe(ckpt, i):
    args = ckpt.get("train_args", {}) or {}
    m = ckpt.get("model")
    return {"seed": SEEDS[i], "file": yolo_weights(i),
            "pretrained_start": args.get("model"), "pretrained_flag": args.get("pretrained"),
            "train_seed": args.get("seed"), "epoch_saved": ckpt.get("epoch"),
            "ema_entry_present": ckpt.get("ema") is not None,
            "model_dtype": str(next(m.parameters()).dtype) if m is not None else None,
            "ultralytics_version": ckpt.get("version"), "date": ckpt.get("date")}


def merge(ckpts, coeffs):
    """Weighted average of floating-point parameters and buffers (as swa_seeds.py);
    integer buffers (num_batches_tracked) are taken from the first checkpoint."""
    base = copy.deepcopy(ckpts[0]["model"]).float()
    sds = [copy.deepcopy(c["model"]).float().state_dict() for c in ckpts]
    avg = {}
    for k in sds[0]:
        if sds[0][k].dtype.is_floating_point:
            avg[k] = sum(c * sd[k] for c, sd in zip(coeffs, sds))
        else:
            avg[k] = sds[0][k]
    base.load_state_dict(avg)
    return base


@torch.no_grad()
def recalibrate_bn(model, imgsz, n_images, batch, seed=0):
    """Re-estimate BatchNorm running statistics on training images."""
    model = model.to(DEVICE).float().eval()
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None          # cumulative average over all batches
        m.train()
    paths = sorted(glob.glob(os.path.join(OUTPUT_ROOT, "train", "images", "*.*")))
    rng = np.random.default_rng(seed)
    paths = [paths[i] for i in rng.permutation(len(paths))[:n_images]]
    for s in range(0, len(paths), batch):
        x = np.stack([np.asarray(Image.open(p).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR))
                      for p in paths[s:s + batch]])
        x = torch.from_numpy(x).permute(0, 3, 1, 2).float().div(255).to(DEVICE)
        model(x)
    model.eval()
    return model.cpu()


def save_tmp(model, template, path):
    ck = dict(template)
    ck["model"] = copy.deepcopy(model).half()
    ck["ema"] = None                # make sure Ultralytics loads THIS model
    ck["epoch"] = -1
    torch.save(ck, path)


def evaluate(path, split, imgsz):
    m = YOLO(path).val(data=DATA_YAML, split=split, imgsz=imgsz, device=DEVICE, plots=False, workers=0,
                       verbose=False, project=OUT, name="val_runs", exist_ok=True)
    return float(m.box.map50), float(m.box.map)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--bn-images", type=int, default=512, help="training images for BN re-estimation")
    ap.add_argument("--batch", type=int, default=16)
    args, _ = ap.parse_known_args()
    os.makedirs(OUT, exist_ok=True)
    tmp = os.path.join(OUT, "tmp_merged.pt")

    ckpts = [load_ckpt(i) for i in range(len(SEEDS))]
    info = [describe(c, i) for i, c in enumerate(ckpts)]
    with open(os.path.join(OUT, "checkpoints.json"), "w") as f:
        json.dump(info, f, indent=2, default=str)
    print(json.dumps(info, indent=2, default=str))

    # weight-level sanity check: an "average" with coefficients [1, 0] must equal seed 0
    ref = copy.deepcopy(ckpts[0]["model"]).float().state_dict()
    test = merge(ckpts[:2], [1.0, 0.0]).state_dict()
    max_diff = max((ref[k].float() - test[k].float()).abs().max().item() for k in ref)
    print(f"Weight check: max |merge([1,0]) - seed {SEEDS[0]}| = {max_diff:.3e} (must be 0)")
    with open(os.path.join(OUT, "weight_check.txt"), "w") as f:
        f.write(f"max abs difference merge([1,0]) vs seed {SEEDS[0]}: {max_diff:.3e}\n")

    rows = []

    def record(exp, alpha, bn, split, path):
        m50, m = evaluate(path, split, args.imgsz)
        rows.append({"experiment": exp, "alpha": alpha, "bn_recalibrated": bn, "split": split,
                     "mAP50": round(m50 * 100, 2), "mAP50-95": round(m * 100, 2)})
        print(f"{exp:28s} alpha={alpha!s:5s} BN-recal={bn!s:5s} {split:5s} "
              f"mAP50={m50*100:6.2f} mAP50-95={m*100:6.2f}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "wa_check.csv"), index=False)

    # 1. saved averaged model
    swa_path = os.path.join(BASELINE_DIR, "yolov8_swa", "weights", "best.pt")
    if os.path.exists(swa_path):
        record("saved swa best.pt", "-", False, "val", swa_path)

    # 2. reference: the original checkpoints
    for i in range(len(SEEDS)):
        record(f"original seed {SEEDS[i]}", "-", False, "val", yolo_weights(i))

    # 2-3. linear path seed 0 -> seed 1 (alpha 0 and 1 are the endpoint check)
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        merged = merge(ckpts[:2], [1 - alpha, alpha])
        save_tmp(merged, ckpts[0], tmp)
        record(f"path {SEEDS[0]}->{SEEDS[1]}", alpha, False, "val", tmp)
        save_tmp(recalibrate_bn(merged, args.imgsz, args.bn_images, args.batch), ckpts[0], tmp)
        record(f"path {SEEDS[0]}->{SEEDS[1]}", alpha, True, "val", tmp)

    # 4. uniform average of all seeds
    k = len(SEEDS)
    merged = merge(ckpts, [1 / k] * k)
    for bn in (False, True):
        model = recalibrate_bn(merged, args.imgsz, args.bn_images, args.batch) if bn else merged
        save_tmp(model, ckpts[0], tmp)
        for split in ("val", "test"):
            record(f"uniform average of {k}", "1/3", bn, split, tmp)

    os.remove(tmp)
    print("\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
