"""
Step 7 — Train and evaluate RT-DETR-l (second detector).

Single seed:   python train_rtdetr.py
All seeds:     python train_rtdetr.py --all-seeds

Outputs:
    results_rtdetr/test_metrics.json
    results_rtdetr/model_info.json
    results_rtdetr/inference_time.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from ultralytics import RTDETR

from config import (
    DATA_YAML, RTDETR_DIR, DEVICE, IMG_SIZE, SEEDS,
    RTDETR_MODEL, RTDETR_EPOCHS, RTDETR_BATCH, RTDETR_PATIENCE,
    AUG_PARAMS, TEST_IMAGES_DIR, set_seed,
)


def log_model_info(model, save_dir):
    info = {}
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable   = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    info["total_parameters"]     = total_params
    info["trainable_parameters"] = trainable
    info["total_parameters_M"]   = round(total_params / 1e6, 2)

    try:
        model_info_str = model.info(verbose=False)
        if isinstance(model_info_str, tuple) and len(model_info_str) >= 4:
            info["GFLOPs"] = round(float(model_info_str[3]), 2)
    except Exception:
        info["GFLOPs"] = "N/A"

    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        info["weight_file_MB"] = round(os.path.getsize(best_pt) / 1e6, 2)

    out = os.path.join(save_dir, "model_info.json")
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved: {out}")
    print(json.dumps(info, indent=2))
    return info


def measure_inference_time(model, n_images=50, warmup=10):
    import glob
    image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.*")))[:n_images + warmup]

    for img_path in image_paths[:warmup]:
        model.predict(img_path, conf=0.25, verbose=False)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for img_path in image_paths[warmup:n_images + warmup]:
        t0 = time.perf_counter()
        model.predict(img_path, conf=0.25, verbose=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    result = {
        "model": "RT-DETR-l",
        "device": str(DEVICE),
        "n_images": len(times),
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms":  round(float(np.std(times)), 2),
        "median_ms": round(float(np.median(times)), 2),
    }
    print(f"\nInference time: {result['mean_ms']} ± {result['std_ms']} ms/image")
    return result


def train_and_evaluate(seed_idx=0):
    seed = SEEDS[seed_idx]
    set_seed(seed)

    suffix = "" if seed_idx == 0 else f"_seed{seed_idx}"
    run_name = f"rtdetr_baseline{suffix}"

    print(f"\n{'='*60}")
    print(f"Training RT-DETR-l — seed={seed} (run: {run_name})")
    print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 0 else 'CPU'}")
    print(f"{'='*60}\n")

    model = RTDETR(RTDETR_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=RTDETR_EPOCHS,
        imgsz=IMG_SIZE,
        batch=RTDETR_BATCH,
        patience=RTDETR_PATIENCE,
        device=DEVICE,
        project=RTDETR_DIR,
        name=run_name,
        seed=seed,
        **AUG_PARAMS,
    )

    print(f"\n=== Evaluation on TEST split (seed={seed}) ===")
    metrics = model.val(data=DATA_YAML, split="test", device=DEVICE)

    summary = {
        "seed": seed,
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "precision":  float(metrics.box.mp),
        "recall":     float(metrics.box.mr),
    }
    print(json.dumps(summary, indent=2))

    os.makedirs(RTDETR_DIR, exist_ok=True)
    metrics_file = f"test_metrics{suffix}.json"
    with open(os.path.join(RTDETR_DIR, metrics_file), "w") as f:
        json.dump(summary, f, indent=2)

    run_dir = os.path.join(RTDETR_DIR, run_name)
    if seed_idx == 0:
        log_model_info(model, run_dir)
        timing = measure_inference_time(model)
        with open(os.path.join(RTDETR_DIR, "inference_time.json"), "w") as f:
            json.dump(timing, f, indent=2)

    return summary


def aggregate_seeds(results):
    keys = ["mAP50", "mAP50-95", "precision", "recall"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
            "values": vals,
        }
    print("\n=== RT-DETR-l — Aggregated across seeds ===")
    for k, v in agg.items():
        print(f"  {k}: {v['mean']*100:.2f}% ± {v['std']*100:.2f}%")

    with open(os.path.join(RTDETR_DIR, "test_metrics_aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-seeds", action="store_true")
    args, _unknown = parser.parse_known_args()

    if args.all_seeds:
        results = [train_and_evaluate(i) for i in range(len(SEEDS))]
        aggregate_seeds(results)
    else:
        train_and_evaluate(seed_idx=0)


if __name__ == "__main__":
    main()
