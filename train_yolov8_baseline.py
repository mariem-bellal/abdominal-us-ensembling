"""
Step 5 — Train and evaluate YOLOv8s baseline.

Single seed:   python train_yolov8_baseline.py
All seeds:     python train_yolov8_baseline.py --all-seeds

Outputs:
    results_baseline/test_metrics.json          (single seed)
    results_baseline/test_metrics_seedN.json    (multi-seed)
    results_baseline/model_info.json            (params, FLOPs, size)
    results_baseline/inference_time.json        (ms per image)
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from ultralytics import YOLO

from config import (
    DATA_YAML, BASELINE_DIR, DEVICE, IMG_SIZE, SEEDS,
    YOLO_MODEL, YOLO_EPOCHS, YOLO_BATCH, YOLO_PATIENCE,
    AUG_PARAMS, TEST_IMAGES_DIR, set_seed,
)


# ------------------------------------------------------------------
# NEW: Model info — parameter count, FLOPs, file size
# ------------------------------------------------------------------
def log_model_info(model, save_dir):
    """Log model parameter count, estimated FLOPs, and weight file size."""
    info = {}

    # parameter count
    total_params = sum(p.numel() for p in model.model.parameters())
    trainable   = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    info["total_parameters"]     = total_params
    info["trainable_parameters"] = trainable
    info["total_parameters_M"]   = round(total_params / 1e6, 2)

    # FLOPs (from Ultralytics model.info)
    try:
        model_info_str = model.info(verbose=False)
        # Ultralytics .info() returns (layers, params, gradients, flops)
        if isinstance(model_info_str, tuple) and len(model_info_str) >= 4:
            info["GFLOPs"] = round(float(model_info_str[3]), 2)
    except Exception:
        info["GFLOPs"] = "N/A"

    # weight file size
    best_pt = os.path.join(save_dir, "weights", "best.pt")
    if os.path.exists(best_pt):
        info["weight_file_MB"] = round(os.path.getsize(best_pt) / 1e6, 2)

    out = os.path.join(save_dir, "model_info.json")
    with open(out, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Model info saved: {out}")
    print(json.dumps(info, indent=2))
    return info


# ------------------------------------------------------------------
# NEW: Inference time measurement
# ------------------------------------------------------------------
def measure_inference_time(model, n_images=50, warmup=10):
    """Measure average inference time per image (ms) on the test set."""
    import glob
    image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES_DIR, "*.*")))[:n_images + warmup]

    # warmup (GPU kernels, memory allocation)
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
        times.append((time.perf_counter() - t0) * 1000)  # ms

    result = {
        "model": "YOLOv8s",
        "device": str(DEVICE),
        "n_images": len(times),
        "mean_ms": round(float(np.mean(times)), 2),
        "std_ms":  round(float(np.std(times)), 2),
        "median_ms": round(float(np.median(times)), 2),
    }
    print(f"\nInference time: {result['mean_ms']} ± {result['std_ms']} ms/image")
    return result


# ------------------------------------------------------------------
# Train + evaluate
# ------------------------------------------------------------------
def train_and_evaluate(seed_idx=0):
    seed = SEEDS[seed_idx]
    set_seed(seed)

    suffix = "" if seed_idx == 0 else f"_seed{seed_idx}"
    run_name = f"yolov8_baseline{suffix}"

    print(f"\n{'='*60}")
    print(f"Training YOLOv8s — seed={seed} (run: {run_name})")
    print(f"Device: {'GPU (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 0 else 'CPU'}")
    print(f"{'='*60}\n")

    model = YOLO(YOLO_MODEL)

    model.train(
        data=DATA_YAML,
        epochs=YOLO_EPOCHS,
        imgsz=IMG_SIZE,
        batch=YOLO_BATCH,
        patience=YOLO_PATIENCE,
        device=DEVICE,
        project=BASELINE_DIR,
        name=run_name,
        seed=seed,           # <-- NEW: pass seed to Ultralytics
        **AUG_PARAMS,
        mosaic=0.5,
    )

    # ---- Evaluate on test split ----
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

    os.makedirs(BASELINE_DIR, exist_ok=True)
    metrics_file = f"test_metrics{suffix}.json"
    with open(os.path.join(BASELINE_DIR, metrics_file), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Model info + inference time (first seed only) ----
    run_dir = os.path.join(BASELINE_DIR, run_name)
    if seed_idx == 0:
        log_model_info(model, run_dir)
        timing = measure_inference_time(model)
        with open(os.path.join(BASELINE_DIR, "inference_time.json"), "w") as f:
            json.dump(timing, f, indent=2)

    return summary


# ------------------------------------------------------------------
# NEW: Multi-seed aggregation
# ------------------------------------------------------------------
def aggregate_seeds(results):
    """Compute mean ± std across seeds for the article."""
    keys = ["mAP50", "mAP50-95", "precision", "recall"]
    agg = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[k] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)), 4),
            "values": vals,
        }
    print("\n=== YOLOv8s — Aggregated across seeds ===")
    for k, v in agg.items():
        print(f"  {k}: {v['mean']*100:.2f}% ± {v['std']*100:.2f}%")

    with open(os.path.join(BASELINE_DIR, "test_metrics_aggregated.json"), "w") as f:
        json.dump(agg, f, indent=2)
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-seeds", action="store_true",
                        help="Train with all seeds defined in config.SEEDS")
    args, _unknown = parser.parse_known_args()

    if args.all_seeds:
        results = []
        for i in range(len(SEEDS)):
            results.append(train_and_evaluate(seed_idx=i))
        aggregate_seeds(results)
    else:
        train_and_evaluate(seed_idx=0)


if __name__ == "__main__":
    main()
