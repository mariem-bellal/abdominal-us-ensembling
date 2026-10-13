# Abdomen MRI — Object Detection — Ensemble YOLOv8 + RT-DETR

## Quick Start

**All configuration is in `config.py`** — edit paths and hyperparameters there once.

### Step 1: Verify GPU
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
python check_gpu.py
```

### Step 2: Inspect your dataset
```bash
python inspect_data.py
```
Check that CSV column names match `COLUMN_MAPPING` in `config.py`.

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Convert dataset to YOLO format
```bash
python prepare_dataset.py
```

### Step 5: Visualize dataset samples (for the article)
```bash
python visualize_dataset_samples.py
```

### Step 6: Train YOLOv8 baseline
```bash
python train_yolov8_baseline.py              # single seed
python train_yolov8_baseline.py --all-seeds  # 3 seeds (for confidence intervals)
```

### Step 7: Train RT-DETR
```bash
python train_rtdetr.py              # single seed
python train_rtdetr.py --all-seeds  # 3 seeds
```

### Step 8: Tune WBF hyperparameters on validation set
```bash
python tune_wbf.py
```
Update `IOU_WBF` and `WBF_WEIGHTS` in `config.py` if the best config differs.

### Step 9: Ensemble evaluation
```bash
python ensemble_eval.py                       # WBF + TTA (default)
python ensemble_eval.py --no-tta              # WBF without TTA
python ensemble_eval.py --all-seeds           # multi-seed
python ensemble_eval.py --no-tta --all-seeds  # multi-seed without TTA
```

### Step 10: Compute F1 for individual models
```bash
python compute_baseline_f1.py              # single seed
python compute_baseline_f1.py --all-seeds  # 3 seeds (mean ± std, recommended)
```

### Step 11: Generate all figures
```bash
python plot_training_curves.py
python plot_detection_overlap.py
python visualize_detections.py
```

### Step 12: Build ablation table
```bash
python build_ablation_table.py
```
Outputs `figures/ablation_table.md` and `figures/ablation_table.tex`.

### Optional: homogeneous 3-seed ensemble (WBF) and weight averaging (SWA)

Once Step 6 has been run with `--all-seeds`, two extra experiments are
available — fusing the 3 already-trained YOLOv8 seeds instead of (or in
addition to) YOLOv8+RT-DETR:

```bash
python ensemble_seeds_wbf.py            # WBF fusion of the 3 seeds, no TTA
python ensemble_seeds_wbf.py --tta      # same, with TTA
python swa_seeds.py                     # weight-averaging (SWA) of the 3 seeds
```

See `ensemble_seeds_and_swa.ipynb` for a ready-to-run notebook version, and
`verification_report.md` for the full comparison against the baseline and
the YOLOv8+RT-DETR ensemble (spoiler: the 3-seed WBF+TTA ensemble is the
best configuration found so far on every metric; SWA does not work here —
see the report for why).

## Project Structure

```
config.py                     <- ALL paths and hyperparameters (edit this)
check_gpu.py                  <- Step 1: GPU check
inspect_data.py                <- Step 2: dataset inspection
prepare_dataset.py             <- Step 4: CSV -> YOLO format conversion
visualize_dataset_samples.py   <- Step 5: dataset sample figures
train_yolov8_baseline.py       <- Step 6: YOLOv8 training + model info + timing
train_rtdetr.py                <- Step 7: RT-DETR training + model info + timing
tune_wbf.py                    <- Step 8: WBF hyperparameter tuning
ensemble_eval.py                <- Step 9: ensemble evaluation + timing
compute_baseline_f1.py          <- Step 10: F1 at optimal threshold for baselines (now --all-seeds capable)
plot_training_curves.py         <- Step 11a: mAP + loss curves
plot_detection_overlap.py       <- Step 11b: detection complementarity
visualize_detections.py         <- Step 11c: qualitative examples
build_ablation_table.py         <- Step 12: ablation table (Markdown + LaTeX)
ensemble_seeds_wbf.py           <- NEW: homogeneous 3-seed WBF ensemble (val-tuned threshold)
swa_seeds.py                    <- NEW: weight-averaging (SWA) of the 3 seeds
ensemble_seeds_and_swa.ipynb    <- NEW: notebook wrapping the two scripts above
```

## What Changed vs. Original

| Issue | Fix |
|---|---|
| Hardcoded paths in every file | Central `config.py` |
| No random seed set | `set_seed()` called in every script, passed to Ultralytics |
| Single seed, no confidence intervals | `--all-seeds` flag, mean ± std aggregation |
| No inference time measurement | `measure_inference_time()` in training + ensemble scripts |
| No model size / parameter count | `log_model_info()` saved as `model_info.json` |
| F1 only for ensemble rows, and computed differently from baselines | `compute_baseline_f1.py` now supports `--all-seeds`, and `build_ablation_table.py` uses its output for every row (same methodology everywhere) |
| WBF hyperparameters unjustified, and not actually applied after tuning | `tune_wbf.py` grid search on validation set; gap vs config.py documented in `verification_report.md` |
| No dataset sample visualization | `visualize_dataset_samples.py` |
| Loss curves not plotted | `plot_training_curves.py` now plots both mAP and loss |
| No figures included in archive | All figures saved to `figures/` directory |
| `check_gpu.py` crashed on `total_mem` | Fixed to `total_memory` |
| argparse scripts crashed when run from a Jupyter cell | All scripts use `parse_known_args()` instead of `parse_args()` |
| Only heterogeneous (YOLOv8+RT-DETR) ensembling tried | Added homogeneous 3-seed WBF ensemble + SWA experiment (`ensemble_seeds_wbf.py`, `swa_seeds.py`) |

## Limitations to Mention in Article

- No patient-level ID in metadata → potential data leakage between splits
- Mono-class dataset — results may not generalize to multi-organ detection
- RT-DETR does not support Ultralytics native TTA (asymmetry in ensemble)
- The detected class name from RF100 is generic — confirm anatomical meaning
- The confidence threshold used for Precision/Recall/F1 is optimized directly
  on the test set for both single-model baselines (via Ultralytics' internal
  metric, which selects the threshold maximizing F1 on the evaluated split)
  and the original heterogeneous ensemble (`ensemble_eval.py`'s threshold
  sweep). This may overestimate real-world performance, since in deployment
  the optimal threshold cannot be known without ground-truth labels. A
  validation-set-tuned threshold, as used in `ensemble_seeds_wbf.py`, is the
  more rigorous approach and should be preferred when reporting final numbers.
