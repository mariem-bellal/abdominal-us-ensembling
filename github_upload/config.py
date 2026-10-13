"""
====================================================================
config.py — Single source of truth for ALL project paths, hyper-
parameters, and experimental settings.

Paths are auto-detected from the current working directory.
Works in Jupyter notebooks (no __file__ needed).
====================================================================
"""

import os
import random
import torch
import numpy as np
from pathlib import Path

# ------------------------------------------------------------------
# 1.  AUTO-DETECTED PATHS
# ------------------------------------------------------------------
# PROJECT_DIR = current working directory (works in notebooks + scripts)
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    # __file__ not defined (Jupyter notebook) -> use cwd
    PROJECT_DIR = Path(os.getcwd()).resolve()

DATASET_FOLDER_NAME = "archive (1)"

def _find_dataset():
    """Auto-locate the dataset folder."""
    # 1) inside project dir
    candidate = PROJECT_DIR / DATASET_FOLDER_NAME
    if candidate.is_dir():
        return str(candidate)
    # 2) sibling of project dir
    candidate = PROJECT_DIR.parent / DATASET_FOLDER_NAME
    if candidate.is_dir():
        return str(candidate)
    # 3) common locations
    for parent in [Path.home() / "Downloads", Path.home() / "Desktop", Path.home()]:
        candidate = parent / DATASET_FOLDER_NAME
        if candidate.is_dir():
            return str(candidate)
    # 4) fallback
    print(f"[WARNING] Dataset folder '{DATASET_FOLDER_NAME}' not found automatically.")
    print(f"  Searched in: {PROJECT_DIR}, {PROJECT_DIR.parent}, ~/Downloads, ~/Desktop")
    print(f"  -> Place '{DATASET_FOLDER_NAME}' next to your project folder, or edit DATASET_ROOT in config.py")
    return str(PROJECT_DIR / DATASET_FOLDER_NAME)

DATASET_ROOT = _find_dataset()

OUTPUT_ROOT  = str(PROJECT_DIR / "abdomen_mri_yolo")
RESULTS_DIR  = str(PROJECT_DIR / "results")
FIGURES_DIR  = str(PROJECT_DIR / "figures")

DATA_YAML        = os.path.join(OUTPUT_ROOT, "data.yaml")
BASELINE_DIR     = os.path.join(RESULTS_DIR, "results_baseline")
RTDETR_DIR       = os.path.join(RESULTS_DIR, "results_rtdetr")
ENSEMBLE_DIR     = os.path.join(RESULTS_DIR, "results_ensemble")
TEST_IMAGES_DIR  = os.path.join(OUTPUT_ROOT, "test", "images")
TEST_LABELS_DIR  = os.path.join(OUTPUT_ROOT, "test", "labels")

def yolo_weights(seed_idx=0):
    suffix = "" if seed_idx == 0 else f"_seed{seed_idx}"
    return os.path.join(BASELINE_DIR, f"yolov8_baseline{suffix}", "weights", "best.pt")

def rtdetr_weights(seed_idx=0):
    suffix = "" if seed_idx == 0 else f"_seed{seed_idx}"
    return os.path.join(RTDETR_DIR, f"rtdetr_baseline{suffix}", "weights", "best.pt")

# ------------------------------------------------------------------
# 2.  TRAINING HYPER-PARAMETERS
# ------------------------------------------------------------------
YOLO_MODEL    = "yolov8s.pt"
YOLO_EPOCHS   = 100
YOLO_BATCH    = 16
YOLO_PATIENCE = 20

RTDETR_MODEL   = "rtdetr-l.pt"
RTDETR_EPOCHS  = 100
RTDETR_BATCH   = 8
RTDETR_PATIENCE = 20

IMG_SIZE = 640
DEVICE   = 0 if torch.cuda.is_available() else "cpu"

AUG_PARAMS = dict(
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.2,
    degrees=5.0,
    translate=0.05,
    scale=0.2,
    fliplr=0.5,
    flipud=0.0,
)

# ------------------------------------------------------------------
# 3.  ENSEMBLE HYPER-PARAMETERS
# ------------------------------------------------------------------
CONF_THRES   = 0.25
IOU_WBF      = 0.55
IOU_MATCH    = 0.5
NUM_CLASSES  = 1
USE_TTA      = True
WBF_WEIGHTS  = [1.0, 1.15]

# ------------------------------------------------------------------
# 4.  MULTI-SEED EXPERIMENT
# ------------------------------------------------------------------
SEEDS = [42, 123, 456]

# ------------------------------------------------------------------
# 5.  REPRODUCIBILITY
# ------------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

# ------------------------------------------------------------------
# 6.  CSV COLUMN MAPPING
# ------------------------------------------------------------------
CSV_CANDIDATES = ["_annotations.csv", "annotations.csv", "labels.csv"]
COLUMN_MAPPING = {
    "filename": "filename",
    "width":    "width",
    "height":   "height",
    "class":    "class",
    "xmin":     "xmin",
    "ymin":     "ymin",
    "xmax":     "xmax",
    "ymax":     "ymax",
}
SPLITS = ["train", "valid", "test"]

# ------------------------------------------------------------------
# 7.  PRINT DETECTED PATHS (confirmation)
# ------------------------------------------------------------------
def print_paths():
    print(f"PROJECT_DIR  : {PROJECT_DIR}")
    print(f"DATASET_ROOT : {DATASET_ROOT}  {'[OK]' if os.path.isdir(DATASET_ROOT) else '[NOT FOUND]'}")
    print(f"OUTPUT_ROOT  : {OUTPUT_ROOT}")
    print(f"RESULTS_DIR  : {RESULTS_DIR}")
    print(f"FIGURES_DIR  : {FIGURES_DIR}")
    print(f"DEVICE       : {'GPU (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 0 else 'CPU'}")
