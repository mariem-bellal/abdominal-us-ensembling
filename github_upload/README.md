# Homogeneous vs. Heterogeneous Ensembling for Object Detection in Abdominal Ultrasound

Code for the paper *"Homogeneous vs. Heterogeneous Ensembling for Object Detection in Abdominal Ultrasound: YOLOv8 and RT-DETR with Weighted Boxes Fusion and Test-Time Augmentation"* (M. Bellal, S. El Fkihi, I2COMSAPP'26, Springer LNNS).

The paper compares two ways of combining bounding-box detectors with Weighted Boxes Fusion (WBF):

- **Heterogeneous ensemble:** YOLOv8s + RT-DETR-l
- **Homogeneous ensemble:** three YOLOv8s models trained with different random seeds (42, 123, 456)

Both are evaluated with and without test-time augmentation (TTA), at equal ensemble size, with a confidence threshold tuned on the validation set and a paired bootstrap over test images.

## Dataset

The data are **not** included in this repository. The dataset is public: Roboflow 100 "Abdomen MRI", originally published on Roboflow Universe as "circle-3train" (CC BY 4.0):
<https://universe.roboflow.com/roboflow-100/abdomen-mri>

Despite its name, it contains pediatric abdominal **ultrasound** images (2,604 images; train / valid / test = 1,887 / 479 / 238), with a single class marking a focal bowel structure. We use the splits of the original release.

## Setup

```bash
pip install -r requirements.txt
```

All paths and hyperparameters are set in `config.py`. Place the downloaded dataset folder next to the project folder, or edit `DATASET_ROOT` in `config.py`.

## How to reproduce

| Step | Script | What it does |
|---|---|---|
| 1 | `prepare_dataset.py` | Converts the CSV annotations to YOLO format (`abdomen_mri_yolo/`) |
| 2 | `count_yolo.py` | Checks image, label and box counts per split |
| 3 | `check_leakage.py` | Looks for exact and near-duplicate images across splits |
| 4 | `train_yolov8_baseline.py --all-seeds` | Trains YOLOv8s with seeds 42, 123, 456 |
| 5 | `train_rtdetr.py --all-seeds` | Trains RT-DETR-l with seeds 42, 123, 456 |
| 6 | `eval_unified.py` | Evaluates every configuration with one protocol: single models, both ensembles, TTA, equal-size ensembles, NMS / Soft-NMS, validation-tuned thresholds, paired bootstrap, per-pair disagreement counts |
| 7 | `baselines_nonlearned.py` | Non-learned baselines (Otsu thresholding, fixed mean box) |
| 8 | `wa_check.py` | Weight averaging of the seeds: linear interpolation and BatchNorm re-estimation |
| 9 | `saturation_all.py` | Checks whether detections fall on colour Doppler overlays (all test images) |

Steps 4, 5, 6 (prediction stage) and 8 need a GPU. Results are written to `results/`.

## Citation

If you use this code, please cite the paper (full reference to be added after publication).

## Contact

Mariem Bellal, ENSIAS, Mohammed V University in Rabat — mariemebellal@gmail.com
