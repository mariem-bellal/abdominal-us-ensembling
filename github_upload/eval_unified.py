#!/usr/bin/env python3
"""
eval_unified.py - One evaluation protocol for every configuration, no retraining.

Reviewer points covered (no new training needed):
  R2  equal-size ensembles, ensemble uncertainty, significance
  R3.W1a single YOLOv8s + TTA          R3.W1b equal-size ensembles
  R3.W1c flip-TTA for RT-DETR-l        R3.W2  complementarity per seed pair
  R3.W3b paired bootstrap              R3.W5  same WBF selection rule (val optimum)
  R4.2   uncertainty                   R4.3   symmetric TTA     R4.7 NMS / Soft-NMS
and it re-derives P/R/F1 of every Table 1 row with the threshold tuned on VALIDATION.

Protocol (identical for every configuration)
  * predictions of the 6 trained checkpoints with conf >= CONF_THRES (config.py)
  * YOLOv8s "+TTA" = Ultralytics native TTA (augment=True);
    RT-DETR-l "+TTA" = horizontal-flip TTA done here (flip input, map boxes back,
    WBF of original + flipped predictions)
  * ensembles: WBF IoU 0.55; weights [1.0, 1.15] for the published heterogeneous
    setting, equal weights elsewhere; NMS / Soft-NMS variants with the same IoU
  * "val-tuned WBF": IoU in {0.30..0.70} (and RT-DETR weight in {1.0..1.3} for the
    heterogeneous ensemble) chosen by mAP@50 on VALIDATION, same rule for both
  * confidence threshold maximising F1 on VALIDATION (0.10..0.90 step 0.05),
    applied unchanged on TEST; P/R/F1 with greedy matching at IoU >= 0.5
  * mAP@50 / mAP@50-95 with mean_average_precision (as in the published scripts)
  * paired bootstrap over the test images (default 10,000 resamples)

Usage (project folder, environment with ultralytics + ensemble_boxes + mean_average_precision):
    python eval_unified.py                    # predict (GPU) + evaluate
    python eval_unified.py --stage predict    # only cache predictions
    python eval_unified.py --stage evaluate   # only evaluate from the cache (CPU is fine)

Outputs in results/unified_eval/:
    preds_*.pkl          cached predictions (reused by later runs)
    instances.csv        one row per configuration instance (seed / seed pair)
    summary.csv/.md      mean +- std per configuration
    bootstrap.csv/.md    paired differences with 95% CIs
    complementarity.csv  per-pair box-level agreement (Section 5.1 analysis)
    wbf_selection.csv    val-selected WBF settings
    ap_check.csv         own AP@50 vs library (sanity check for the bootstrap)
"""

import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from config import (CONF_THRES, IOU_MATCH, NUM_CLASSES, OUTPUT_ROOT, RESULTS_DIR,
                    SEEDS, rtdetr_weights, yolo_weights)

OUT = os.path.join(RESULTS_DIR, "unified_eval")
SPLITS = {"valid": os.path.join(OUTPUT_ROOT, "valid"),
          "test": os.path.join(OUTPUT_ROOT, "test")}
THRESHOLDS = np.round(np.arange(0.10, 0.95, 0.05), 2)
IOU_WBF = 0.55
HET_WEIGHTS = [1.0, 1.15]
GRID_IOU = [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]
GRID_W = [1.0, 1.1, 1.15, 1.2, 1.3]
N = len(SEEDS)
MODELS = {f"Y{i}": ("yolo", i) for i in range(N)}
MODELS.update({f"R{i}": ("rtdetr", i) for i in range(N)})


# ------------------------------------------------------------------
# Stage 1: cache predictions
# ------------------------------------------------------------------
def cache_path(key, split, tta):
    return os.path.join(OUT, f"preds_{key}_{split}_{'tta' if tta else 'notta'}.pkl")


def _to_dict(result, iw, ih):
    b = result.boxes
    if len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    xyxy = b.xyxy.cpu().numpy()
    return (np.clip(xyxy / np.array([iw, ih, iw, ih]), 0, 1),
            b.conf.cpu().numpy(), b.cls.cpu().numpy().astype(int))


def predict_all(force=False):
    import torch
    from ensemble_boxes import weighted_boxes_fusion
    from ultralytics import RTDETR, YOLO
    device = 0 if torch.cuda.is_available() else "cpu"
    for key, (kind, idx) in MODELS.items():
        model = None
        for split, root in SPLITS.items():
            for tta in (False, True):
                path = cache_path(key, split, tta)
                if os.path.exists(path) and not force:
                    print(f"  cached: {os.path.basename(path)}")
                    continue
                if model is None:
                    w = yolo_weights(idx) if kind == "yolo" else rtdetr_weights(idx)
                    model = YOLO(w) if kind == "yolo" else RTDETR(w)
                out = {}
                for p in sorted(glob.glob(os.path.join(root, "images", "*.*"))):
                    img = Image.open(p).convert("RGB")
                    iw, ih = img.size
                    kw = {"conf": CONF_THRES, "verbose": False, "device": device}
                    if kind == "yolo":
                        r = model.predict(p, augment=tta, **kw)[0]
                        b, s, l = _to_dict(r, iw, ih)
                    else:
                        b, s, l = _to_dict(model.predict(p, **kw)[0], iw, ih)
                        if tta:   # horizontal-flip TTA for RT-DETR-l
                            flipped = np.ascontiguousarray(np.array(ImageOps.mirror(img))[:, :, ::-1])
                            bf, sf, lf = _to_dict(model.predict(flipped, **kw)[0], iw, ih)
                            if len(bf):
                                bf = np.stack([1 - bf[:, 2], bf[:, 1], 1 - bf[:, 0], bf[:, 3]], 1)
                            if len(b) or len(bf):
                                b, s, l = weighted_boxes_fusion([b.tolist(), bf.tolist()],
                                                                [s.tolist(), sf.tolist()],
                                                                [l.tolist(), lf.tolist()],
                                                                iou_thr=IOU_WBF, skip_box_thr=0.0)
                    out[os.path.splitext(os.path.basename(p))[0]] = {
                        "size": (iw, ih), "boxes": np.asarray(b).reshape(-1, 4).tolist(),
                        "scores": np.asarray(s).tolist(), "labels": np.asarray(l).astype(int).tolist()}
                with open(path, "wb") as f:
                    pickle.dump(out, f)
                print(f"  saved: {os.path.basename(path)} ({len(out)} images)")


# ------------------------------------------------------------------
# Configurations: (name, instances, weights, method, iou)
#   weights: list, "equal", or "tune" (val-selected); method: wbf | nms | soft_nms
# ------------------------------------------------------------------
def build_configs():
    pairs = [(i, (i + 1) % N) for i in range(N)]            # (0,1) (1,2) (2,0)
    C = []
    for tta in (False, True):
        t = " + TTA" if tta else ""
        het = [[(f"Y{i}", tta), (f"R{i}", False)] for i in range(N)]
        homo3 = [[(f"Y{i}", tta) for i in range(N)]]
        C += [
            ("YOLOv8s (single)" + t, [[(f"Y{i}", tta)] for i in range(N)], None, "wbf", IOU_WBF),
            ("RT-DETR-l (single)" + (" + flip-TTA" if tta else ""),
             [[(f"R{i}", tta)] for i in range(N)], None, "wbf", IOU_WBF),
            ("Hetero, published WBF" + t, het, HET_WEIGHTS, "wbf", IOU_WBF),
            ("Hetero, equal-weight WBF" + t, het, "equal", "wbf", IOU_WBF),
            ("Hetero, val-tuned WBF" + t, het, "tune", "wbf", None),
            ("Hetero, NMS" + t, het, "equal", "nms", IOU_WBF),
            ("Hetero, Soft-NMS" + t, het, "equal", "soft_nms", IOU_WBF),
            ("Homo 2x YOLOv8s" + t, [[(f"Y{a}", tta), (f"Y{b}", tta)] for a, b in pairs],
             "equal", "wbf", IOU_WBF),
            ("Homo 3x YOLOv8s" + t, homo3, "equal", "wbf", IOU_WBF),
            ("Homo 3x YOLOv8s, val-tuned WBF" + t, homo3, "tune", "wbf", None),
            ("Homo 3x YOLOv8s, NMS" + t, homo3, "equal", "nms", IOU_WBF),
            ("Homo 3x YOLOv8s, Soft-NMS" + t, homo3, "equal", "soft_nms", IOU_WBF),
            ("2x YOLOv8s + RT-DETR-l" + t,
             [[(f"Y{a}", tta), (f"Y{b}", tta), (f"R{(b + 1) % N}", False)] for a, b in pairs],
             "equal", "wbf", IOU_WBF),
        ]
        if tta:   # symmetric TTA: RT-DETR-l with flip-TTA as well
            C.append(("Hetero, published WBF + TTA on both",
                      [[(f"Y{i}", True), (f"R{i}", True)] for i in range(N)], HET_WEIGHTS, "wbf", IOU_WBF))
    return C


def load_cache():
    cache = {}
    for key in MODELS:
        for split in SPLITS:
            for tta in (False, True):
                path = cache_path(key, split, tta)
                if not os.path.exists(path):
                    sys.exit(f"Missing {path}. Run with --stage predict first.")
                with open(path, "rb") as f:
                    cache[(key, split, tta)] = pickle.load(f)
    return cache


def load_gt(split, stem, iw, ih):
    path = os.path.join(SPLITS[split], "labels", stem + ".txt")
    rows = []
    if os.path.exists(path):
        for line in open(path):
            if line.strip():
                c, xc, yc, w, h = map(float, line.split())
                rows.append([(xc - w / 2) * iw, (yc - h / 2) * ih, (xc + w / 2) * iw, (yc + h / 2) * ih, int(c)])
    return np.array(rows) if rows else np.zeros((0, 5))


def combine(cache, members, weights, split, method="wbf", iou=IOU_WBF):
    """Per-image (N, 6) arrays [x1, y1, x2, y2, label, score] in pixels."""
    from ensemble_boxes import nms, soft_nms, weighted_boxes_fusion
    stems = sorted(cache[(members[0][0], split, members[0][1])].keys())
    w = None if weights is None else ([1.0] * len(members) if weights == "equal" else list(weights))
    out = []
    for s in stems:
        per = [cache[(k, split, t)][s] for k, t in members]
        iw, ih = per[0]["size"]
        if len(members) == 1:
            b, sc, lb = (np.array(per[0]["boxes"]).reshape(-1, 4), np.array(per[0]["scores"]),
                         np.array(per[0]["labels"]))
        elif not any(len(p["boxes"]) for p in per):
            out.append(np.zeros((0, 6)))
            continue
        else:
            args = ([np.asarray(p["boxes"], float).reshape(-1, 4) for p in per],
                    [np.asarray(p["scores"], float) for p in per],
                    [np.asarray(p["labels"], float) for p in per])
            if method == "wbf":
                b, sc, lb = weighted_boxes_fusion(*args, weights=w, iou_thr=iou, skip_box_thr=0.0)
            elif method == "nms":
                b, sc, lb = nms(*args, weights=w, iou_thr=iou)
            else:
                b, sc, lb = soft_nms(*args, weights=w, iou_thr=iou, sigma=0.5, thresh=0.001)
        if len(b) == 0:
            out.append(np.zeros((0, 6)))
            continue
        b = np.asarray(b) * np.array([iw, ih, iw, ih])
        out.append(np.column_stack([b, np.asarray(lb), np.asarray(sc)]))
    return stems, out


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------
def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-12)


def counts_at(preds, gt, thr):
    """TP, FP, FN for one image (greedy rule of ensemble_seeds_wbf.py)."""
    p = preds[preds[:, 5] >= thr] if len(preds) else preds
    p = p[np.argsort(-p[:, 5])] if len(p) else p
    ious = iou_matrix(p[:, :4], gt[:, :4]) if len(p) and len(gt) else np.zeros((len(p), len(gt)))
    matched, tp = set(), 0
    for i in range(len(p)):
        best, bj = 0.0, -1
        for j in range(len(gt)):
            if j not in matched and ious[i, j] > best:
                best, bj = ious[i, j], j
        if best >= IOU_MATCH:
            tp += 1
            matched.add(bj)
    return tp, len(p) - tp, len(gt) - len(matched)


def prf(c):
    tp, fp, fn = c
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def tune_threshold(preds, gts):
    best_t, best_f1 = THRESHOLDS[0], -1.0
    for t in THRESHOLDS:
        f1 = prf(np.sum([counts_at(p, g, t) for p, g in zip(preds, gts)], axis=0))[2]
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t)


def library_map(preds, gts, with_5095=True):
    from mean_average_precision import MetricBuilder
    m = MetricBuilder.build_evaluation_metric("map_2d", async_mode=False, num_classes=NUM_CLASSES)
    for p, g in zip(preds, gts):
        m.add(p if len(p) else np.zeros((0, 6)),
              np.hstack([g, np.zeros((len(g), 2))]) if len(g) else np.zeros((0, 7)))
    m50 = float(m.value(iou_thresholds=0.5)["mAP"])
    if not with_5095:
        return m50
    return m50, float(m.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), mpolicy="soft")["mAP"])


def ap_records(preds, gts):
    """Detections for AP@50, replicating mean_average_precision ('greedy'):
    same IoU formula (+1 pixel convention), same global confidence ordering
    (pandas sort), a detection is TP if its best-IoU GT has IoU > 0.5 and is
    not yet matched, otherwise FP."""
    from mean_average_precision.utils import compute_iou
    img, score, ious = [], [], []
    for k, (p, g) in enumerate(zip(preds, gts)):
        for i in range(len(p)):
            img.append(k); score.append(float(p[i, 5]))
            ious.append(compute_iou(p[i:i + 1, :4], g[:, :4])[0] if len(g) else np.zeros(0))
    ngt = np.array([len(g) for g in gts], float)
    if not img:
        return np.zeros(0, int), np.zeros(0), ngt
    order = pd.DataFrame({"confidence": score}).sort_values(by=["confidence"], ascending=False).index.values
    matched = {}
    tp = np.zeros(len(order))
    for d, idx in enumerate(order):
        iou = ious[idx]
        if len(iou):
            j = int(np.argsort(iou)[::-1][0])
            if iou[j] > 0.5 and j not in matched.setdefault(img[idx], set()):
                matched[img[idx]].add(j)
                tp[d] = 1.0
    return np.asarray(img, int)[order], tp, ngt


def ap_weighted(rec, w_img):
    """All-point interpolated AP@50, each image counted w_img times (bootstrap)."""
    img, tp, ngt = rec
    n = float(np.dot(w_img, ngt))
    if n == 0 or len(tp) == 0:
        return 0.0
    w = w_img[img]
    ctp, cfp = np.cumsum(w * tp), np.cumsum(w * (1 - tp))
    rec_, prec = ctp / n, ctp / np.maximum(ctp + cfp, 1e-12)
    mrec = np.concatenate([[0.0], rec_, [1.0]]); mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def detected(preds, gt, thr):
    """Boolean per GT box: matched by a prediction with score >= thr at IoU >= 0.5."""
    p = preds[preds[:, 5] >= thr] if len(preds) else preds
    if len(gt) == 0:
        return np.zeros(0, bool)
    if len(p) == 0:
        return np.zeros(len(gt), bool)
    return iou_matrix(gt[:, :4], p[:, :4]).max(axis=1) >= IOU_MATCH


# ------------------------------------------------------------------
# Stage 2: evaluation
# ------------------------------------------------------------------
def evaluate(n_boot, seed):
    cache = load_cache()
    gts = {}
    for split in SPLITS:
        ref = cache[("Y0", split, False)]
        gts[split] = [load_gt(split, s, *ref[s]["size"]) for s in sorted(ref)]

    rows, store, ap_check, wsel = [], {}, [], []
    for name, instances, weights, method, iou in build_configs():
        for k, members in enumerate(instances):
            label = "+".join(m + ("*" if t else "") for m, t in members)
            w_use, iou_use = weights, iou
            if weights == "tune":   # same selection rule for every ensemble: val mAP@50
                is_het = any(m.startswith("R") for m, _ in members)
                grid = [(i_, [1.0, w_]) for i_ in GRID_IOU for w_ in GRID_W] if is_het \
                    else [(i_, "equal") for i_ in GRID_IOU]
                scored = [(library_map(combine(cache, members, w_, "valid", "wbf", i_)[1],
                                       gts["valid"], False), i_, w_) for i_, w_ in grid]
                best = max(scored, key=lambda x: x[0])
                _, iou_use, w_use = best
                wsel.append({"config": name, "members": label, "iou": iou_use,
                             "weights": str(w_use), "val_mAP50": best[0]})
            _, pv = combine(cache, members, w_use, "valid", method, iou_use)
            thr = tune_threshold(pv, gts["valid"])
            _, pt = combine(cache, members, w_use, "test", method, iou_use)
            per_img = np.array([counts_at(p, g, thr) for p, g in zip(pt, gts["test"])])
            p, r, f1 = prf(per_img.sum(axis=0))
            m50, m5095 = library_map(pt, gts["test"])
            rec = ap_records(pt, gts["test"])
            ap_check.append({"config": name, "members": label, "library_mAP50": m50,
                             "own_AP50": ap_weighted(rec, np.ones(len(pt)))})
            tp, fp, fn = per_img.sum(axis=0)
            rows.append({"config": name, "instance": k, "members": label, "val_threshold": thr,
                         "wbf_iou": iou_use, "weights": str(w_use), "method": method,
                         "mAP50": m50, "mAP50-95": m5095, "precision": p, "recall": r, "f1": f1,
                         "TP": int(tp), "FP": int(fp), "FN": int(fn)})
            store[(name, k)] = (per_img, rec, label)
            print(f"{name:42s} [{label:14s}] thr={thr:.2f} mAP50={m50*100:6.2f} mAP50-95={m5095*100:6.2f} "
                  f"P={p*100:6.2f} R={r*100:6.2f} F1={f1*100:6.2f} TP/FP/FN={tp}/{fp}/{fn}", flush=True)

    inst = pd.DataFrame(rows)
    inst.to_csv(os.path.join(OUT, "instances.csv"), index=False)
    pd.DataFrame(wsel).to_csv(os.path.join(OUT, "wbf_selection.csv"), index=False)
    chk = pd.DataFrame(ap_check)
    chk["abs_diff"] = (chk.library_mAP50 - chk.own_AP50).abs()
    chk.to_csv(os.path.join(OUT, "ap_check.csv"), index=False)
    if chk.abs_diff.max() > 0.002:
        print(f"[WARNING] own AP@50 differs from the library by up to {chk.abs_diff.max():.4f} "
              "(bootstrap mAP@50 CIs use the own implementation).")

    met = ["mAP50", "mAP50-95", "precision", "recall", "f1"]
    summ = []
    for cfg, g in inst.groupby("config", sort=False):
        d = {"config": cfg, "n": len(g)}
        for m in met:
            mu, sd = g[m].mean() * 100, (g[m].std(ddof=1) * 100 if len(g) > 1 else 0.0)
            d[m] = f"{mu:.2f}" if len(g) == 1 else f"{mu:.2f} ± {sd:.2f}"
        d["FP (mean)"] = round(g.FP.mean(), 1)
        summ.append(d)
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    with open(os.path.join(OUT, "summary.md"), "w") as f:
        f.write("Mean ± sample std over instances (seeds or seed combinations); n = number of instances. "
                "Threshold tuned on validation for every row.\n\n" + summ.to_markdown(index=False))

    # ---- complementarity (Section 5.1 analysis, R3.W2) -------------------
    single_thr = {}
    for key in MODELS:
        tta_name = "YOLOv8s (single)" if key.startswith("Y") else "RT-DETR-l (single)"
        single_thr[key] = float(inst[(inst.config == tta_name) &
                                     (inst.members == key)].val_threshold.iloc[0])
    pairs = [("Y0", "Y1"), ("Y1", "Y2"), ("Y0", "Y2"), ("R0", "R1"), ("R1", "R2"), ("R0", "R2"),
             ("Y0", "R0"), ("Y1", "R1"), ("Y2", "R2")]
    comp = []
    for a, b in pairs:
        _, pa = combine(cache, [(a, False)], None, "test")
        _, pb = combine(cache, [(b, False)], None, "test")
        for setting, ta, tb in (("conf>=%.2f (cache floor)" % CONF_THRES, CONF_THRES, CONF_THRES),
                                ("val-tuned thresholds", single_thr[a], single_thr[b])):
            da = np.concatenate([detected(p, g, ta) for p, g in zip(pa, gts["test"])])
            db = np.concatenate([detected(p, g, tb) for p, g in zip(pb, gts["test"])])
            comp.append({"pair": f"{a}-{b}", "setting": setting, "thr_a": ta, "thr_b": tb,
                         "n_gt": len(da), "both": int((da & db).sum()), "a_only": int((da & ~db).sum()),
                         "b_only": int((~da & db).sum()), "neither": int((~da & ~db).sum()),
                         "disagreement": int((da ^ db).sum())})
    pd.DataFrame(comp).to_csv(os.path.join(OUT, "complementarity.csv"), index=False)

    # ---- paired bootstrap --------------------------------------------------
    rng = np.random.default_rng(seed)
    n_img = len(gts["test"])
    W = np.stack([np.bincount(rng.integers(0, n_img, n_img), minlength=n_img)
                  for _ in range(n_boot)]).astype(float)

    def f1_of(s):
        den = 2 * s[:, 0] + s[:, 1] + s[:, 2]
        return np.where(den > 0, 2 * s[:, 0] / np.maximum(den, 1e-12), 0.0)

    def boot(a, b):
        ca, ra, la = store[a]
        cb, rb, lb = store[b]
        d_f1 = f1_of(W @ ca) - f1_of(W @ cb)
        d_ap = np.array([ap_weighted(ra, w) - ap_weighted(rb, w) for w in W])
        full = np.ones(n_img)
        res = {"dF1": (prf(ca.sum(0))[2] - prf(cb.sum(0))[2], d_f1),
               "dAP50": (ap_weighted(ra, full) - ap_weighted(rb, full), d_ap)}
        return la, lb, res

    comps = []
    for tta in (False, True):
        t = " + TTA" if tta else ""
        for k in range(N):
            comps += [(("Homo 3x YOLOv8s" + t, 0), ("Hetero, published WBF" + t, k),
                       "3x YOLOv8s vs YOLOv8s+RT-DETR-l (published setting)"),
                      (("Homo 2x YOLOv8s" + t, k), ("Hetero, equal-weight WBF" + t, k),
                       "2x YOLOv8s vs YOLOv8s+RT-DETR-l (equal size)"),
                      (("Homo 3x YOLOv8s" + t, 0), ("2x YOLOv8s + RT-DETR-l" + t, k),
                       "3x YOLOv8s vs 2x YOLOv8s+RT-DETR-l (equal size)"),
                      (("Homo 3x YOLOv8s, val-tuned WBF" + t, 0), ("Hetero, val-tuned WBF" + t, k),
                       "3x YOLOv8s vs YOLOv8s+RT-DETR-l (both val-tuned WBF)"),
                      (("Homo 3x YOLOv8s" + t, 0), ("YOLOv8s (single)" + t, k),
                       "3x YOLOv8s vs single YOLOv8s")]
            if tta:
                comps.append((("Homo 3x YOLOv8s + TTA", 0), ("Hetero, published WBF + TTA on both", k),
                              "3x YOLOv8s+TTA vs YOLOv8s+RT-DETR-l, TTA on both"))
    brows = []
    for a, b, desc in comps:
        la, lb, res = boot(a, b)
        for nm, (point, d) in res.items():
            lo, hi = np.percentile(d, [2.5, 97.5])
            brows.append({"comparison": desc, "A": la, "B": lb, "metric": nm,
                          "diff_points": round(point * 100, 2), "ci95_low": round(lo * 100, 2),
                          "ci95_high": round(hi * 100, 2), "share_resamples_diff_le_0": round(float(np.mean(d <= 0)), 4)})
    bdf = pd.DataFrame(brows)
    bdf.to_csv(os.path.join(OUT, "bootstrap.csv"), index=False)
    with open(os.path.join(OUT, "bootstrap.md"), "w") as f:
        f.write(f"Paired bootstrap over {n_img} test images, {n_boot} resamples (seed {seed}). "
                "diff = A - B in percentage points; * = TTA.\n\n" + bdf.to_markdown(index=False))

    print("\n" + summ.to_string(index=False))
    print("\n" + pd.DataFrame(comp).to_string(index=False))
    print("\n" + bdf.to_string(index=False))
    print(f"\nAll outputs in {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "predict", "evaluate"], default="all")
    ap.add_argument("--force", action="store_true", help="recompute cached predictions")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args, _ = ap.parse_known_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stage in ("all", "predict"):
        predict_all(args.force)
    if args.stage in ("all", "evaluate"):
        evaluate(args.n_boot, args.seed)


if __name__ == "__main__":
    main()
