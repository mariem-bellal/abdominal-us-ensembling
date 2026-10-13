"""
check_leakage.py - Read-only check of exact and near-duplicate images
across the train / valid / test splits (steps C, D, E, F, G).

Nothing in the dataset is modified, moved or deleted.
All outputs are written to a NEW folder: <project>/leakage_check/

Usage
-----
    pip install pillow imagehash numpy pandas
    python check_leakage.py            # from a terminal
    %run check_leakage.py              # from a Jupyter cell

Outputs (in leakage_check/)
---------------------------
    summary.md                    human-readable summary of every check
    summary.json                  same numbers, machine-readable
    image_index.csv               one row per image (split, index, hashes, box)
    exact_duplicates.csv          cross-split pairs with identical file / pixels
    top_pairs_<A>_<B>.csv         TOP_K most similar cross-split pairs
    nearest_train_<split>.csv     nearest train image for each valid/test image
    adjacent_index_pairs.csv      cross-split pairs whose original indices differ by <= 2
    index_runs.csv                contiguous runs of original indices per split
    sheet_<A>_<B>.png             visual sheets (image A | image B | |A-B|) to inspect
"""

import hashlib
import json
import re
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# ------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------
# Works on Windows and on the SSH server: the dataset folder is looked up
# next to this script, then in the current working directory.
try:
    _HERE = Path(__file__).resolve().parent
except NameError:            # Jupyter without __file__
    _HERE = Path.cwd()
ROOT = next((p for p in (_HERE / "abdomen_mri_yolo", Path.cwd() / "abdomen_mri_yolo")
             if p.is_dir()), _HERE / "abdomen_mri_yolo")
OUT_DIR = ROOT.parent / "leakage_check"

SPLITS = ["train", "valid", "test"]
EXPECTED = {"train": 1887, "valid": 479, "test": 238}
SPLIT_PAIRS = [("train", "valid"), ("train", "test"), ("valid", "test")]

TOP_K = 50        # pairs written to top_pairs_*.csv for each split pair
N_SHEET = 30      # pairs drawn in each sheet_*.png
SMALL = 64        # side of the grayscale thumbnail used for correlation
TILE = 256        # tile size in the sheets
PHASH_THRESHOLDS = [0, 2, 4, 6, 8, 10, 12]
CORR_THRESHOLDS = [0.90, 0.95, 0.98, 0.99]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

# Roboflow export name:  <original index>_png_jpg.rf.<hash>.jpg
NAME_RE = re.compile(r"^(\d+)_png_jpg\.rf\.([0-9a-f]+)\.(jpg|jpeg|png)$", re.I)

# popcount lookup for uint8
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_to_bytes(ih: imagehash.ImageHash) -> np.ndarray:
    """64-bit perceptual hash -> 8 uint8 values."""
    return np.packbits(ih.hash.flatten().astype(np.uint8))


def hamming_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N, 8) uint8, b: (M, 8) uint8 -> (N, M) Hamming distances."""
    x = np.bitwise_xor(a[:, None, :], b[None, :, :])
    return _POP[x].sum(axis=2, dtype=np.int16)


def read_first_box(label_path: Path) -> str:
    if not label_path.exists():
        return ""
    lines = [ln.strip() for ln in label_path.read_text().splitlines() if ln.strip()]
    return f"{len(lines)} box(es); first: {lines[0]}" if lines else "0 box"


def load_split(split: str) -> pd.DataFrame:
    img_dir = ROOT / split / "images"
    lbl_dir = ROOT / split / "labels"
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    rows, thumbs, ph, dh = [], [], [], []
    for i, p in enumerate(files, 1):
        img = Image.open(p)
        gray = img.convert("L")
        arr = np.asarray(gray)
        small = np.asarray(gray.resize((SMALL, SMALL), Image.BILINEAR),
                           dtype=np.float64).ravel()
        small = small - small.mean()
        norm = np.linalg.norm(small)
        thumbs.append(small / norm if norm > 0 else small)
        ph.append(hash_to_bytes(imagehash.phash(gray)))
        dh.append(hash_to_bytes(imagehash.dhash(gray)))
        m = NAME_RE.match(p.name)
        rows.append({
            "split": split,
            "file": p.name,
            "path": str(p),
            "orig_index": int(m.group(1)) if m else np.nan,
            "name_matches_pattern": bool(m),
            "width": img.width,
            "height": img.height,
            "sha256_file": sha256_file(p),
            "sha256_pixels": hashlib.sha256(arr.tobytes()).hexdigest(),
            "label": read_first_box(lbl_dir / (p.stem + ".txt")),
        })
        if i % 500 == 0:
            print(f"  {split}: {i}/{len(files)}")
    df = pd.DataFrame(rows)
    feats = {
        "thumbs": np.vstack(thumbs) if thumbs else np.zeros((0, SMALL * SMALL)),
        "phash": np.vstack(ph) if ph else np.zeros((0, 8), np.uint8),
        "dhash": np.vstack(dh) if dh else np.zeros((0, 8), np.uint8),
    }
    return df, feats


def index_runs(all_df: pd.DataFrame) -> pd.DataFrame:
    d = all_df.dropna(subset=["orig_index"]).sort_values("orig_index")
    runs, cur = [], None
    for idx, split in zip(d["orig_index"].astype(int), d["split"]):
        if cur and cur["split"] == split:
            cur["end"] = idx
            cur["n_images"] += 1
        else:
            if cur:
                runs.append(cur)
            cur = {"split": split, "start": idx, "end": idx, "n_images": 1}
    if cur:
        runs.append(cur)
    return pd.DataFrame(runs)


def make_sheet(pairs: pd.DataFrame, out_path: Path, title: str):
    pairs = pairs.head(N_SHEET)
    if pairs.empty:
        return
    cap_h = 34
    w = 3 * TILE
    h = len(pairs) * (TILE + cap_h) + 30
    sheet = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((5, 8), title, fill="black")
    y = 30
    for _, r in pairs.iterrows():
        a = Image.open(r["path_a"]).convert("L").resize((TILE, TILE))
        b = Image.open(r["path_b"]).convert("L").resize((TILE, TILE))
        diff = Image.fromarray(np.abs(np.asarray(a, np.int16)
                                      - np.asarray(b, np.int16)).astype(np.uint8))
        for k, im in enumerate((a, b, diff)):
            sheet.paste(im.convert("RGB"), (k * TILE, y))
        cap = (f"#{int(r['rank'])} corr={r['corr']:.4f} pHash={int(r['phash_dist'])} "
               f"dHash={int(r['dhash_dist'])} | idx {r['index_a']} vs {r['index_b']}")
        draw.text((5, y + TILE + 2), cap, fill="black")
        draw.text((5, y + TILE + 16), f"{r['file_a'][:55]}  |  {r['file_b'][:55]}",
                  fill="gray")
        y += TILE + cap_h
    sheet.save(out_path)


def describe(x: np.ndarray) -> dict:
    if len(x) == 0:
        return {}
    q = np.percentile(x, [5, 50, 95])
    return {"n": int(len(x)), "p5": float(q[0]), "median": float(q[1]),
            "p95": float(q[2]), "min": float(np.min(x)), "max": float(np.max(x))}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    if not ROOT.is_dir():
        raise SystemExit(f"Dataset folder not found: {ROOT}\n"
                         "Run this script from the project folder that contains abdomen_mri_yolo/.")
    OUT_DIR.mkdir(exist_ok=True)
    summary = {"root": str(ROOT)}

    print("Loading images and computing hashes ...")
    data, feats = {}, {}
    for s in SPLITS:
        data[s], feats[s] = load_split(s)
    all_df = pd.concat([data[s] for s in SPLITS], ignore_index=True)
    all_df.to_csv(OUT_DIR / "image_index.csv", index=False)

    # ---- G. image counts -------------------------------------------------
    counts = {s: int(len(data[s])) for s in SPLITS}
    counts["total"] = int(sum(counts.values()))
    summary["G_counts"] = counts
    summary["G_counts_match_paper"] = all(counts[s] == EXPECTED[s] for s in SPLITS)
    summary["G_image_sizes"] = (all_df.groupby(["width", "height"]).size()
                                .rename("n").reset_index().to_dict("records"))

    # ---- F. file names ---------------------------------------------------
    summary["F_names_not_matching_pattern"] = all_df.loc[
        ~all_df["name_matches_pattern"], "file"].tolist()
    dup_idx = all_df.dropna(subset=["orig_index"]).groupby("orig_index").filter(
        lambda g: len(g) > 1)
    summary["F_same_original_index_in_several_files"] = int(dup_idx["orig_index"].nunique())
    runs = index_runs(all_df)
    runs.to_csv(OUT_DIR / "index_runs.csv", index=False)
    summary["F_number_of_index_runs"] = int(len(runs))
    summary["F_index_range_per_split"] = {
        s: [int(data[s]["orig_index"].min()), int(data[s]["orig_index"].max())]
        for s in SPLITS if data[s]["orig_index"].notna().any()}

    # ---- C. exact duplicates --------------------------------------------
    exact_rows = []
    summary["C_exact_duplicates"] = {}
    for sa, sb in SPLIT_PAIRS:
        res = {}
        for key in ("sha256_file", "sha256_pixels"):
            m = data[sa][["file", key]].merge(data[sb][["file", key]], on=key,
                                               suffixes=("_a", "_b"))
            res[key] = int(len(m))
            for _, r in m.iterrows():
                exact_rows.append({"pair": f"{sa}-{sb}", "hash_type": key,
                                   "file_a": r["file_a"], "file_b": r["file_b"]})
        summary["C_exact_duplicates"][f"{sa}-{sb}"] = res
    summary["C_within_split_exact_duplicates"] = {
        s: int(data[s]["sha256_pixels"].duplicated().sum()) for s in SPLITS}
    pd.DataFrame(exact_rows, columns=["pair", "hash_type", "file_a", "file_b"]).to_csv(
        OUT_DIR / "exact_duplicates.csv", index=False)

    # ---- Reference distributions (to calibrate what "very similar" means)
    # (1) consecutive original indices inside the SAME split: likely
    #     neighbouring frames, a reference for "near-duplicate frames".
    # (2) random cross-split pairs: a reference for "unrelated images".
    rng = np.random.default_rng(0)
    ref = {}
    cons_corr, cons_ph = [], []
    for s in SPLITS:
        d = data[s]
        pos = {int(v): i for i, v in d["orig_index"].dropna().items()}
        T, P = feats[s]["thumbs"], feats[s]["phash"]
        for idx, i in pos.items():
            j = pos.get(idx + 1)
            if j is not None:
                cons_corr.append(float(T[i] @ T[j]))
                cons_ph.append(int(_POP[np.bitwise_xor(P[i], P[j])].sum()))
    ref["consecutive_index_same_split"] = {"corr": describe(np.array(cons_corr)),
                                           "phash": describe(np.array(cons_ph))}
    rnd_corr, rnd_ph = [], []
    for sa, sb in SPLIT_PAIRS:
        FA, FB = feats[sa], feats[sb]
        ia = rng.integers(0, len(data[sa]), 2000)
        ib = rng.integers(0, len(data[sb]), 2000)
        rnd_corr += list(np.einsum("ij,ij->i", FA["thumbs"][ia], FB["thumbs"][ib]))
        rnd_ph += list(_POP[np.bitwise_xor(FA["phash"][ia], FB["phash"][ib])].sum(axis=1))
    ref["random_cross_split"] = {"corr": describe(np.array(rnd_corr)),
                                 "phash": describe(np.array(rnd_ph))}
    summary["D_reference_distributions"] = ref

    # ---- D. near duplicates ---------------------------------------------
    summary["D_near_duplicates"] = {}
    for sa, sb in SPLIT_PAIRS:
        A, B = data[sa], data[sb]
        corr = feats[sa]["thumbs"] @ feats[sb]["thumbs"].T
        ph = hamming_matrix(feats[sa]["phash"], feats[sb]["phash"])
        dh = hamming_matrix(feats[sa]["dhash"], feats[sb]["dhash"])

        res = {"n_pairs_compared": int(corr.size)}
        res["pairs_phash_le"] = {str(t): int((ph <= t).sum()) for t in PHASH_THRESHOLDS}
        res["pairs_corr_ge"] = {str(t): int((corr >= t).sum()) for t in CORR_THRESHOLDS}
        # images of split B that have at least one close image in split A
        res["imagesB_with_phash_le"] = {str(t): int((ph <= t).any(axis=0).sum())
                                        for t in PHASH_THRESHOLDS}
        res["imagesB_with_corr_ge"] = {str(t): int((corr >= t).any(axis=0).sum())
                                       for t in CORR_THRESHOLDS}
        res["max_corr"] = float(corr.max())
        res["min_phash"] = int(ph.min())
        summary["D_near_duplicates"][f"{sa}-{sb}"] = res

        # top-K pairs: by correlation, plus by pHash (union, ranked by corr)
        k = min(TOP_K, corr.size)
        flat_c = np.argpartition(-corr.ravel(), k - 1)[:k]
        flat_p = np.argpartition(ph.ravel(), k - 1)[:k]
        flat = np.unique(np.concatenate([flat_c, flat_p]))
        ia, ib = np.unravel_index(flat, corr.shape)
        top = pd.DataFrame({
            "file_a": A.loc[ia, "file"].values, "file_b": B.loc[ib, "file"].values,
            "index_a": A.loc[ia, "orig_index"].values, "index_b": B.loc[ib, "orig_index"].values,
            "corr": corr[ia, ib], "phash_dist": ph[ia, ib], "dhash_dist": dh[ia, ib],
            "label_a": A.loc[ia, "label"].values, "label_b": B.loc[ib, "label"].values,
            "path_a": A.loc[ia, "path"].values, "path_b": B.loc[ib, "path"].values,
        }).sort_values(["corr", "phash_dist"], ascending=[False, True]).reset_index(drop=True)
        top.insert(0, "rank", np.arange(1, len(top) + 1))
        top["case_to_fill_after_inspection"] = ""   # 1 = same image, 2 = near frame, 3 = different
        top.to_csv(OUT_DIR / f"top_pairs_{sa}_{sb}.csv", index=False)
        make_sheet(top, OUT_DIR / f"sheet_{sa}_{sb}.png",
                   f"{sa} (left) vs {sb} (middle) | right: absolute difference")

        # nearest train image for every valid / test image
        if sa == "train":
            j = corr.argmax(axis=0)
            near = pd.DataFrame({
                "file": B["file"], "orig_index": B["orig_index"],
                "nearest_train_file": A.loc[j, "file"].values,
                "nearest_train_index": A.loc[j, "orig_index"].values,
                "corr": corr[j, np.arange(len(B))],
                "phash_dist": ph[j, np.arange(len(B))],
                "min_phash_any_train": ph.min(axis=0),
            }).sort_values("corr", ascending=False)
            near.to_csv(OUT_DIR / f"nearest_train_{sb}.csv", index=False)

    # ---- F (bis). cross-split pairs with close original indices ----------
    adj = []
    lookup = {int(r.orig_index): r for r in all_df.dropna(subset=["orig_index"]).itertuples()}
    pos = {s: {int(v): i for i, v in data[s]["orig_index"].dropna().items()} for s in SPLITS}
    for idx, r in lookup.items():
        for d in (1, 2):
            o = lookup.get(idx + d)
            if o is not None and o.split != r.split:
                ta = feats[r.split]["thumbs"][pos[r.split][idx]]
                tb = feats[o.split]["thumbs"][pos[o.split][idx + d]]
                pa = feats[r.split]["phash"][pos[r.split][idx]]
                pb = feats[o.split]["phash"][pos[o.split][idx + d]]
                adj.append({"index_a": idx, "split_a": r.split, "file_a": r.file,
                            "index_b": idx + d, "split_b": o.split, "file_b": o.file,
                            "corr": float(ta @ tb),
                            "phash_dist": int(_POP[np.bitwise_xor(pa, pb)].sum())})
    adj_df = pd.DataFrame(adj, columns=["index_a", "split_a", "file_a", "index_b", "split_b",
                                        "file_b", "corr", "phash_dist"])
    adj_df.to_csv(OUT_DIR / "adjacent_index_pairs.csv", index=False)
    summary["F_cross_split_pairs_index_gap_le_2"] = int(len(adj_df))

    # ---- write summary ----------------------------------------------------
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    lines = ["# Leakage check (read-only)", "", f"Root: `{ROOT}`", "",
             "## G. Image counts", ""]
    for s in SPLITS:
        lines.append(f"- {s}: {counts[s]} (expected {EXPECTED[s]})")
    lines += [f"- total: {counts['total']}", "", "## C. Exact duplicates (cross-split)", ""]
    for p, v in summary["C_exact_duplicates"].items():
        lines.append(f"- {p}: identical file = {v['sha256_file']}, "
                     f"identical pixels = {v['sha256_pixels']}")
    lines += ["", "## D. Reference distributions", ""]
    for name, v in ref.items():
        lines.append(f"- {name}: corr {v['corr']} | pHash {v['phash']}")
    lines += ["", "## D. Near duplicates (cross-split)", ""]
    for p, v in summary["D_near_duplicates"].items():
        lines.append(f"### {p}")
        lines.append(f"- max corr = {v['max_corr']:.4f}, min pHash distance = {v['min_phash']}")
        lines.append(f"- images of 2nd split with a pHash match <= t: {v['imagesB_with_phash_le']}")
        lines.append(f"- images of 2nd split with corr >= t: {v['imagesB_with_corr_ge']}")
        lines.append("")
    lines += ["## F. File names", "",
              f"- names not matching `<index>_png_jpg.rf.<hash>`: "
              f"{len(summary['F_names_not_matching_pattern'])}",
              f"- original indices used by more than one file: "
              f"{summary['F_same_original_index_in_several_files']}",
              f"- contiguous index runs: {summary['F_number_of_index_runs']} (see index_runs.csv)",
              f"- cross-split pairs with index gap <= 2: "
              f"{summary['F_cross_split_pairs_index_gap_le_2']} (see adjacent_index_pairs.csv)",
              "", "## E. Manual inspection", "",
              "Open sheet_*.png and fill `case_to_fill_after_inspection` in top_pairs_*.csv:",
              "1 = same image, 2 = near-identical frame, 3 = different image, similar content.",
              "Compare the scores with the 'consecutive_index_same_split' reference above."]
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nAll outputs written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
