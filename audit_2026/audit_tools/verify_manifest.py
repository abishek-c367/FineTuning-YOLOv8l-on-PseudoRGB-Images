#!/usr/bin/env python3
"""Dataset / split manifest builder and verifier for FLIR ADAS v2-style layouts (16 Oct 2026, review P1).

Layout expected (per split directory):  <root>/<split>/data/<stem>.tiff|.tif  (14-bit counts)
                                        <root>/<split>/data/<stem>.jpg          (8-bit AGC)
                                        <root>/<split>/coco.json                (COCO annotations; file_name 'data/<stem>.jpg')
Usage:  python verify_manifest.py --root <root> --splits images_thermal_train images_thermal_val video_thermal_test --out <dir>
Outputs: <out>/manifest_<split>.csv, <out>/REPORT.md ; exit 1 on any HARD failure.

Hard failures: a video id in more than one split; a TIFF without a JPEG pair or with different dimensions;
an exact-duplicate image (SHA-256) across splits; counts above the 14-bit container; a box outside its image;
a category id not declared; a duplicate image id.  Warnings: near-duplicate frames across splits (dHash
Hamming <= 4); JPEG pair whose isotonic-fit residual is large (only if --fit-residual); bit depth not 16.
Nothing here uses labels for anything but integrity checks; the test split's labels are read only for box geometry.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, re, sys
from collections import defaultdict
import numpy as np
from PIL import Image

BITS = 14
FULL = 2 ** BITS - 1
VIDEO_RE = re.compile(r"video-([A-Za-z0-9]+)-frame-(\d+)")


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash(arr: np.ndarray, size: int = 8) -> int:
    img = Image.fromarray(arr.astype(np.float32)).resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(img, dtype=np.float32)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def read_tiff(path: str):
    im = Image.open(path)
    arr = np.asarray(im)
    return arr, im.mode


def video_id(stem: str):
    m = VIDEO_RE.search(stem)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def build_split(root: str, split: str, fit_residual: bool):
    d = os.path.join(root, split, "data")
    coco_path = os.path.join(root, split, "coco.json")
    rows, errors, warnings = [], [], []
    ann = json.load(open(coco_path)) if os.path.exists(coco_path) else None
    cats = {c["id"] for c in ann["categories"]} if ann else set()
    img_by_name = {}
    if ann:
        ids = [im["id"] for im in ann["images"]]
        if len(ids) != len(set(ids)):
            errors.append(f"{split}: duplicate image ids in coco.json")
        for im in ann["images"]:
            img_by_name[os.path.splitext(os.path.basename(im["file_name"]))[0]] = im
        boxes_by_img = defaultdict(list)
        for a in ann["annotations"]:
            boxes_by_img[a["image_id"]].append(a)
    tiffs = sorted(f for f in os.listdir(d) if f.lower().endswith((".tiff", ".tif")))
    for f in tiffs:
        stem = os.path.splitext(f)[0]
        tp = os.path.join(d, f)
        arr, mode = read_tiff(tp)
        jp = next((os.path.join(d, stem + e) for e in (".jpg", ".jpeg", ".JPG") if os.path.exists(os.path.join(d, stem + e))), None)
        vid, frame = video_id(stem)
        row = {"split": split, "stem": stem, "video_id": vid, "frame": frame, "sha256_tiff": sha256(tp), "tiff_mode": mode,
               "height": arr.shape[0], "width": arr.shape[1], "min_count": int(arr.min()), "max_count": int(arr.max()),
               "p1": float(np.percentile(arr, 1)), "p99": float(np.percentile(arr, 99)), "dhash": dhash(arr),
               "has_jpeg": jp is not None, "jpeg_dims_match": None, "n_boxes": 0, "boxes_per_cat": "", "fit_residual": ""}
        if vid is None:
            errors.append(f"{split}/{stem}: no video id parsable from the file name")
        if mode not in ("I;16", "I;16B", "I", "I;16L"):
            warnings.append(f"{split}/{stem}: TIFF mode {mode}, expected 16-bit container")
        if arr.max() > FULL:
            errors.append(f"{split}/{stem}: max count {arr.max()} exceeds the {BITS}-bit container")
        if jp is None:
            errors.append(f"{split}/{stem}: TIFF without a JPEG pair")
        else:
            j = Image.open(jp)
            row["jpeg_dims_match"] = (j.size[1], j.size[0]) == arr.shape[:2]
            if not row["jpeg_dims_match"]:
                errors.append(f"{split}/{stem}: JPEG {j.size} vs TIFF {arr.shape[:2][::-1]} dimension mismatch")
            if fit_residual and row["jpeg_dims_match"]:
                try:
                    from sklearn.isotonic import IsotonicRegression
                    ja = np.asarray(j.convert("L"), dtype=np.float64).reshape(-1); xa = arr.astype(np.float64).reshape(-1)
                    idx = np.random.default_rng(0).choice(xa.size, size=min(10000, xa.size), replace=False)
                    ir = IsotonicRegression(increasing=True).fit(xa[idx], ja[idx])
                    row["fit_residual"] = float(np.sqrt(np.mean((ir.predict(xa[idx]) - ja[idx]) ** 2)))
                    if row["fit_residual"] > 8.0:
                        warnings.append(f"{split}/{stem}: isotonic fit RMS residual {row['fit_residual']:.1f} > 8 grey levels; vendor map not a global monotone curve here")
                except ImportError:
                    warnings.append("scikit-learn not installed; fit residual skipped")
        if ann:
            im = img_by_name.get(stem)
            if im is None:
                warnings.append(f"{split}/{stem}: no coco.json entry")
            else:
                if (im["height"], im["width"]) != arr.shape[:2]:
                    errors.append(f"{split}/{stem}: coco.json dims {(im['height'], im['width'])} vs TIFF {arr.shape[:2]}")
                per = defaultdict(int)
                for a in boxes_by_img.get(im["id"], []):
                    x, y, w, h = a["bbox"]
                    if x < 0 or y < 0 or x + w > im["width"] + 1e-6 or y + h > im["height"] + 1e-6 or w <= 0 or h <= 0:
                        errors.append(f"{split}/{stem}: box {a['bbox']} outside the image or degenerate")
                    if a["category_id"] not in cats:
                        errors.append(f"{split}/{stem}: undeclared category id {a['category_id']}")
                    per[a["category_id"]] += 1
                row["n_boxes"] = sum(per.values()); row["boxes_per_cat"] = ";".join(f"{k}:{v}" for k, v in sorted(per.items()))
        rows.append(row)
    return rows, errors, warnings


def cross_split_checks(all_rows):
    errors, warnings = [], []
    vids = defaultdict(set); shas = defaultdict(set)
    for r in all_rows:
        if r["video_id"]:
            vids[r["video_id"]].add(r["split"])
        shas[r["sha256_tiff"]].add(r["split"])
    for v, s in vids.items():
        if len(s) > 1:
            errors.append(f"video id {v} appears in splits {sorted(s)}")
    for h, s in shas.items():
        if len(s) > 1:
            errors.append(f"exact duplicate TIFF (sha256 {h[:12]}) in splits {sorted(s)}")
    by_split = defaultdict(list)
    for r in all_rows:
        by_split[r["split"]].append(r)
    splits = sorted(by_split)
    for i in range(len(splits)):
        for j in range(i + 1, len(splits)):
            for a in by_split[splits[i]]:
                for b in by_split[splits[j]]:
                    if a["sha256_tiff"] != b["sha256_tiff"] and hamming(a["dhash"], b["dhash"]) <= 4:
                        warnings.append(f"near-duplicate frames: {splits[i]}/{a['stem']} ~ {splits[j]}/{b['stem']} (dHash distance {hamming(a['dhash'], b['dhash'])})")
    return errors, warnings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--fit-residual", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    all_rows, errors, warnings = [], [], []
    for s in args.splits:
        rows, e, w = build_split(args.root, s, args.fit_residual)
        all_rows += rows; errors += e; warnings += w
        with open(os.path.join(args.out, f"manifest_{s}.csv"), "w", newline="") as f:
            if rows:
                wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    e, w = cross_split_checks(all_rows); errors += e; warnings += w
    with open(os.path.join(args.out, "REPORT.md"), "w") as f:
        f.write(f"# Manifest verification report\n\nroot: `{args.root}`\n\n")
        for s in args.splits:
            rs = [r for r in all_rows if r["split"] == s]
            f.write(f"- {s}: {len(rs)} TIFF frames, {len({r['video_id'] for r in rs if r['video_id']})} videos, "
                    f"{sum(r['n_boxes'] for r in rs)} boxes, max count {max((r['max_count'] for r in rs), default='n/a')}\n")
        f.write(f"\n## Hard failures ({len(errors)})\n\n" + "".join(f"- {x}\n" for x in errors) + f"\n## Warnings ({len(warnings)})\n\n" + "".join(f"- {x}\n" for x in warnings))
    print(json.dumps({"frames": len(all_rows), "errors": len(errors), "warnings": len(warnings)}))
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
