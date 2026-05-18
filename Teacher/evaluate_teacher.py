"""
evaluate_teacher.py
--------------------
Evaluates YOLOv8 teacher model detection accuracy on thermal images
converted to pseudo-RGB via CycleGAN.

Pipeline:
    thermal image  ->  CycleGAN (B2A)  ->  pseudo-RGB  ->  YOLOv8 teacher  ->  predictions
                                                                                      |
                                                                               vs ground-truth labels
                                                                                      |
                                                                               mAP / P / R / F1

Usage:
    python evaluate_teacher.py \
        --thermal_dir  /path/to/FLIR_ADAS_v2/images_thermal_val/data \
        --labels_dir   /path/to/FLIR_ADAS_v2/labels_val \           # YOLO-format .txt files
        --cyclegan_weights cyclegan_epoch_45.pth \
        --yolo_weights     yolov8l.pt \
        --img_size 640 \
        --conf_thresh 0.25 \
        --iou_thresh  0.50 \
        --batch_size  8 \
        --num_workers 4 \
        --save_visuals          # optional: save side-by-side images with boxes
        --visuals_dir  ./vis_output

Label format expected (YOLO .txt per image):
    class_id  cx  cy  w  h   (all normalised 0-1)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import kornia.augmentation as K
# ──────────────────────────────────────────────────────────────────────────────
# Optional: rich tables for pretty printing
# ──────────────────────────────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.table import Table
    _console = Console()
    def _print(*a, **kw): _console.print(*a, **kw)
except ImportError:
    _console = None
    def _print(*a, **kw): print(*a, **kw)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class ThermalEvalDataset(Dataset):
    """
    Loads thermal images + their YOLO-format ground-truth labels.
    Returns raw uint8 image (H,W,3) and list of [cls, cx, cy, w, h] boxes.
    """

    def __init__(
        self,
        thermal_dir: str,
        labels_dir: str,
        img_size: int = 512,
    ) -> None:
        self.img_size = img_size
        thermal_dir = Path(thermal_dir)
        labels_dir  = Path(labels_dir)

        image_paths = sorted(
            p for p in thermal_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTS
        )

        self.samples: List[Tuple[Path, Path]] = []
        missing_labels = 0
        for img_path in image_paths:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                self.samples.append((img_path, lbl_path))
            else:
                missing_labels += 1

        if missing_labels:
            _print(f"[yellow]Warning:[/yellow] {missing_labels} images skipped (no matching label).")
        if not self.samples:
            raise RuntimeError(
                f"No paired image+label files found.\n"
                f"  thermal_dir : {thermal_dir}\n"
                f"  labels_dir  : {labels_dir}"
            )

        _print(f"[green]Dataset:[/green] {len(self.samples)} paired samples found.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, lbl_path = self.samples[idx]
        # Read as-is (handles 8-bit, 16-bit, 1-channel or 3-channel)
        img_raw = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img_raw is None:
            raise IOError(f"Could not read image: {img_path}")

        # If 16-bit thermal, normalise to 8-bit
        if img_raw.dtype == np.uint16:
            img_raw = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # If single-channel (grayscale/thermal), replicate to 3 channels (H,W) → (H,W,3)
        if img_raw.ndim == 2:
            img_rgb = np.stack([img_raw, img_raw, img_raw], axis=-1)  # grayscale → RGB
        elif img_raw.shape[2] == 1:
            img_rgb = np.repeat(img_raw, 3, axis=-1)                  # (H,W,1) → (H,W,3)
        else:
            img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)        # normal BGR → RGB

        # # ── Load image (always 3-channel) ──────────────────────────────────
        # img_bgr = cv2.imread(str(img_path))
        # if img_bgr is None:
        #     raise IOError(f"Could not read image: {img_path}")
        # img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_rgb.shape[:2]

        # Resize to network input size (keep as uint8 numpy for CycleGAN pre-proc later)
        img_resized = cv2.resize(img_rgb, (self.img_size, self.img_size))

        # ── Load labels ────────────────────────────────────────────────────
        boxes: List[List[float]] = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id, cx, cy, w, h = parts
                    boxes.append([int(cls_id), float(cx), float(cy), float(w), float(h)])

        return {
            "image":     img_resized,           # ndarray uint8 (H,W,3) RGB
            "boxes":     boxes,                  # list of [cls, cx, cy, w, h]
            "img_path":  str(img_path),
            "orig_hw":   (orig_h, orig_w),
        }


def collate_fn(batch):
    """Custom collate: images → stacked tensor, boxes kept as list-of-lists."""
    images  = [b["image"]    for b in batch]
    boxes   = [b["boxes"]    for b in batch]
    paths   = [b["img_path"] for b in batch]
    orig_hws= [b["orig_hw"]  for b in batch]
    images_np = np.stack(images, axis=0)   # (B,H,W,3)
    return {"images": images_np, "boxes": boxes, "img_paths": paths, "orig_hws": orig_hws}


# ──────────────────────────────────────────────────────────────────────────────
# CycleGAN helpers  (mirrors your existing loader API)
# ──────────────────────────────────────────────────────────────────────────────

def build_cyclegan(weights: str, device: torch.device, input_nc=3, output_nc=3, n_res=9):
    """Load CycleGAN generator B→A (thermal → pseudo-RGB)."""
    from Cycle_gan.cyclegan_loader import load_cyclegan_generator
    from Cycle_gan.cyclegan import Generator
    G = load_cyclegan_generator(
        weights, device=device,
        input_nc=input_nc, output_nc=output_nc,
        n_residual_blocks=n_res, direction="B2A"
    )
    G.eval()
    return G

gpu_normalize = K.Normalize(
    mean=torch.tensor([0.5, 0.5, 0.5]),
    std=torch.tensor([0.5, 0.5, 0.5]),
)
# def gpu_normalize(x: torch.Tensor) -> torch.Tensor:
#     """Normalise [0,1] tensor to [-1,1]."""
#     return x * 2.0 - 1.0


@torch.no_grad()
def thermal_batch_to_pseudo_rgb(
    images_np: np.ndarray,     # (B,H,W,3) uint8 RGB
    G: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Convert a batch of thermal images to pseudo-RGB via CycleGAN.
    Returns float32 tensor in [0,1] with shape (B,3,H,W).
    """
    # uint8 → float32 [0,1] → tensor (B,3,H,W)
    x = torch.from_numpy(images_np).float().div(255.0)   # (B,H,W,3)
    x = x.permute(0, 3, 1, 2).to(device)                 # (B,3,H,W)
    x = gpu_normalize(x)
    out = G(x)
    out = (out + 1.0) / 2.0        # [-1,1] → [0,1]
    return out.clamp(0.0, 1.0)


# ──────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ──────────────────────────────────────────────────────────────────────────────

def xywhn_to_xyxy(boxes_norm: List[List[float]], img_w: int, img_h: int) -> np.ndarray:
    """Convert normalised [cls, cx, cy, w, h] → absolute [cls, x1, y1, x2, y2]."""
    if not boxes_norm:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.array(boxes_norm, dtype=np.float32)          # (N,5)
    cls = arr[:, 0:1]
    cx, cy, w, h = arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return np.stack([cls[:, 0], x1, y1, x2, y2], axis=1)


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """
    Compute IoU between every pair (a_i, b_j).
    boxes_a / boxes_b: (N,4) / (M,4) in xyxy format.
    Returns (N,M) float32 IoU matrix.
    """
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
    bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

    inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
    inter_y1 = np.maximum(ay1[:, None], by1[None, :])
    inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
    inter_y2 = np.minimum(ay2[:, None], by2[None, :])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter   = inter_w * inter_h

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union  = area_a[:, None] + area_b[None, :] - inter + 1e-7

    return (inter / union).astype(np.float32)


def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute area under precision-recall curve using 11-point interpolation."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        mask = recalls >= thr
        ap += precisions[mask].max() if mask.any() else 0.0
    return ap / 11.0


class DetectionMetrics:
    """Accumulates per-image predictions and GTs, then computes P/R/mAP."""

    def __init__(self, iou_thresh: float = 0.50, num_classes: int = 80) -> None:
        self.iou_thresh  = iou_thresh
        self.num_classes = num_classes
        # per-class lists of (confidence, tp_flag)
        self._detections: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        self._n_gt: Dict[int, int]                           = defaultdict(int)

    def update(
        self,
        pred_boxes:  np.ndarray,   # (N,6) [x1,y1,x2,y2,conf,cls]
        gt_boxes_abs: np.ndarray,  # (M,5) [cls,x1,y1,x2,y2]
        img_w: int,
        img_h: int,
    ) -> None:
        # Count GTs per class
        for cls_id in gt_boxes_abs[:, 0].astype(int):
            self._n_gt[cls_id] += 1

        if len(pred_boxes) == 0:
            return

        # Per-class matching
        gt_matched = np.zeros(len(gt_boxes_abs), dtype=bool)

        # Sort preds by confidence (high → low)
        order = np.argsort(-pred_boxes[:, 4])
        pred_boxes = pred_boxes[order]

        for pred in pred_boxes:
            px1, py1, px2, py2, conf, pcls = pred
            pcls = int(pcls)

            # Find GTs of same class
            same_cls_mask = (gt_boxes_abs[:, 0].astype(int) == pcls)
            gt_same = gt_boxes_abs[same_cls_mask]
            gt_idx_map = np.where(same_cls_mask)[0]

            tp = 0
            if len(gt_same) > 0:
                ious = box_iou_matrix(
                    np.array([[px1, py1, px2, py2]], dtype=np.float32),
                    gt_same[:, 1:].astype(np.float32),
                )[0]  # (M_cls,)
                best_iou_idx = int(np.argmax(ious))
                best_iou     = ious[best_iou_idx]
                orig_idx     = gt_idx_map[best_iou_idx]
                if best_iou >= self.iou_thresh and not gt_matched[orig_idx]:
                    tp = 1
                    gt_matched[orig_idx] = True

            self._detections[pcls].append((float(conf), tp))

    def compute(self) -> Dict:
        """Return dict with per-class AP and overall mAP, P, R."""
        class_aps: Dict[int, float] = {}
        all_p, all_r = [], []

        for cls_id, dets in self._detections.items():
            n_gt = self._n_gt.get(cls_id, 0)
            if n_gt == 0:
                continue

            dets_sorted = sorted(dets, key=lambda x: -x[0])
            confs = np.array([d[0] for d in dets_sorted])
            tps   = np.array([d[1] for d in dets_sorted], dtype=np.float32)
            fps   = 1 - tps

            tp_cum = np.cumsum(tps)
            fp_cum = np.cumsum(fps)

            recalls    = tp_cum / (n_gt + 1e-7)
            precisions = tp_cum / (tp_cum + fp_cum + 1e-7)

            ap = compute_ap(recalls, precisions)
            class_aps[cls_id] = ap
            all_p.append(float(precisions[-1]) if len(precisions) else 0.0)
            all_r.append(float(recalls[-1])    if len(recalls)    else 0.0)

        mAP = float(np.mean(list(class_aps.values()))) if class_aps else 0.0
        mean_p = float(np.mean(all_p)) if all_p else 0.0
        mean_r = float(np.mean(all_r)) if all_r else 0.0
        f1 = 2 * mean_p * mean_r / (mean_p + mean_r + 1e-7)

        return {
            "mAP@50":      mAP,
            "Precision":   mean_p,
            "Recall":      mean_r,
            "F1":          f1,
            "per_class_AP": class_aps,
            "total_gt":    dict(self._n_gt),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation helper
# ──────────────────────────────────────────────────────────────────────────────

def save_visual(
    thermal_img_np: np.ndarray,     # (H,W,3) uint8 RGB
    pseudo_rgb_tensor: torch.Tensor,# (3,H,W) float [0,1]
    pred_boxes: np.ndarray,         # (N,6) [x1,y1,x2,y2,conf,cls]
    gt_boxes_abs: np.ndarray,       # (M,5) [cls,x1,y1,x2,y2]
    save_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    pseudo_np = (pseudo_rgb_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    thermal_bgr = cv2.cvtColor(thermal_img_np, cv2.COLOR_RGB2BGR)
    pseudo_bgr  = cv2.cvtColor(pseudo_np,      cv2.COLOR_RGB2BGR)

    # Draw GT (green)
    for box in gt_boxes_abs:
        cls_id, x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3]), int(box[4])
        cv2.rectangle(pseudo_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        lbl = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        cv2.putText(pseudo_bgr, f"GT:{lbl}", (x1, max(y1-4, 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

    # Draw predictions (red)
    for box in pred_boxes:
        x1, y1, x2, y2, conf, cls_id = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4], int(box[5])
        cv2.rectangle(pseudo_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
        lbl = class_names[cls_id] if class_names and cls_id < len(class_names) else str(cls_id)
        cv2.putText(pseudo_bgr, f"{lbl}:{conf:.2f}", (x1, max(y1-4, 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

    # Side-by-side: thermal | pseudo-RGB with boxes
    combo = np.concatenate([thermal_bgr, pseudo_bgr], axis=1)
    cv2.imwrite(save_path, combo)


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ──────────────────────────────────────────────────────────────────────────────

def run_evaluation(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _print(f"\n[bold cyan]Device:[/bold cyan] {device}")

    # ── Load CycleGAN ─────────────────────────────────────────────────────
    _print("\n[bold]Loading CycleGAN generator …[/bold]")
    G = build_cyclegan(args.cyclegan_weights, device)

    # ── Load YOLOv8 teacher ───────────────────────────────────────────────
    _print("[bold]Loading YOLOv8 teacher …[/bold]")
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    yolo = YOLO(args.yolo_weights)
    yolo_model = yolo.model.eval().to(device)

    # ── Dataset & DataLoader ──────────────────────────────────────────────
    dataset = ThermalEvalDataset(
        thermal_dir=args.thermal_dir,
        labels_dir=args.labels_dir,
        img_size=args.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory= False,
        # pin_memory=(device.type == "cuda"),
    )

    # ── Optional: class names from dataset YAML ───────────────────────────
    class_names: Optional[List[str]] = None
    if args.data_yaml:
        try:
            import yaml
            with open(args.data_yaml) as f:
                data = yaml.safe_load(f)
            class_names = data.get("names", None)
            _print(f"[green]Class names loaded:[/green] {class_names}")
        except Exception as e:
            _print(f"[yellow]Could not load class names from YAML: {e}[/yellow]")

    # ── Visuals dir ───────────────────────────────────────────────────────
    if args.save_visuals:
        vis_dir = Path(args.visuals_dir)
        vis_dir.mkdir(parents=True, exist_ok=True)
        vis_count = 0

    # ── Metric accumulator ────────────────────────────────────────────────
    metrics = DetectionMetrics(
        iou_thresh=args.iou_thresh,
        num_classes=len(class_names) if class_names else 80,
    )

    total_images = 0
    total_time   = 0.0

    _print(f"\n[bold]Starting evaluation on {len(dataset)} images …[/bold]\n")

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", unit="batch"):
            images_np  = batch["images"]   # (B,H,W,3) uint8
            gt_boxes_b = batch["boxes"]    # list of lists
            img_paths  = batch["img_paths"]

            B = len(images_np)
            total_images += B

            # ── Step 1: CycleGAN thermal → pseudo-RGB ─────────────────
            t0 = time.perf_counter()
            pseudo_rgb = thermal_batch_to_pseudo_rgb(images_np, G, device)

            results = yolo.predict(
                pseudo_rgb,
                conf=args.conf_thresh,
                iou=args.nms_iou,
                imgsz=args.img_size,
                verbose=False,
                device=device,
            )
            total_time += time.perf_counter() - t0

            # ── Step 3: Accumulate metrics ────────────────────────────
            for i, result in enumerate(results):
                img_w = img_h = args.img_size

                # Ground truth
                gt_raw = gt_boxes_b[i]
                gt_abs = xywhn_to_xyxy(gt_raw, img_w, img_h)  # (M,5) [cls,x1,y1,x2,y2]

                # Predictions
                boxes  = result.boxes
                if boxes is not None and len(boxes) > 0:
                    xyxy  = boxes.xyxy.cpu().numpy()   # (N,4)
                    confs = boxes.conf.cpu().numpy()   # (N,)
                    clss  = boxes.cls.cpu().numpy()    # (N,)
                    pred_arr = np.concatenate(
                        [xyxy, confs[:, None], clss[:, None]], axis=1
                    )  # (N,6)
                else:
                    pred_arr = np.zeros((0, 6), dtype=np.float32)

                metrics.update(pred_arr, gt_abs, img_w, img_h)

                # ── Optional visuals ──────────────────────────────────
                if args.save_visuals and vis_count < args.max_visuals:
                    stem = Path(img_paths[i]).stem
                    save_path = str(vis_dir / f"{stem}_eval.jpg")
                    save_visual(
                        thermal_img_np   = images_np[i],
                        pseudo_rgb_tensor= pseudo_rgb[i],
                        pred_boxes       = pred_arr,
                        gt_boxes_abs     = gt_abs,
                        save_path        = save_path,
                        class_names      = class_names,
                    )
                    vis_count += 1

    # ── Print results ──────────────────────────────────────────────────────
    results_dict = metrics.compute()
    fps = total_images / total_time if total_time > 0 else 0.0

    _print("\n" + "═" * 55)
    _print("[bold green]  TEACHER MODEL EVALUATION RESULTS[/bold green]")
    _print("═" * 55)
    _print(f"  Images evaluated : {total_images}")
    _print(f"  Throughput       : {fps:.1f} img/s  ({1000/fps:.1f} ms/img)")
    _print(f"  IoU threshold    : {args.iou_thresh:.2f}")
    _print(f"  Conf threshold   : {args.conf_thresh:.2f}")
    _print("─" * 55)
    _print(f"  [bold]mAP@50    : {results_dict['mAP@50']:.4f}[/bold]")
    _print(f"  Precision : {results_dict['Precision']:.4f}")
    _print(f"  Recall    : {results_dict['Recall']:.4f}")
    _print(f"  F1        : {results_dict['F1']:.4f}")

    per_cls = results_dict["per_class_AP"]
    if per_cls:
        _print("\n  Per-class AP:")
        for cls_id, ap in sorted(per_cls.items()):
            n_gt = results_dict["total_gt"].get(cls_id, 0)
            name = class_names[cls_id] if class_names and cls_id < len(class_names) else f"cls{cls_id}"
            _print(f"    [{cls_id:>3}] {name:<20s}  AP={ap:.4f}  (GT={n_gt})")
    _print("═" * 55 + "\n")

    # ── Save to file ───────────────────────────────────────────────────────
    if args.save_results:
        import json
        out = {
            "mAP@50":    results_dict["mAP@50"],
            "Precision": results_dict["Precision"],
            "Recall":    results_dict["Recall"],
            "F1":        results_dict["F1"],
            "per_class_AP": {str(k): v for k, v in per_cls.items()},
            "total_gt":  {str(k): v for k, v in results_dict["total_gt"].items()},
            "iou_thresh":  args.iou_thresh,
            "conf_thresh": args.conf_thresh,
            "total_images": total_images,
            "fps": fps,
        }
        with open(args.save_results, "w") as f:
            json.dump(out, f, indent=2)
        _print(f"[green]Results saved to:[/green] {args.save_results}")

    if args.save_visuals:
        _print(f"[green]Visuals saved to:[/green] {args.visuals_dir}  ({vis_count} images)")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate YOLOv8 teacher on CycleGAN pseudo-RGB images."
    )
    # Paths
    p.add_argument("--thermal_dir",      required=True,
                   help="Directory of thermal validation images.")
    p.add_argument("--labels_dir",       required=True,
                   help="Directory of YOLO-format .txt label files.")
    p.add_argument("--cyclegan_weights", default="cyclegan_epoch_45.pth",
                   help="CycleGAN .pth checkpoint.")
    p.add_argument("--yolo_weights",     default="yolov8l.pt",
                   help="YOLOv8 teacher weights (.pt).")
    p.add_argument("--data_yaml",        default=None,
                   help="Optional path to dataset YAML for class names.")

    # Inference params
    p.add_argument("--img_size",    type=int,   default=640)
    p.add_argument("--conf_thresh", type=float, default=0.25,
                   help="Detection confidence threshold.")
    p.add_argument("--iou_thresh",  type=float, default=0.50,
                   help="IoU threshold for mAP matching.")
    p.add_argument("--nms_iou",     type=float, default=0.45,
                   help="NMS IoU threshold for YOLOv8 predict.")
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--num_workers", type=int,   default=4)

    # Output
    p.add_argument("--save_results",  default="eval_results.json",
                   help="Path to save JSON results (pass '' to skip).")
    p.add_argument("--save_visuals",  action="store_true",
                   help="Save side-by-side (thermal | pseudo-RGB+boxes) images.")
    p.add_argument("--visuals_dir",   default="./eval_visuals")
    p.add_argument("--max_visuals",   type=int, default=50,
                   help="Maximum number of visual outputs to save.")

    return p.parse_args()


if __name__ == "__main__":
    import sys
    sys.argv = [
        "evaluate_teacher.py",
        "--thermal_dir",  "C:\Project\Project_Detection\FLIR_ADAS_v2\images_thermal_val\data",
        "--labels_dir"  , "C:\Project\Project_Detection\FLIR_ADAS_v2\images_thermal_val\labels_thermal_val" ,
        "--cyclegan_weights" ,"C:\Project\Project_Detection\Cycle_gan\cyclegan_epoch_45.pth ",
        "--yolo_weights" ,    "C:\Project\Project_Detection\yolov8s.pt",
        "--img_size" ,"512" ,
        "--conf_thresh", "0.25",
        "--iou_thresh" , "0.50 ",
        "--batch_size" , "1" ,
        "--num_workers", "4",
        "--save_visuals" ,
        "--visuals_dir", "C:\Project\Project_Detection\Saved_Visuals"]
    args = parse_args()
    run_evaluation(args)
