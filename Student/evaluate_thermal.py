"""
evaluate_thermal.py
-------------------
Evaluate a YOLOv8Thermal student model loaded from a distillation
checkpoint against a FLIR/COCO-format thermal validation set.

Outputs:
  - Per-class AP @IoU=0.50
  - mAP50  (primary metric)
  - mAP50-95
  - Precision / Recall at best F1 threshold

Usage:
  python Student/evaluate_thermal.py \
      --checkpoint checkpoints_distill/distill_epoch_35.pth \
      --flir-root  FLIR_ADAS_v2 \
      --split      images_thermal_val \
      --coco-json  coco.json \
      --num-classes 3 \
      --imgsz      512 \
      --batch      8 \
      --conf       0.25 \
      --iou-nms    0.45 \
      --device     cuda
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import sys
import os

# Add parent directory to path to support both module and direct execution
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw

# ── Import your project modules (adjust paths if needed) ─────────────────────
from Student.yolov8_thermal import yolov8s_thermal   # model factory
# If your dataset loader is in a different path, adjust these two imports:
from data.flir_dataset import (
    FlirCocoPaths,
    FlirThermalCocoDataset,
    ultralytics_collate,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_student_from_checkpoint(
    checkpoint_path: str,
    num_classes: int,
    include_p2: bool = True,
    use_transformer_neck: bool = False,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    Reconstruct the YOLOv8Thermal student and load weights from a
    distillation checkpoint saved by DistillationTrainer.save_checkpoint().

    The checkpoint dict has the key 'student_state'.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    model = yolov8s_thermal(
        num_classes=num_classes,
        include_p2=include_p2,
        use_transformer_neck=use_transformer_neck,
    )

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "student_state" not in ckpt:
        raise KeyError(
            f"Key 'student_state' not found in checkpoint. "
            f"Available keys: {list(ckpt.keys())}"
        )

    model.load_state_dict(ckpt["student_state"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"✓ Loaded student weights from epoch {epoch}  ({checkpoint_path})")
    return model, device


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Decode raw head output → bounding boxes
# ─────────────────────────────────────────────────────────────────────────────

def _make_anchors(feats: List[torch.Tensor], strides: torch.Tensor, offset: float = 0.5):
    """Build anchor centre-points for each feature level."""
    anchor_points, stride_tensor = [], []
    for feat, stride in zip(feats, strides):
        _, _, h, w = feat.shape
        sx = torch.arange(w, device=feat.device) + offset
        sy = torch.arange(h, device=feat.device) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack([sx, sy], dim=-1).reshape(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, device=feat.device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def _dfl_decode(reg: torch.Tensor, reg_max: int = 16) -> torch.Tensor:
    """
    Decode DFL distribution to ltrb distances.
    reg: (N, 4*reg_max)
    returns: (N, 4)
    """
    N = reg.shape[0]
    reg = reg.reshape(N, 4, reg_max)
    # soft-argmax
    prob = reg.softmax(dim=-1)
    grid = torch.arange(reg_max, device=reg.device, dtype=reg.dtype)
    return (prob * grid).sum(dim=-1)   # (N, 4)


def decode_predictions(
    raw_preds: List[torch.Tensor],
    strides: torch.Tensor,
    num_classes: int,
    reg_max: int = 16,
    conf_thres: float = 0.25,
) -> List[torch.Tensor]:
    """
    Convert raw head tensors to boxes.

    raw_preds : List[Tensor(B, 4*reg_max + nc, H, W)]  — one per scale
    Returns   : List[Tensor(K, 6)]  one per image;
                columns = [x1, y1, x2, y2, score, class_id]
                coordinates are in pixels of the INPUT image.
    """
    B = raw_preds[0].shape[0]

    # Collect all predictions across scales
    all_ltrb, all_scores, all_stride = [], [], []

    anchor_points, stride_tensor = _make_anchors(raw_preds, strides)

    offset = 0
    for feat, stride in zip(raw_preds, strides.tolist()):
        b, c, h, w = feat.shape
        n = h * w
        feat_flat = feat.permute(0, 2, 3, 1).reshape(B, n, c)  # (B, n, c)

        ltrb_raw = feat_flat[..., : 4 * reg_max]   # (B, n, 4*reg_max)
        cls_raw  = feat_flat[..., 4 * reg_max:]    # (B, n, nc)

        all_ltrb.append(ltrb_raw)
        all_scores.append(cls_raw.sigmoid())
        offset += n

    all_ltrb   = torch.cat(all_ltrb,   dim=1)   # (B, total_anchors, 4*reg_max)
    all_scores = torch.cat(all_scores, dim=1)   # (B, total_anchors, nc)

    results = []
    for b in range(B):
        ltrb_b  = all_ltrb[b]    # (A, 4*reg_max)
        score_b = all_scores[b]  # (A, nc)

        # Decode DFL
        ltrb_dec = _dfl_decode(ltrb_b, reg_max)  # (A, 4)

        # Convert to xyxy using anchor centres
        xy = anchor_points                             # (A, 2)
        lt = ltrb_dec[:, :2]
        rb = ltrb_dec[:, 2:]
        x1y1 = (xy - lt) * stride_tensor
        x2y2 = (xy + rb) * stride_tensor
        boxes = torch.cat([x1y1, x2y2], dim=-1)       # (A, 4)  pixel coords

        # Score filter
        max_score, class_id = score_b.max(dim=-1)     # (A,)
        keep = max_score >= conf_thres
        if keep.sum() == 0:
            results.append(torch.zeros((0, 6), device=ltrb_b.device))
            continue

        boxes    = boxes[keep]
        scores   = max_score[keep]
        class_id = class_id[keep].float()

        dets = torch.cat([boxes, scores.unsqueeze(1), class_id.unsqueeze(1)], dim=1)
        results.append(dets)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  NMS
# ─────────────────────────────────────────────────────────────────────────────

def nms_per_class(
    dets: torch.Tensor,
    iou_thres: float = 0.45,
    max_det: int = 300,
) -> torch.Tensor:
    """
    Class-aware NMS.
    dets: (K, 6)  [x1, y1, x2, y2, score, class_id]
    """
    if dets.shape[0] == 0:
        return dets

    from torchvision.ops import nms  # torchvision or you can use a pure-torch impl

    keep_all = []
    for cls in dets[:, 5].unique():
        mask = dets[:, 5] == cls
        d    = dets[mask]
        idx  = nms(d[:, :4], d[:, 4], iou_thres)
        keep_all.append(d[idx])

    out = torch.cat(keep_all, dim=0)
    # Sort by score descending and limit
    order = out[:, 4].argsort(descending=True)[:max_det]
    return out[order]


# ─────────────────────────────────────────────────────────────────────────────
# 4.  IoU helpers for mAP
# ─────────────────────────────────────────────────────────────────────────────

def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Vectorised IoU between two sets of xyxy boxes.
    boxes1: (M, 4)  boxes2: (N, 4)
    returns: (M, N)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    inter_x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    inter_y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    inter_x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    inter_y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.maximum(inter_x2 - inter_x1, 0)
    inter_h = np.maximum(inter_y2 - inter_y1, 0)
    inter_area = inter_w * inter_h

    union = area1[:, None] + area2[None, :] - inter_area + 1e-7
    return inter_area / union


# ─────────────────────────────────────────────────────────────────────────────
# 5.  mAP computation (PASCAL VOC 11-point + COCO 101-point)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """Compute AP using all-point interpolation (COCO-style)."""
    recall    = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    # Make precision monotonically decreasing
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])
    idx = np.where(recall[1:] != recall[:-1])[0]
    ap  = np.sum((recall[idx + 1] - recall[idx]) * precision[idx + 1])
    return float(ap)


def compute_map(
    all_preds: List[Dict],   # [{"boxes":(K,4), "scores":(K,), "labels":(K,)}, ...]
    all_gts:   List[Dict],   # [{"boxes":(G,4), "labels":(G,)}, ...]
    num_classes: int,
    iou_thresholds: np.ndarray,
) -> Tuple[Dict[str, float], Dict[int, float]]:
    """
    Compute mAP over a list of IoU thresholds.

    Returns
    -------
    metrics : dict with keys  mAP50, mAP50_95, precision, recall
    per_class_ap50 : {class_id: AP@50}
    """
    # Accumulate TP/FP per class and per IoU threshold
    # Shape: [num_classes, num_iou_thresh, num_detections_total]
    n_iou = len(iou_thresholds)

    # Per class: list of (score, tp[n_iou], n_gt)
    class_data: Dict[int, Dict] = {
        c: {"scores": [], "tps": [[] for _ in range(n_iou)], "n_gt": 0}
        for c in range(num_classes)
    }

    for preds, gts in zip(all_preds, all_gts):
        pred_boxes  = preds["boxes"]    # (K, 4)  numpy
        pred_scores = preds["scores"]   # (K,)
        # Ensure 1D labels (some pipelines store as (K,1))
        pred_labels = np.asarray(preds["labels"]).reshape(-1)   # (K,)

        gt_boxes    = gts["boxes"]      # (G, 4)
        # Ensure 1D labels (dataset collate may produce (G,1))
        gt_labels   = np.asarray(gts["labels"]).reshape(-1)     # (G,)

        for c in range(num_classes):
            gt_mask   = gt_labels == c
            pred_mask = pred_labels == c

            gt_c   = gt_boxes[gt_mask]    if gt_mask.any()   else np.zeros((0, 4))
            pred_c = pred_boxes[pred_mask] if pred_mask.any() else np.zeros((0, 4))
            sc_c   = pred_scores[pred_mask] if pred_mask.any() else np.zeros(0)

            class_data[c]["n_gt"] += len(gt_c)

            if len(pred_c) == 0:
                continue

            class_data[c]["scores"].extend(sc_c.tolist())

            if len(gt_c) == 0:
                for t in range(n_iou):
                    class_data[c]["tps"][t].extend([0] * len(pred_c))
                continue

            iou_mat = box_iou(pred_c, gt_c)   # (K_c, G_c)

            for t_idx, iou_thr in enumerate(iou_thresholds):
                matched_gt = set()
                tp_list    = []
                # Sort predictions by score (descending)
                order = np.argsort(-sc_c)
                for k in order:
                    best_iou  = -1.0
                    best_gt   = -1
                    for g in range(len(gt_c)):
                        if g in matched_gt:
                            continue
                        if iou_mat[k, g] > best_iou:
                            best_iou = iou_mat[k, g]
                            best_gt  = g
                    if best_iou >= iou_thr and best_gt not in matched_gt:
                        matched_gt.add(best_gt)
                        tp_list.append(1)
                    else:
                        tp_list.append(0)

                # Re-order back to original pred order
                tp_ordered = [0] * len(pred_c)
                for rank, k in enumerate(order):
                    tp_ordered[k] = tp_list[rank]
                class_data[c]["tps"][t_idx].extend(tp_ordered)

    # Compute per-class AP at each IoU threshold
    ap_matrix = np.zeros((num_classes, n_iou))   # [class, iou]

    per_class_ap50 = {}

    for c in range(num_classes):
        cd     = class_data[c]
        n_gt   = cd["n_gt"]
        scores = np.array(cd["scores"])

        if n_gt == 0 or len(scores) == 0:
            per_class_ap50[c] = float("nan")
            continue

        order = np.argsort(-scores)

        for t_idx in range(n_iou):
            tp_arr = np.array(cd["tps"][t_idx])[order]
            fp_arr = 1 - tp_arr

            tp_cum = np.cumsum(tp_arr)
            fp_cum = np.cumsum(fp_arr)

            recall    = tp_cum / (n_gt + 1e-7)
            precision = tp_cum / (tp_cum + fp_cum + 1e-7)

            ap_matrix[c, t_idx] = compute_ap(recall, precision)

        per_class_ap50[c] = float(ap_matrix[c, 0])  # IoU=0.50 is first

    # Aggregate
    valid_mask = ~np.isnan(ap_matrix)
    ap50_vals  = ap_matrix[:, 0]

    # mAP50
    valid_ap50 = ap50_vals[~np.isnan(ap50_vals)]
    mAP50 = float(valid_ap50.mean()) if len(valid_ap50) else 0.0

    # mAP50-95
    per_class_map5095 = np.nanmean(ap_matrix, axis=1)  # mean over IoU thresholds
    mAP5095 = float(np.nanmean(per_class_map5095))

    # Overall precision / recall at IoU=0.50, best-F1 threshold
    all_scores, all_tps = [], []
    total_gt = 0
    for c in range(num_classes):
        cd = class_data[c]
        total_gt += cd["n_gt"]
        if len(cd["scores"]) == 0:
            continue
        all_scores.extend(cd["scores"])
        all_tps.extend(cd["tps"][0])

    if all_scores:
        order      = np.argsort(-np.array(all_scores))
        tp_cum     = np.cumsum(np.array(all_tps)[order])
        fp_cum     = np.cumsum(1 - np.array(all_tps)[order])
        recall_v   = tp_cum / (total_gt + 1e-7)
        precision_v = tp_cum / (tp_cum + fp_cum + 1e-7)
        f1          = 2 * precision_v * recall_v / (precision_v + recall_v + 1e-7)
        best        = int(np.argmax(f1))
        best_prec   = float(precision_v[best])
        best_rec    = float(recall_v[best])
    else:
        best_prec = best_rec = 0.0

    metrics = {
        "mAP50":     mAP50,
        "mAP50_95":  mAP5095,
        "precision": best_prec,
        "recall":    best_rec,
    }
    return metrics, per_class_ap50


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    strides: torch.Tensor,
    num_classes: int,
    conf_thres: float,
    iou_nms: float,
    device: torch.device,
    class_names: List[str] = None,
    include_p2: bool = True,
    save_vis_dir: str = "",
    save_vis_n: int = 0,
) -> Dict:
    """Run evaluation and return metrics dict."""

    model.eval()
    strides = strides.to(device)

    iou_thresholds = np.linspace(0.5, 0.95, 10)   # COCO: 0.50:0.05:0.95

    all_preds: List[Dict] = []
    all_gts:   List[Dict] = []

    vis_dir = Path(save_vis_dir) if save_vis_dir else None
    vis_limit = int(save_vis_n) if save_vis_n else 0
    vis_saved = 0
    if vis_dir and vis_limit > 0:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(dataloader, desc="Evaluating"):
        # ── Unpack batch ─────────────────────────────────────────────────────
        # ultralytics_collate returns (images_tensor, targets_dict)
        if isinstance(batch, (list, tuple)):
            images  = batch[0].to(device)
            targets = batch[1]   # dict with 'batch_idx', 'cls', 'bboxes' (normalised xywh)
        else:
            images  = batch.to(device)
            targets = None

        B, C, H, W = images.shape

        # ── Forward pass ─────────────────────────────────────────────────────
        # Put model in training mode temporarily so the head returns the
        # Ultralytics-format list (reg+cls concatenated per scale).
        model.train()
        raw_output = model(images, return_features=True)
        model.eval()

        raw_preds: List[torch.Tensor] = raw_output["predictions_ultralytics"]

        # Drop P2 if the loss wrapper expects only [P3,P4,P5]
        # (keep consistent with however training was done)
        scales_to_use = raw_preds  # use all scales for evaluation

        # ── Decode ───────────────────────────────────────────────────────────
        decoded = decode_predictions(
            scales_to_use,
            strides,
            num_classes,
            conf_thres=conf_thres,
        )

        # ── NMS ──────────────────────────────────────────────────────────────
        batch_dets: List[torch.Tensor] = []
        for b in range(B):
            dets = nms_per_class(decoded[b], iou_thres=iou_nms)
            batch_dets.append(dets)

            if dets.shape[0] > 0:
                pred_entry = {
                    "boxes":  dets[:, :4].cpu().numpy(),
                    "scores": dets[:,  4].cpu().numpy(),
                    "labels": dets[:,  5].cpu().numpy().astype(int),
                }
            else:
                pred_entry = {
                    "boxes":  np.zeros((0, 4)),
                    "scores": np.zeros(0),
                    "labels": np.zeros(0, dtype=int),
                }
            all_preds.append(pred_entry)

        # ── Ground truth ─────────────────────────────────────────────────────
        if targets is not None:
            batch_idx = targets["batch_idx"]    # (total_gt,)
            # Collate may return cls as (N,1); flatten to (N,)
            cls_ids   = targets["cls"].view(-1)  # (total_gt,)  0-indexed
            bboxes    = targets["bboxes"]        # (total_gt, 4)  normalised xywh

            # Convert normalised xywh → pixel xyxy
            bboxes_xyxy = bboxes.clone()
            bboxes_xyxy[:, 0] = (bboxes[:, 0] - bboxes[:, 2] / 2) * W
            bboxes_xyxy[:, 1] = (bboxes[:, 1] - bboxes[:, 3] / 2) * H
            bboxes_xyxy[:, 2] = (bboxes[:, 0] + bboxes[:, 2] / 2) * W
            bboxes_xyxy[:, 3] = (bboxes[:, 1] + bboxes[:, 3] / 2) * H

            for b in range(B):
                mask = batch_idx == b
                gt_boxes_b = bboxes_xyxy[mask].cpu().numpy()
                gt_labels_b = cls_ids[mask].cpu().numpy().astype(int)

                all_gts.append({"boxes": gt_boxes_b, "labels": gt_labels_b})

                # ── Optional visualization export (pred + GT) ───────────────
                if vis_dir and vis_saved < vis_limit:
                    img_t = images[b].detach().cpu()  # (C,H,W)
                    # dataset provides 1-channel thermal in [0,1]
                    img_np = (img_t.squeeze(0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)  # (H,W)
                    pil = Image.fromarray(img_np, mode="L").convert("RGB")
                    draw = ImageDraw.Draw(pil)

                    # GT boxes in green
                    for (x1, y1, x2, y2), cls in zip(gt_boxes_b, gt_labels_b):
                        name = class_names[int(cls)] if class_names and int(cls) < len(class_names) else str(int(cls))
                        draw.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=(0, 255, 0), width=2)
                        draw.text((float(x1), max(0.0, float(y1) - 10.0)), f"GT:{name}", fill=(0, 255, 0))

                    # Pred boxes in red
                    dets_b = batch_dets[b].detach().cpu()
                    if dets_b.numel():
                        for x1, y1, x2, y2, score, cls in dets_b.numpy():
                            cls_i = int(cls)
                            name = class_names[cls_i] if class_names and cls_i < len(class_names) else str(cls_i)
                            draw.rectangle([float(x1), float(y1), float(x2), float(y2)], outline=(255, 0, 0), width=2)
                            draw.text((float(x1), float(y1)), f"P:{name} {float(score):.2f}", fill=(255, 0, 0))

                    out_path = vis_dir / f"val_{vis_saved:03d}.jpg"
                    pil.save(out_path, quality=95)
                    vis_saved += 1
        else:
            # No targets provided — fill with empty GTs (metrics will be 0)
            for _ in range(B):
                all_gts.append({"boxes": np.zeros((0, 4)), "labels": np.zeros(0, dtype=int)})

    # ── Compute mAP ──────────────────────────────────────────────────────────
    metrics, per_class_ap50 = compute_map(
        all_preds, all_gts, num_classes, iou_thresholds
    )
    metrics["per_class_AP50"] = per_class_ap50

    # ── Pretty-print ─────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("  Evaluation Results")
    print("=" * 50)
    print(f"  mAP@50        : {metrics['mAP50']:.4f}")
    print(f"  mAP@50:95     : {metrics['mAP50_95']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print("-" * 50)
    print("  Per-class AP@50:")
    for c, ap in per_class_ap50.items():
        name = class_names[c] if class_names and c < len(class_names) else f"class_{c}"
        ap_str = f"{ap:.4f}" if not np.isnan(ap) else "  N/A "
        print(f"    [{c}] {name:20s}: {ap_str}")
    print("=" * 50)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Evaluate YOLOv8Thermal from distillation checkpoint."
    )
    ap.add_argument("--checkpoint",   required=True,
                    help="Path to .pth checkpoint (from DistillationTrainer).")

    # Dataset
    ap.add_argument("--flir-root",  default="FLIR_ADAS_v2")
    ap.add_argument("--split",      default="images_thermal_val",
                    help="Sub-folder under flir-root containing images.")
    ap.add_argument("--coco-json",  default="coco.json",
                    help="COCO annotation file inside --split folder.")

    # Model
    ap.add_argument("--num-classes", type=int, default=3,
                    help="Number of detection classes (must match training).")
    ap.add_argument("--include-p2", action="store_true", default=True,
                    help="Include P2 head (must match training).")
    ap.add_argument("--no-transformer", action="store_true", default=False,
                    help="Disable transformer neck (must match training).")
    ap.add_argument("--class-names", nargs="+",
                    default=["person", "car", "bicycle"],
                    help="Human-readable class names in class-id order.")

    # Inference
    ap.add_argument("--imgsz",   type=int,   default=512)
    ap.add_argument("--batch",   type=int,   default=8)
    ap.add_argument("--workers", type=int,   default=2)
    ap.add_argument("--conf",    type=float, default=0.25,
                    help="Confidence threshold for detections.")
    ap.add_argument("--iou-nms", type=float, default=0.45,
                    help="IoU threshold for NMS.")
    ap.add_argument("--device",  default="cuda")

    # Output
    ap.add_argument("--save-json", default="/home/aryan_s2/Detection_with_Distillation/Student/results/eval_metrics.json",
                    help="Optional path to save metrics as JSON.")
    ap.add_argument("--save-vis-dir", default="/home/aryan_s2/Detection_with_Distillation/Student/results/visualizations",
                    help="Optional output folder to save sample images with GT+pred boxes.")
    ap.add_argument("--save-vis-n", type=int, default=5,
                    help="How many images to export to --save-vis-dir (0 disables).")
    return ap.parse_args()


def main():
    args = parse_args()

    # ── Load model ───────────────────────────────────────────────────────────
    model, device = load_student_from_checkpoint(
        checkpoint_path=args.checkpoint,
        num_classes=args.num_classes,
        include_p2=args.include_p2,
        use_transformer_neck=not args.no_transformer,
        device=args.device,
    )

    # ── Build dataset & dataloader ───────────────────────────────────────────
    flir_root = Path(args.flir_root)
    split_dir = flir_root / args.split
    coco_json = split_dir / args.coco_json

    if not coco_json.exists():
        raise FileNotFoundError(
            f"COCO annotation not found: {coco_json}\n"
            f"Check --flir-root / --split / --coco-json arguments."
        )

    dataset = FlirThermalCocoDataset(
        paths=FlirCocoPaths(images_dir=split_dir, coco_json=coco_json),
        imgsz=args.imgsz,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=ultralytics_collate,
    )

    print(f"Dataset : {len(dataset)} images  |  batch={args.batch}")
    print(f"Device  : {device}")
    print(f"Conf    : {args.conf}   IoU-NMS : {args.iou_nms}")

    # ── Strides — must match include_p2 setting ──────────────────────────────
    strides = model.stride  # already a tensor from YOLOv8Thermal.__init__

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate(
        model=model,
        dataloader=dataloader,
        strides=strides,
        num_classes=args.num_classes,
        conf_thres=args.conf,
        iou_nms=args.iou_nms,
        device=device,
        class_names=args.class_names,
        include_p2=args.include_p2,
        save_vis_dir=args.save_vis_dir,
        save_vis_n=args.save_vis_n,
    )

    # ── Optionally save results ───────────────────────────────────────────────
    if args.save_json:
        out = {k: (v if not isinstance(v, dict) else
                   {str(kk): (float(vv) if not np.isnan(vv) else None)
                    for kk, vv in v.items()})
               for k, v in metrics.items()}
        Path(args.save_json).write_text(json.dumps(out, indent=2))
        print(f"\nMetrics saved → {args.save_json}")


if __name__ == "__main__":
    main()
