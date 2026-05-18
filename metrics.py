"""
metrics.py
==========
Computes mAP50, mAP75, Precision, Recall for the thermal student model
during distillation training.

Designed to slot cleanly into DistillationTrainer with zero changes to
the rest of the codebase.

Ground-truth format (from FlirThermalCocoDataset + ultralytics_collate):
    targets["bboxes"]    : (M, 4)  normalized xywh  in [0,1]
    targets["cls"]       : (M, 1)  float class index
    targets["batch_idx"] : (M,)    int64 image index inside batch

Prediction format (from student via Ultralytics decode):
    List of 3 tensors at P3/P4/P5 scales, each (B, 4+nc+reg, H, W)
    → decoded with Ultralytics non_max_suppression into
      List[Tensor(N_i, 6)]  where each row is [x1,y1,x2,y2,conf,cls]
      in absolute pixel coords of the letterboxed imgsz image.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Box format helpers
# ─────────────────────────────────────────────────────────────────────────────

def xywh_to_xyxy(boxes: torch.Tensor, imgsz: int) -> torch.Tensor:
    """
    Convert normalized xywh → absolute xyxy.
    boxes: (N, 4)  values in [0,1]
    returns: (N, 4)  absolute pixel coords
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = (cx - w / 2) * imgsz
    y1 = (cy - h / 2) * imgsz
    x2 = (cx + w / 2) * imgsz
    y2 = (cy + h / 2) * imgsz
    return torch.stack([x1, y1, x2, y2], dim=1)


def box_iou(boxes_a: torch.Tensor, boxes_b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise IoU between two sets of xyxy boxes.
    boxes_a: (N, 4)
    boxes_b: (M, 4)
    returns: (N, M)
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])  # (N,)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])  # (M,)

    inter_x1 = torch.max(boxes_a[:, 0].unsqueeze(1), boxes_b[:, 0].unsqueeze(0))  # (N,M)
    inter_y1 = torch.max(boxes_a[:, 1].unsqueeze(1), boxes_b[:, 1].unsqueeze(0))
    inter_x2 = torch.min(boxes_a[:, 2].unsqueeze(1), boxes_b[:, 2].unsqueeze(0))
    inter_y2 = torch.min(boxes_a[:, 3].unsqueeze(1), boxes_b[:, 3].unsqueeze(0))

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter   = inter_w * inter_h                                                    # (N,M)

    union = area_a.unsqueeze(1) + area_b.unsqueeze(0) - inter
    return inter / union.clamp(min=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# Decode student raw predictions → list of (N,6) detection tensors
# ─────────────────────────────────────────────────────────────────────────────

def decode_student_predictions(
    student_predictions: List[torch.Tensor],
    student_model,
    imgsz: int,
    conf_threshold: float = 0.001,
    iou_threshold:  float = 0.6,
) -> List[torch.Tensor]:
    """
    Decode raw Ultralytics-format student predictions into detections.

    student_predictions: List[Tensor]  3 FPN tensors from student
    student_model: the YOLOv8Thermal student (needs .stride, .nc, .reg_max)
    imgsz: image size (square)

    Returns:
        List of length B, each element Tensor(N_i, 6)
        columns: [x1, y1, x2, y2, confidence, class_id]  — absolute pixel coords
    """
    try:
        from ultralytics.utils.tal import make_anchors, dist2bbox
    except ImportError:
        raise ImportError(
            "Ultralytics is required for decoding predictions.\n"
            "pip install ultralytics"
        )

    device = student_predictions[0].device
    nc     = student_model.nc
    reg_max = student_model.reg_max  # usually 16

    # ── Build anchor grid from stride ────────────────────────────────────────
    strides = student_model.stride.to(device)   # [8, 16, 32]
    if len(student_predictions) != len(strides):
        raise ValueError(
            f"Mismatch: {len(student_predictions)} feature maps but {len(strides)} strides"
        )

    # Compute feature map spatial sizes for each FPN level
    shapes = [p.shape[2:] for p in student_predictions]  # [(H3,W3), (H4,W4), (H5,W5)]

    anchors, anchor_strides = make_anchors(student_predictions, strides, grid_cell_offset=0.5)
    # anchors:        (total_anchors, 2)  in feature-map pixel space
    # anchor_strides: (total_anchors, 1)

    # ── Concatenate all scale outputs ────────────────────────────────────────
    # Each prediction tensor: (B, reg_max*4 + nc, H, W)
    pred_cat = torch.cat(
        [p.view(p.shape[0], -1, p.shape[2] * p.shape[3]) for p in student_predictions],
        dim=2
    )  # (B, reg_max*4 + nc, total_anchors)
    pred_cat = pred_cat.permute(0, 2, 1)  # (B, total_anchors, reg_max*4 + nc)

    # Split into box distribution and class logits
    box_dist = pred_cat[..., : reg_max * 4]   # (B, A, reg_max*4)
    cls_logits = pred_cat[..., reg_max * 4:]  # (B, A, nc)

    # ── DFL decode: distribution → ltrb offsets ──────────────────────────────
    # Reshape for softmax over reg_max bins
    B, A, _ = box_dist.shape
    box_dist = box_dist.view(B, A, 4, reg_max)                  # (B, A, 4, reg_max)
    box_dist = box_dist.softmax(dim=-1)                          # (B, A, 4, reg_max)
    bins = torch.arange(reg_max, dtype=torch.float32, device=device)
    ltrb = (box_dist * bins).sum(dim=-1)                         # (B, A, 4)

    # ── ltrb → xyxy using anchor centres ─────────────────────────────────────
    # dist2bbox expects (B, 4, A) and anchors (A, 2)
    ltrb_t = ltrb.permute(0, 2, 1)                               # (B, 4, A)
    xy_centres = anchors.unsqueeze(0).expand(B, -1, -1)          # (B, A, 2)

    # Manual dist2bbox: xyxy = [cx-l, cy-t, cx+r, cy+b] * stride
    x1 = (xy_centres[..., 0] - ltrb[..., 0]) * anchor_strides.squeeze(-1)
    y1 = (xy_centres[..., 1] - ltrb[..., 1]) * anchor_strides.squeeze(-1)
    x2 = (xy_centres[..., 0] + ltrb[..., 2]) * anchor_strides.squeeze(-1)
    y2 = (xy_centres[..., 1] + ltrb[..., 3]) * anchor_strides.squeeze(-1)
    boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)           # (B, A, 4) absolute pixels

    # ── Class scores ─────────────────────────────────────────────────────────
    cls_scores = cls_logits.sigmoid()                             # (B, A, nc)
    conf, cls_ids = cls_scores.max(dim=-1)                        # (B, A)

    # ── Per-image NMS ─────────────────────────────────────────────────────────
    all_dets = torch.cat(
        [boxes_xyxy, conf.unsqueeze(-1), cls_ids.float().unsqueeze(-1)],
        dim=-1
    )  # (B, A, 6)

    results = []
    for i in range(B):
        det = all_dets[i]                                         # (A, 6)
        mask = det[:, 4] >= conf_threshold
        det  = det[mask]

        if det.shape[0] == 0:
            results.append(torch.zeros((0, 6), device=device))
            continue

        # Clamp boxes to image bounds
        det[:, 0].clamp_(0, imgsz)
        det[:, 1].clamp_(0, imgsz)
        det[:, 2].clamp_(0, imgsz)
        det[:, 3].clamp_(0, imgsz)

        # Per-class NMS
        keep_boxes = []
        for cls_id in det[:, 5].unique():
            cls_mask = det[:, 5] == cls_id
            cls_det  = det[cls_mask]
            # Sort by confidence descending
            order = cls_det[:, 4].argsort(descending=True)
            cls_det = cls_det[order]

            kept = []
            while cls_det.shape[0] > 0:
                kept.append(cls_det[0])
                if cls_det.shape[0] == 1:
                    break
                iou = box_iou(cls_det[0:1, :4], cls_det[1:, :4]).squeeze(0)
                cls_det = cls_det[1:][iou < iou_threshold]

            keep_boxes.append(torch.stack(kept))

        if keep_boxes:
            results.append(torch.cat(keep_boxes, dim=0))
        else:
            results.append(torch.zeros((0, 6), device=device))

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Per-class AP computation (11-point interpolation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ap(recalls: torch.Tensor, precisions: torch.Tensor) -> float:
    """
    Compute AP using 101-point COCO interpolation.
    recalls, precisions: 1-D tensors sorted by recall ascending.
    """
    r = torch.cat([torch.tensor([0.0]), recalls, torch.tensor([1.0])])
    p = torch.cat([torch.tensor([0.0]), precisions, torch.tensor([0.0])])

    # Make precision monotonically decreasing
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])

    ap = 0.0
    thresholds = torch.linspace(0, 1, 101)
    for t in thresholds:
        mask = r >= t
        if mask.any():
            ap += p[mask].max().item()
    return ap / 101.0


def compute_ap_per_class(
    predictions: List[torch.Tensor],   # List[Tensor(N_i,6)]:  x1y1x2y2, conf, cls
    gt_boxes:    List[torch.Tensor],   # List[Tensor(M_i,4)]:  xyxy absolute
    gt_classes:  List[torch.Tensor],   # List[Tensor(M_i,)]    int class ids
    num_classes: int,
    iou_threshold: float = 0.5,
) -> Dict[int, float]:
    """
    Compute per-class AP at a single IoU threshold.
    Returns dict: {class_id: AP_value}
    """
    # Collect all detections and GTs with image index
    all_dets = []   # (img_idx, x1,y1,x2,y2, conf, cls)
    all_gts  = {}   # img_idx -> {cls_id -> list of boxes, matched flags}

    for img_idx, (dets, gts, gt_cls) in enumerate(zip(predictions, gt_boxes, gt_classes)):
        for d in dets:
            all_dets.append((img_idx, d))
        for box, cls in zip(gts, gt_cls):
            cls = int(cls.item())
            all_gts.setdefault(img_idx, {}).setdefault(cls, {"boxes": [], "matched": []})
            all_gts[img_idx][cls]["boxes"].append(box)
            all_gts[img_idx][cls]["matched"].append(False)

    ap_per_class = {}

    for cls_id in range(num_classes):
        # Filter detections for this class
        cls_dets = [(img_i, d) for img_i, d in all_dets if int(d[5].item()) == cls_id]

        # Count total GTs for this class
        n_gt = sum(
            len(v[cls_id]["boxes"])
            for v in all_gts.values()
            if cls_id in v
        )

        if n_gt == 0:
            ap_per_class[cls_id] = 0.0
            continue

        if len(cls_dets) == 0:
            ap_per_class[cls_id] = 0.0
            continue

        # Sort by confidence descending
        cls_dets.sort(key=lambda x: x[1][4].item(), reverse=True)

        tp = torch.zeros(len(cls_dets))
        fp = torch.zeros(len(cls_dets))

        for det_i, (img_idx, det) in enumerate(cls_dets):
            pred_box = det[:4]

            gt_info = all_gts.get(img_idx, {}).get(cls_id, None)
            if gt_info is None or len(gt_info["boxes"]) == 0:
                fp[det_i] = 1
                continue

            gt_tensor = torch.stack(gt_info["boxes"])           # (G, 4)
            ious = box_iou(pred_box.unsqueeze(0), gt_tensor)    # (1, G)
            best_iou, best_gt = ious[0].max(dim=0)

            if best_iou.item() >= iou_threshold and not gt_info["matched"][best_gt.item()]:
                tp[det_i] = 1
                gt_info["matched"][best_gt.item()] = True
            else:
                fp[det_i] = 1

        cum_tp = tp.cumsum(0)
        cum_fp = fp.cumsum(0)
        recalls    = cum_tp / (n_gt + 1e-7)
        precisions = cum_tp / (cum_tp + cum_fp + 1e-7)

        ap_per_class[cls_id] = _compute_ap(recalls, precisions)

    return ap_per_class


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation function — called from train_distillation loop
# ─────────────────────────────────────────────────────────────────────────────

class DetectionMetricsAccumulator:
    """
    Accumulates predictions and ground truths across batches,
    then computes mAP50, mAP75, Precision, Recall in one call.

    Usage (inside train_distillation loop):

        accumulator = DetectionMetricsAccumulator(num_classes=3, imgsz=512)

        # inside batch loop:
        accumulator.update(student_predictions, targets, student_model)

        # every 100 steps:
        metrics = accumulator.compute()
        accumulator.reset()
    """

    def __init__(self, num_classes: int, imgsz: int, conf_threshold: float = 0.001):
        self.num_classes    = num_classes
        self.imgsz          = imgsz
        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self):
        self._all_preds:      List[torch.Tensor] = []   # List of (N,6) per image
        self._all_gt_boxes:   List[torch.Tensor] = []   # List of (M,4) xyxy per image
        self._all_gt_classes: List[torch.Tensor] = []   # List of (M,)  per image

    def update(
        self,
        student_predictions: List[torch.Tensor],
        targets: Dict[str, torch.Tensor],
        student_model,
    ):
        """
        Process one batch.

        student_predictions : raw output from student (3 FPN tensors)
        targets             : ultralytics_collate format dict
        student_model       : the YOLOv8Thermal student instance
        """
        with torch.no_grad():
            # Decode predictions → List[Tensor(N_i,6)] absolute xyxy
            try:
                decoded = decode_student_predictions(
                    student_predictions,
                    student_model,
                    self.imgsz,
                    conf_threshold=self.conf_threshold,
                )
            except Exception as e:
                print(f"  [metrics] Decode failed: {e}")
                return

            # Parse ground truths from ultralytics_collate format
            batch_idx  = targets["batch_idx"]                   # (M,)
            gt_cls_all = targets["cls"].squeeze(-1)             # (M,)  float
            gt_box_all = targets["bboxes"]                      # (M,4) norm xywh

            # Convert GT boxes: norm xywh → absolute xyxy
            gt_xyxy_all = xywh_to_xyxy(gt_box_all, self.imgsz) # (M,4)

            batch_size = len(decoded)
            for i in range(batch_size):
                mask = batch_idx == i
                gt_boxes_i   = gt_xyxy_all[mask]                # (M_i, 4)
                gt_classes_i = gt_cls_all[mask].long()          # (M_i,)

                self._all_preds.append(decoded[i].cpu())
                self._all_gt_boxes.append(gt_boxes_i.cpu())
                self._all_gt_classes.append(gt_classes_i.cpu())

    def compute(self) -> Dict[str, float]:
        """
        Compute and return mAP50, mAP75, mean Precision, mean Recall.
        Returns empty dict if no data has been accumulated.
        """
        if not self._all_preds:
            return {}

        # ── AP at IoU=0.50 ───────────────────────────────────────────────────
        ap50_per_class = compute_ap_per_class(
            self._all_preds,
            self._all_gt_boxes,
            self._all_gt_classes,
            self.num_classes,
            iou_threshold=0.5,
        )

        # ── AP at IoU=0.75 ───────────────────────────────────────────────────
        ap75_per_class = compute_ap_per_class(
            self._all_preds,
            self._all_gt_boxes,
            self._all_gt_classes,
            self.num_classes,
            iou_threshold=0.75,
        )

        mAP50 = sum(ap50_per_class.values()) / max(len(ap50_per_class), 1)
        mAP75 = sum(ap75_per_class.values()) / max(len(ap75_per_class), 1)

        # ── Global Precision & Recall at IoU=0.50, conf=0.25 ─────────────────
        total_tp = total_fp = total_fn = 0

        for img_idx in range(len(self._all_preds)):
            preds    = self._all_preds[img_idx]          # (N,6)
            gt_boxes = self._all_gt_boxes[img_idx]       # (M,4)
            gt_cls   = self._all_gt_classes[img_idx]     # (M,)

            # Filter preds by conf>=0.25 for P/R computation
            if preds.shape[0] > 0:
                preds = preds[preds[:, 4] >= 0.25]

            n_gt = gt_boxes.shape[0]

            if preds.shape[0] == 0 and n_gt == 0:
                continue
            if preds.shape[0] == 0:
                total_fn += n_gt
                continue
            if n_gt == 0:
                total_fp += preds.shape[0]
                continue

            iou_matrix = box_iou(preds[:, :4], gt_boxes)   # (N, M)
            matched_gt = set()

            for pred_i in range(preds.shape[0]):
                best_iou, best_gt = iou_matrix[pred_i].max(dim=0)
                pred_cls = int(preds[pred_i, 5].item())
                gt_cls_j = int(gt_cls[best_gt.item()].item()) if gt_boxes.shape[0] > 0 else -1

                if (best_iou.item() >= 0.5
                        and best_gt.item() not in matched_gt
                        and pred_cls == gt_cls_j):
                    total_tp += 1
                    matched_gt.add(best_gt.item())
                else:
                    total_fp += 1

            total_fn += n_gt - len(matched_gt)

        precision = total_tp / max(total_tp + total_fp, 1)
        recall    = total_tp / max(total_tp + total_fn, 1)

        return {
            "mAP50":     round(mAP50, 4),
            "mAP75":     round(mAP75, 4),
            "Precision": round(precision, 4),
            "Recall":    round(recall, 4),
            # Per-class breakdown
            **{f"AP50_cls{c}": round(v, 4) for c, v in ap50_per_class.items()},
        }


def print_metrics(metrics: Dict[str, float], step: int, epoch: int):
    """Pretty-print metrics to console."""
    if not metrics:
        print(f"  [Epoch {epoch} | Step {step}] No predictions accumulated yet.")
        return

    sep = "─" * 52
    print(f"\n{sep}")
    print(f"  📊  Metrics  │  Epoch {epoch}  │  Step {step}")
    print(sep)
    print(f"  mAP@50   : {metrics['mAP50']:.4f}")
    print(f"  mAP@75   : {metrics['mAP75']:.4f}")
    print(f"  Precision: {metrics['Precision']:.4f}")
    print(f"  Recall   : {metrics['Recall']:.4f}")

    # Per-class APs
    cls_keys = [k for k in metrics if k.startswith("AP50_cls")]
    if cls_keys:
        print(f"  {'─'*30}")
        class_names = ["bicycle", "car", "person"]   # sorted alphabetically = FLIR class order
        for k in sorted(cls_keys):
            cls_id = int(k.replace("AP50_cls", ""))
            name   = class_names[cls_id] if cls_id < len(class_names) else f"cls{cls_id}"
            print(f"  AP50 {name:<10}: {metrics[k]:.4f}")

    print(f"{sep}\n")
