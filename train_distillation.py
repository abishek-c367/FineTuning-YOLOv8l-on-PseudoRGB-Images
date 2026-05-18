from __future__ import annotations
from pickle import FALSE
import argparse
from pathlib import Path
from typing import Dict
from ultralytics import YOLO


import torch
from torch.utils.data import DataLoader

from Distillation import DistillationTrainer, create_yolo_loss, train_distillation
from Cycle_gan.cyclegan_loader import load_cyclegan_generator
from data.flir_dataset import FlirCocoPaths, FlirThermalCocoDataset, ultralytics_collate
from Teacher.teacher_yolov8_feature_extractor import TeacherYOLOv8FeatureExtractor
from Student.yolov8_thermal import yolov8s_thermal
# import sys
# import os


import config as cfg


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Teacher-student distillation: YOLOv8-L teacher on pseudo-RGB, thermal student.")

    ap.add_argument("--cyclegan-weights", type=str, default=cfg.cyclegan_weights, help="Path to cyclegan.pth (IR->RGB generator).")
    ap.add_argument("--teacher-weights", type=str, default=cfg.teacher_weights, help="Ultralytics teacher weights (e.g., yolov8l.pt).")

    ap.add_argument("--flir-root", type=str, default=cfg.flir_root, help="Path to FLIR_ADAS_v2 root (thermal split folders).")
    ap.add_argument("--split", type=str, default=cfg.split, help="Split folder name under FLIR root.")
    ap.add_argument("--coco-json", type=str, default=cfg.coco_json, help="COCO json filename inside split folder.")

    ap.add_argument("--imgsz", type=int, default=cfg.imgsz)
    ap.add_argument("--batch", type=int, default=cfg.batch)
    ap.add_argument("--workers", type=int, default=cfg.workers)
    ap.add_argument("--epochs", type=int, default=cfg.epochs)
    ap.add_argument("--alpha", type=float, default=cfg.alpha, help="Feature mimic loss weight.")

    ap.add_argument("--num-classes", type=int, default=cfg.num_classes, help="Number of detection classes (person/car/bicycle => 3).")
    ap.add_argument("--device", type=str, default=cfg.device, help="cuda or cpu")
    ap.add_argument("--checkpoint-dir", type=str, default=cfg.checkpoint_dir)
    ap.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    ap.add_argument("--pseudo-rgb-cache", type=str, default=cfg.pseudo_rgb_cache, help="Path to pseudo-RGB cache directory")

    return ap.parse_args()

import torch

def infer_yolov8_pyramid_channels(student, imgsz=512, device=None):
    if device is None:
        device = torch.device('cpu')
    elif isinstance(device, str):
        device = torch.device(device)

    model = student.model  # YOLO wrapper → DetectionModel
    layers = model.model

    features = []

    # Hook to collect all feature maps
    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor) and output.dim() == 4:
            features.append(output)

    hooks = []
    for layer in layers:
        hooks.append(layer.register_forward_hook(hook_fn))

    # Dummy forward
    dummy = torch.randn(1, 3, imgsz, imgsz, device=device)
    with torch.no_grad():
        model(dummy)

    # Remove hooks
    for h in hooks:
        h.remove()

    # --------------------------------------------------
    # Select pyramid features based on spatial size
    # --------------------------------------------------
    # Keep only unique spatial resolutions
    unique_feats = {}
    for f in features:
        h, w = f.shape[2:]
        unique_feats[(h, w)] = f  # overwrite duplicates

    # Sort by resolution (largest → smallest)
    sorted_feats = sorted(unique_feats.values(), key=lambda x: x.shape[2], reverse=True)

    # Take top 3 pyramid levels
    pyramid_feats = sorted_feats[:3]

    # Assign P3, P4, P5
    pyramid_feats = sorted(pyramid_feats, key=lambda x: x.shape[2], reverse=True)

    names = ["P3", "P4", "P5"]
    channels = {name: feat.shape[1] for name, feat in zip(names, pyramid_feats)}

    return channels


def main() -> None:

    args = _parse_args()
    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')

    student = yolov8s_thermal(num_classes=args.num_classes, include_p2=True, use_transformer_neck=FALSE).to(device)

    # Teacher (Ultralytics YOLOv8-L)
    teacher = TeacherYOLOv8FeatureExtractor(weights=args.teacher_weights, device=device)

    # CycleGAN generator (thermal -> pseudoRGB)
    cyclegan = load_cyclegan_generator(args.cyclegan_weights, device=device, input_nc=3, output_nc=3)

    # Dataset + loader (COCO -> Ultralytics loss targets)
    flir_root = Path(args.flir_root)
    split_dir = flir_root / args.split
    coco_json = split_dir / args.coco_json
    ds = FlirThermalCocoDataset(
        paths=FlirCocoPaths(images_dir=split_dir, coco_json=coco_json),
        imgsz=args.imgsz,
    )
    val_split_dir = flir_root / "images_thermal_val"   # FLIR standard val folder
    val_coco_json = val_split_dir / args.coco_json     # same filename: coco.json

    val_ds = FlirThermalCocoDataset(
        paths=FlirCocoPaths(images_dir=val_split_dir, coco_json=val_coco_json),
        imgsz=args.imgsz,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=ultralytics_collate,
    )
    
    val_dl = DataLoader(
        val_ds,
        batch_size=args.batch,
        shuffle=False,             # no shuffle for validation
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        collate_fn=ultralytics_collate,
    )



    # In train_distillation.py main(), after creating yolo_loss_fn:
    yolo_loss_fn = create_yolo_loss(num_classes=args.num_classes, model=student, require_ultralytics=True)

    # Move loss internals to the same device as the student model.
    yolo_loss_fn = yolo_loss_fn.to(device)

    # Also fix proj specifically (sometimes .to() misses it)
    if hasattr(yolo_loss_fn, 'loss_fn'):
        if hasattr(yolo_loss_fn.loss_fn, 'proj'):
            yolo_loss_fn.loss_fn.proj = yolo_loss_fn.loss_fn.proj.to(device)
        if hasattr(yolo_loss_fn.loss_fn, 'assigner'):
            for name, buf in yolo_loss_fn.loss_fn.assigner.named_buffers():
                setattr(yolo_loss_fn.loss_fn.assigner, name, buf.to(device))
            for name, param in yolo_loss_fn.loss_fn.assigner.named_parameters():
                setattr(yolo_loss_fn.loss_fn.assigner, name, param.to(device))







    # Channel maps for mimic loss
    student_channels: Dict[str, int] = {k: int(v) for k, v in student.neck.out_channels.items() if k in ("P3", "P4", "P5")}
    # student_channels = infer_yolov8_pyramid_channels(
    #     student,
    #     imgsz=args.imgsz,
    #     device=args.device
    # )
    print(student_channels.keys())

    teacher_channels = teacher.infer_channels((args.imgsz, args.imgsz))
    print("Teacher channels:",teacher_channels.keys())


    trainer = DistillationTrainer(
        student=student,
        teacher=teacher,
        cyclegan_generator=cyclegan,
        yolo_loss_fn=yolo_loss_fn,
        student_channels=student_channels,
        teacher_channels=teacher_channels,
        distill_layers=["P3", "P4"],
        alpha=float(args.alpha),
        device=device,
        lr=1e-3,
        weight_decay=5e-4,
        pseudo_rgb_cache_dir=args.pseudo_rgb_cache
    )

    # Resume from checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        start_epoch = trainer.load_checkpoint(args.checkpoint)
        print(f"Resumed from epoch {start_epoch}")
 
    train_distillation(
        trainer=trainer,
        train_dataloader=dl,
        val_dataloader=val_dl,
        epochs=int(args.epochs),
        start_epoch=start_epoch,
        log_interval=1000,
        save_interval=5,
        checkpoint_dir=str(args.checkpoint_dir),
        num_classes=args.num_classes,
        imgsz=args.imgsz,
        validation_interval=1000,
    )

if __name__ == "__main__":
    main()

