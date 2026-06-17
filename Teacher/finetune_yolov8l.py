#import necessary libraries
import os
import sys
import shutil
import random
import logging
import argparse
from pathlib import Path
import torch
import yaml
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO

#Path and config

DATASET_ROOT = ''
MODEL_WEIGHTS = ''
OUTPUT_DIR = ''

CLASS_NAMES = ['person', 'car', 'bicycle'] 

#Training hyperparameters

EPOCHS = 100
IMG_SIZE = 512
BATCH_SIZE = 16
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005
WARMUP_EPOCHS = 3
PATIENCE = 20
WORKERS = 4

#FREEZE teacher backbone layers and only train the neck and head

FREEZE_LAYERS = 10

#AUTO TRAIN AND VAL SPLIT(ONLY IF VAL/SUB-DIR DOESN'T EXIST)

AUTO_SPLIT = True
VAL_RATIO = 0.15

#DATA AUGMENTATION FLAGS

AUGMENT_CFG = dict(
    hsv_h       = 0.015,   # hue jitter — keep low for thermal-derived images
    hsv_s       = 0.5,     # saturation
    hsv_v       = 0.4,
    degrees     = 10.0,    # rotation
    translate   = 0.1,
    scale       = 0.5,
    shear       = 2.0,
    perspective = 0.0,
    flipud      = 0.0,
    fliplr      = 0.5,
    mosaic      = 1.0,
    mixup       = 0.1,
    copy_paste  = 0.0,
)

#LOGGING SETUP

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt= "%H:%M:%S"
)
log = logging.getLogger(__name__)

#HELPER UTILITIES
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif' , '.WEBP'}

#to search through a folder (and all of its subfolders) 
# to find any image files and return them as a sorted list.

def collect_image_paths(directory:Path) -> list[Path]:
   #recursively collect image paths from a directory
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


#to generate the corresponding label file path for a given image path,
def label_path_for(img_path:Path,labels_root:Path) -> Path:
    return labels_root / img_path.with_suffix(".txt").name

#to check that each image has a corresponding label file, and to count how many pairs are valid vs. missing.
def verify_dataset(images_dir:Path,labels_dir:Path) -> tuple[int,int,list[str]]:
    """Check image/label pairing and return stats."""
    imgs = collect_image_paths(images_dir)
    ok = 0
    missing = 0
    warns = []
    for img in imgs:
        lbl = labels_dir / img.with_suffix(".txt").name
        if lbl.exists():
            ok += 1
        else:
            missing += 1
            if missing <= 10:
                warns.append(f"Missing label: {lbl}")
    if missing > 10:
        warns.append(f"... and {missing - 10} more missing label files.")
    return ok, missing, warns

"""
    Split a flat images+labels directory into train/val sub-directories
    inside *dst_root* (which may equal the parent of src_images).
    """
def auto_split(
    src_images : Path,
    src_labels : Path,
    dst_root   : Path,
    val_ratio  : float = 0.15,
    seed       : int   = 42,
) -> None:
    
    imgs = collect_image_paths(src_images)
    random.seed(seed)
    random.shuffle(imgs)
 
    n_val  = max(1, int(len(imgs) * val_ratio))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}
 
    for split, paths in splits.items():
        (dst_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dst_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img in tqdm(paths, desc=f"  Copying {split}", unit="file"):
            shutil.copy2(img, dst_root / "images" / split / img.name)
            lbl = src_labels / img.with_suffix(".txt").name
            if lbl.exists():
                shutil.copy2(lbl, dst_root / "labels" / split / lbl.name)
 
    log.info(
        "Auto-split complete → train: %d  val: %d",
        len(splits["train"]),
        len(splits["val"]),
    )

#During Training , YOLO reads this YAML file to understand where the training
#  and validation data are located, how many classes there are, and what their names are.
def build_data_yaml(
    dataset_root : Path,
    class_names  : list[str],
    yaml_path    : Path,
) -> None:
    """Write the dataset YAML consumed by Ultralytics."""
    cfg = {
        "path" : str(dataset_root.resolve()),
        "train": "images/train",
        "val"  : "images/val",
        "nc"   : len(class_names),
        "names": class_names,
    }
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    log.info("Dataset YAML written → %s", yaml_path)

#This code checks whether CUDA-enabled GPUs are available in the system
def detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' based on what is available."""
    if torch.cuda.is_available():
        n   = torch.cuda.device_count()
        ids = ",".join(str(i) for i in range(n))
        log.info("CUDA available — %d GPU(s): %s", n, ids)
        return ids if n > 1 else "0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        log.info("Apple MPS available.")
        return "mps"
    log.warning("No GPU found — training on CPU (will be slow).")
    return "cpu"

def run_pipeline(args) -> None:
    # ── 1. Resolve paths ────────────────────────────────────────────────────
    dataset_root  = Path(args.dataset_root).resolve()
    model_weights = args.weights
    output_dir    = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
 
    log.info("=" * 60)
    log.info("  YOLOv8l Fine-Tuning — Pseudo-RGB Thermal Images")
    log.info("=" * 60)
    log.info("Dataset root : %s", dataset_root)
    log.info("Weights      : %s", model_weights)
    log.info("Output dir   : %s", output_dir)
 
    # ── 2. Validate / prepare dataset ───────────────────────────────────────
    images_root  = dataset_root / "images"
    labels_root  = dataset_root / "labels"
 
    train_imgs   = images_root / "train"
    val_imgs     = images_root / "val"
    train_labels = labels_root / "train"
    val_labels   = labels_root / "val"
 
    need_split = not train_imgs.exists() or not val_imgs.exists()
 
    if need_split:
        if not args.auto_split:
            log.error(
                "train/val sub-directories not found under %s/images/.\n"
                "Re-run with --auto_split to create them automatically.",
                dataset_root,
            )
            sys.exit(1)
 
        log.info("train/val dirs missing → running auto-split (%.0f%% val)…",
                 args.val_ratio * 100)
 
        flat_images = images_root if images_root.exists() else dataset_root
        flat_labels = labels_root if labels_root.exists() else dataset_root
 
        auto_split(
            src_images = flat_images,
            src_labels = flat_labels,
            dst_root   = dataset_root,
            val_ratio  = args.val_ratio,
        )
 
    # ── 3. Sanity-check labels ───────────────────────────────────────────────
    log.info("Verifying train split …")
    ok, miss, warns = verify_dataset(train_imgs, train_labels)
    for w in warns:
        log.warning(w)
    log.info("  Train → %d paired  |  %d missing labels", ok, miss)
 
    log.info("Verifying val split …")
    ok, miss, warns = verify_dataset(val_imgs, val_labels)
    for w in warns:
        log.warning(w)
    log.info("  Val   → %d paired  |  %d missing labels", ok, miss)
 
    if ok == 0:
        log.error("No valid image–label pairs found. Aborting.")
        sys.exit(1)
 
    # ── 4. Write data.yaml ───────────────────────────────────────────────────
    yaml_path = output_dir / "pseudo_rgb_data.yaml"
    build_data_yaml(dataset_root, args.class_names, yaml_path)
 
    # ── 5. Detect device ────────────────────────────────────────────────────
    device = detect_device()
 
    # ── 6. Load model ───────────────────────────────────────────────────────
    log.info("Loading model: %s", model_weights)
    try:
        from ultralytics import YOLO
    except ImportError:
        log.error("ultralytics not installed.  Run:  pip install ultralytics")
        sys.exit(1)
 
    model = YOLO(model_weights)
 
    # ── 7. Optionally freeze backbone ────────────────────────────────────────
    if args.freeze > 0:
        log.info("Freezing first %d layers (backbone).", args.freeze)
 
    # ── 8. Train ─────────────────────────────────────────────────────────────
    log.info("Starting training …")
    log.info(
        "  Epochs: %d  |  img_size: %d  |  batch: %d  |  device: %s",
        args.epochs, args.img_size, args.batch, device,
    )
 
    train_args = dict(
        data          = str(yaml_path),
        epochs        = args.epochs,
        imgsz         = args.img_size,
        batch         = args.batch,
        lr0           = args.lr,
        weight_decay  = args.weight_decay,
        warmup_epochs = args.warmup_epochs,
        patience      = args.patience,
        freeze        = args.freeze,
        workers       = args.workers,
        device        = device,
        project       = str(output_dir),
        name          = "exp",
        exist_ok      = True,
        pretrained    = True,
        optimizer     = "AdamW",
        cos_lr        = True,           # cosine LR decay
        plots         = True,           # save training plots
        save          = True,
        save_period   = 10,             # checkpoint every N epochs
        cache         = False,          # set True to cache images in RAM
        amp           = True,           # mixed precision (ignored on CPU)
        verbose       = True,
        # ── augmentation ──
        **AUGMENT_CFG,
    )
 
    results = model.train(**train_args)
 
    # ── 9. Post-training summary ─────────────────────────────────────────────
    best_weights = output_dir / "exp" / "weights" / "best.pt"
    last_weights = output_dir / "exp" / "weights" / "last.pt"
 
    log.info("=" * 60)
    log.info("Training complete.")
    if best_weights.exists():
        log.info("  Best weights : %s", best_weights)
    if last_weights.exists():
        log.info("  Last weights : %s", last_weights)
 
    # ── 10. Quick validation on best weights ─────────────────────────────────
    if best_weights.exists():
        log.info("Running validation on best weights …")
        best_model = YOLO(str(best_weights))
        metrics    = best_model.val(
            data    = str(yaml_path),
            imgsz   = args.img_size,
            batch   = args.batch,
            device  = device,
            workers = args.workers,
            project = str(output_dir),
            name    = "val_best",
            exist_ok= True,
        )
        log.info("  mAP@0.5      : %.4f", metrics.box.map50)
        log.info("  mAP@0.5:0.95 : %.4f", metrics.box.map)
        log.info("  Precision    : %.4f", metrics.box.mp)
        log.info("  Recall       : %.4f", metrics.box.mr)
 
    log.info("All done. Results saved to: %s", output_dir / "exp")
 
 