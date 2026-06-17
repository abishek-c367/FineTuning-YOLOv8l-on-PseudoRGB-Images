"""
Pre-caching Script for Pseudo-RGB Images
=========================================
Run this ONCE before training to generate and save all pseudo-RGB images.
During training, these are loaded instantly instead of running CycleGAN each step.

Usage:
    python cache_pseudo_rgb.py

After this completes, set USE_PSEUDO_RGB_CACHE = True in train_distillation.py
and point PSEUDO_RGB_CACHE_DIR to the same directory used here.
"""
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
import kornia.augmentation as K
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

import config as args
from Cycle_gan.cyclegan_loader import load_cyclegan_generator
from data.flir_dataset import FlirCocoPaths, FlirThermalCocoDataset, ultralytics_collate

# ─────────────────────────────────────────────────────────────
# CONFIG — match these to your train_distillation.py settings
# ─────────────────────────────────────────────────────────────
CACHE_DIR         = "Cycle_gan/pseudo_rgb_cache"         # where to save .png files
CYCLEGAN_WEIGHTS  = args.cyclegan_weights      # path to your CycleGAN checkpoint
FLIR_ROOT         = Path(args.flir_root)
SPLIT             = args.split                 # e.g. "images_thermal_train"
COCO_JSON         = args.coco_json             # e.g. "coco.json"
IMGSZ             = args.imgsz                 # must match training imgsz
BATCH_SIZE        = args.batch                          # can be larger than training batch, just for speed
NUM_WORKERS       = args.workers
DEVICE            = torch.device(args.device)
# ─────────────────────────────────────────────────────────────


def generate_pseudo_rgb_batch(
    thermal_batch: torch.Tensor,
    cyclegan_generator: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Mirrors _generate_pseudo_rgb() in DistillationTrainer exactly.
    thermal_batch: (B, 1, H, W) — single channel thermal, values in [0, 1]
    returns:       (B, 3, H, W) — pseudo RGB in [0, 1]
    """
    with torch.no_grad():
        thermal_batch = thermal_batch.to(device)

        # Step 1: repeat single channel → 3 channels (same as trainer)
        thermal_3ch = thermal_batch.repeat(1, 3, 1, 1)   # (B, 3, H, W)

        # Step 2: normalize to [-1, 1] for CycleGAN (same as trainer)
        thermal_normalized = K.Normalize(
            mean=torch.tensor([0.5, 0.5, 0.5]),
            std=torch.tensor([0.5, 0.5, 0.5]),
        )(thermal_3ch)

        # Step 3: CycleGAN forward
        pseudo_rgb = cyclegan_generator(thermal_normalized)  # (B, 3, H, W), range [-1, 1]

        # Step 4: convert back to [0, 1] (same as trainer)
        pseudo_rgb = (pseudo_rgb + 1) / 2                   # (B, 3, H, W), range [0, 1]

    return pseudo_rgb.cpu()   # save on CPU to avoid GPU memory bloat


def build_image_id_list(dataset: FlirThermalCocoDataset) -> list:
    """
    Extract stable image IDs from the dataset for naming cache files.
    Uses dataset index as fallback if no explicit image_id available.
    """
    ids = []
    for i in range(len(dataset)):
        # FlirThermalCocoDataset stores image paths; use stem as ID
        try:
            img_path = dataset.image_paths[i]          # adjust if attribute differs
            ids.append(Path(img_path).stem)
        except AttributeError:
            ids.append(str(i).zfill(6))                # fallback: zero-padded index
    return ids


def cache_already_complete(cache_dir: str, dataset_len: int) -> bool:
    """Check if cache already has all files so we can skip re-generation."""
    if not os.path.exists(cache_dir):
        return False
    existing = len([f for f in os.listdir(cache_dir) if f.endswith(".png")])
    return existing >= dataset_len


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ── Dataset (no shuffle — we need stable index → file mapping) ──
    split_dir = FLIR_ROOT / SPLIT
    coco_json  = split_dir / COCO_JSON

    dataset = FlirThermalCocoDataset(
        paths=FlirCocoPaths(images_dir=split_dir, coco_json=coco_json),
        imgsz=IMGSZ,
    )
    #The above dataset now returns (thermal tensor, file_name , target)
    print(f"Dataset size: {len(dataset)} images")

    # Check if cache is already complete
    if cache_already_complete(CACHE_DIR, len(dataset)):
        print(f"✓ Cache already complete ({len(dataset)} files in '{CACHE_DIR}'). Nothing to do.")
        return

    # Count existing files to allow resuming interrupted caching
    existing_files = set(os.listdir(CACHE_DIR))
    existing_indices = set()
    for fname in existing_files:
        if fname.endswith(".png"):
            try:
                idx = int(fname.replace("pseudo_rgb_", "").replace(".png", ""))
                existing_indices.add(idx)
            except ValueError:
                pass
    print(f"Found {len(existing_indices)} existing cached files — will skip those.")

    # ── DataLoader (shuffle=False is critical for index stability) ──
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,          # MUST be False — index must match cache filename
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=ultralytics_collate,
    )

    # ── CycleGAN Generator ──
    print(f"Loading CycleGAN generator from: {CYCLEGAN_WEIGHTS}")
    cyclegan_generator = load_cyclegan_generator(
        CYCLEGAN_WEIGHTS,
        device=DEVICE,
        input_nc=3,
        output_nc=3,
    )
    cyclegan_generator.eval()
    print("✓ Generator loaded and frozen.")

    # ── Main caching loop ──
    skipped    = 0
    saved      = 0

    print(f"\nSaving pseudo-RGB images to: '{CACHE_DIR}/'")
    print("Each file: pseudo_rgb_<index>.png  →  PNG image (H, W, 3), uint8 [0,255]\n")

    pbar = tqdm(dataloader, desc="Generating pseudo-RGB cache")

    for batch in pbar:
        # Unpack batch — same logic as train_distillation loop
        thermal_batch, _,file_names = batch  # thermal_batch: (B, 1, H, W), files_names: list of file names
        #set of existing files in cache dir to avoid repeated os.listdir calls
        existing_files = set(os.listdir(CACHE_DIR))
        #Initialize lists to track which indices/files need processing
        indices_to_process = []
        filenames_to_process = []

        # Check each file in the batch to see if its corresponding cache file already exists

        for i, fname in enumerate(file_names):

            cache_name = os.path.splitext(fname)[0] + ".png"

            if cache_name not in existing_files:
                indices_to_process.append(i)
                filenames_to_process.append(cache_name)
            else:
                skipped += 1

        #make a subset of the batch that only includes the files that need processing to save GPU time and memory
        subset = thermal_batch[indices_to_process]        # (N, 1, H, W)
        # Generate pseudo-RGB for the subset
        pseudo_rgb_batch = generate_pseudo_rgb_batch(
            subset,
            cyclegan_generator,
            DEVICE
        )
        # Now save each generated pseudo-RGB image to disk with the correct filename
        for local_i, cache_filename in enumerate(filenames_to_process):

            save_path = os.path.join(CACHE_DIR, cache_filename)

            img_tensor = pseudo_rgb_batch[local_i]

            img_np = (
                img_tensor.permute(1, 2, 0)
                .cpu()
                .numpy() * 255
            ).astype(np.uint8)

            img_pil = Image.fromarray(img_np)

            img_pil.save(save_path)

            saved += 1





    print(f"\n✓ Caching complete!")
    print(f"  Saved  : {saved} new files")
    print(f"  Skipped: {skipped} already cached files")
    print(f"  Total  : {saved + skipped} files in '{CACHE_DIR}/'")
    print(f"\nNext step: set USE_PSEUDO_RGB_CACHE = True in train_distillation.py")


if __name__ == "__main__":
    main()
    # import json
    # i = 0
    # with open("/home/aryan_s2/Detection_with_Distillation/FLIR_ADAS_v2/images_rgb_val/coco.json") as f:
    #     data = json.load(f)
    # # Create image-id -> filename mapping
    # image_map = {img["id"]: img["file_name"] for img in data["images"]}
    # for key,value in image_map.items():
    #     if (i <= 10):
    #         print(f"{key}:{value}\n")
    #         i+=1
        
    # # Example annotation
    # ann = data["annotations"][0]
    # print(ann)

    # image_name = image_map[ann["image_id"]]

    # print(image_name)