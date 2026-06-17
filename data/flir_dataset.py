from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FlirCocoPaths:
    images_dir: Path
    coco_json: Path


def _letterbox(
    img: torch.Tensor,  # (1,H,W)
    new_size: int = 640,
    pad_value: float = 0.0,
) -> Tuple[torch.Tensor, float, Tuple[int, int]]:
    """
    Resize with aspect ratio preserved + pad to square.

    Returns:
        img_out: (1,new_size,new_size)
        gain: scale factor applied to original coords
        pad: (pad_w, pad_h) added on left/top in pixels of resized space
    """
    _, h, w = img.shape
    gain = min(new_size / h, new_size / w)
    nh, nw = int(round(h * gain)), int(round(w * gain))

    if (nh, nw) != (h, w):
        img_resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(nh, nw), mode="bilinear", align_corners=False
        ).squeeze(0)
    else:
        img_resized = img

    pad_h = new_size - nh
    pad_w = new_size - nw
    top = pad_h // 2
    left = pad_w // 2

    out = torch.full((1, new_size, new_size), pad_value, dtype=img.dtype)
    out[:, top : top + nh, left : left + nw] = img_resized
    return out, gain, (left, top)


class FlirThermalCocoDataset(Dataset):
    """
    FLIR ADAS thermal dataset reader from COCO json.

    Returns:
        thermal: Float tensor (1, imgsz, imgsz) in [0,1]
        target: dict with:
          - cls: (M,1) float tensor (class ids)
          - bboxes: (M,4) float tensor (xywh) normalized to [0,1] in resized+letterboxed image space
    """

    def __init__(
        self,
        paths: FlirCocoPaths,
        imgsz: int = 640,
        allowed_class_names: Tuple[str, ...] = ("person", "car", "bicycle"),
        coco_name_aliases: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.paths = paths
        self.imgsz = int(imgsz)
        self.allowed_class_names = set(allowed_class_names)
        self.coco_name_aliases = coco_name_aliases or {"bike": "bicycle"}

        if not self.paths.coco_json.exists():
            raise FileNotFoundError(f"COCO json not found: {self.paths.coco_json}")

        with self.paths.coco_json.open("r", encoding="utf-8") as f:
            coco = json.load(f)

        # category_id -> contiguous class id [0..nc-1]
        cats = coco.get("categories", [])
        name_by_id = {int(c["id"]): str(c.get("name", "")) for c in cats}
        self.class_name_to_idx = {n: i for i, n in enumerate(sorted(self.allowed_class_names))}

        self.catid_to_classidx: Dict[int, int] = {}
        for cat_id, name in name_by_id.items():
            name = self.coco_name_aliases.get(name, name)
            if name in self.class_name_to_idx:
                self.catid_to_classidx[int(cat_id)] = int(self.class_name_to_idx[name])

        # images list
        self.images = coco.get("images", [])
        self.image_by_id = {int(im["id"]): im for im in self.images}

        # annotations grouped by image
        self.ann_by_image: Dict[int, List[dict]] = {}
        for a in coco.get("annotations", []):
            if a.get("iscrowd", False):
                continue
            cat_id = int(a.get("category_id", -1))
            if cat_id not in self.catid_to_classidx:
                continue
            img_id = int(a["image_id"])
            self.ann_by_image.setdefault(img_id, []).append(a)

        # Keep only images that have at least a file_name
        self.ids: List[int] = [int(im["id"]) for im in self.images if "file_name" in im]

    def __len__(self) -> int:
        return len(self.ids)

    def _load_thermal(self, rel_path: str) -> torch.Tensor:
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise ImportError("Pillow is required to load images. Install with: pip install pillow") from e

        p = self.paths.images_dir / rel_path
        if not p.exists():
            # COCO file_name sometimes includes "data/..." already; try resolving as-is
            p = self.paths.images_dir / Path(rel_path)
        if not p.exists():
            raise FileNotFoundError(f"Thermal image not found: {p}")

        im = Image.open(p)
        im = im.convert("L")  # ensure 1-channel
        arr = torch.from_numpy(__import__("numpy").array(im)).float()  # (H,W)
        # normalize to [0,1]
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr.unsqueeze(0)  # (1,H,W)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        im = self.image_by_id[img_id]
        file_name = str(im["file_name"])
        w = int(im.get("width", 0))
        h = int(im.get("height", 0))

        thermal = self._load_thermal(file_name)
        _, oh, ow = thermal.shape
        if w <= 0 or h <= 0:
            w, h = ow, oh

        thermal, gain, (pad_x, pad_y) = _letterbox(thermal, new_size=self.imgsz, pad_value=0.0)

        anns = self.ann_by_image.get(img_id, [])
        cls_list: List[List[float]] = []
        box_list: List[List[float]] = []

        for a in anns:
            bbox = a.get("bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x, y, bw, bh = map(float, bbox)  # in original pixels
            if bw <= 0 or bh <= 0:
                continue

            # scale + pad in resized space
            x2 = x * gain + pad_x
            y2 = y * gain + pad_y
            bw2 = bw * gain
            bh2 = bh * gain

            # convert to normalized xywh in letterboxed (imgsz,imgsz)
            xc = (x2 + bw2 / 2.0) / self.imgsz
            yc = (y2 + bh2 / 2.0) / self.imgsz
            bw_n = bw2 / self.imgsz
            bh_n = bh2 / self.imgsz

            # clamp
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            bw_n = min(max(bw_n, 0.0), 1.0)
            bh_n = min(max(bh_n, 0.0), 1.0)
            if bw_n <= 0.0 or bh_n <= 0.0:
                continue

            cat_id = int(a["category_id"])
            cls_idx = float(self.catid_to_classidx[cat_id])
            cls_list.append([cls_idx])
            box_list.append([xc, yc, bw_n, bh_n])

        if box_list:
            cls_t = torch.tensor(cls_list, dtype=torch.float32)
            box_t = torch.tensor(box_list, dtype=torch.float32)
        else:
            cls_t = torch.zeros((0, 1), dtype=torch.float32)
            box_t = torch.zeros((0, 4), dtype=torch.float32)

        target = {"cls": cls_t, "bboxes": box_t}
        return thermal,file_name, target


def ultralytics_collate(batch):
    """
    Collate function producing Ultralytics v8DetectionLoss target dict.

    Returns:
        images: (B,1,H,W)
        targets: dict with keys batch_idx, cls, bboxes
                 - batch_idx: (M,) int64
                 - cls: (M,) float32 (NOT (M,1))
                 - bboxes: (M,4) float32 in xywh normalized format
    """

    images = torch.stack([b[0] for b in batch], dim=0)
    cls_all = []
    box_all = []
    batch_idx_all = []
    files = []
    for i, (_img,f, t) in enumerate(batch):
        cls = t["cls"]
        bboxes = t["bboxes"]
        files.append(f)
        if cls.numel() == 0:
            continue
        # Flatten cls from (n, 1) to (n,) if needed
        if cls.dim() > 1:
            cls = cls.squeeze(1)
        cls_all.append(cls)
        box_all.append(bboxes)
        batch_idx_all.append(torch.full((cls.shape[0],), i, dtype=torch.int64))

    if cls_all:
        cls_cat = torch.cat(cls_all, dim=0)  # Now (M,) not (M, 1)
        box_cat = torch.cat(box_all, dim=0)
        batch_idx = torch.cat(batch_idx_all, dim=0)
    else:
        cls_cat = torch.zeros((0,), dtype=torch.float32)  # (0,) not (0, 1)
        box_cat = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx = torch.zeros((0,), dtype=torch.int64)

    targets = {"batch_idx": batch_idx, "cls": cls_cat, "bboxes": box_cat}
    return images, targets,files

if __name__ == '__main__':
    import sys
    import os
    from torch.utils.data import DataLoader
    ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    import config as args
    
    FLIR_ROOT = Path(args.flir_root)
    SPLIT = args.split
    COCO_JSON = args.coco_json
    IMGSZ = args.imgsz

    split_dir = FLIR_ROOT / SPLIT
    coco_json  = split_dir / COCO_JSON

    dataset = FlirThermalCocoDataset(
        paths=FlirCocoPaths(images_dir=split_dir, coco_json=coco_json),
        imgsz=IMGSZ,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,          # MUST be False — index must match cache filename
        num_workers=4,
        pin_memory=False,
        collate_fn=ultralytics_collate,
    )
    # Just print the first batch to verify everything works
    print(dataloader)
    #i want to see the first batch in dataloader
    for batch in dataloader:
        print(batch[2])  # file names

        print(len(batch[2]))
        break



    # print((dataset[0]))