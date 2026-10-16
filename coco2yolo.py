# convert_coco_to_yolo.py
import json
from pathlib import Path

import sys
# audit C6: previously wrote category_id-1 for ALL 15 FLIR classes (person 0, bike 1, car 2, ..., sign 9)
# while the training scripts use nc=3; Ultralytics rejects any image with a label id >= nc.
coco_json   = sys.argv[1] if len(sys.argv) > 1 else "FLIR_ADAS_v2/images_thermal_val/coco.json"
output_dir  = Path(sys.argv[2] if len(sys.argv) > 2 else "FLIR_ADAS_v2/images_thermal_val/labels_thermal_val")
CLASSES = ("bicycle", "car", "person")          # must match data/flir_dataset.CLASSES
ALIASES = {"bike": "bicycle"}
output_dir.mkdir(exist_ok=True)

with open(coco_json) as f:
    coco = json.load(f)

# category_id -> class index (only the 3 allowed classes)
cat_to_cls = {}
for c in coco["categories"]:
    name = ALIASES.get(c["name"], c["name"])
    if name in CLASSES:
        cat_to_cls[c["id"]] = CLASSES.index(name)

# Build image_id → filename map
img_map = {img["id"]: img for img in coco["images"]}

# Build image_id → list of annotations
from collections import defaultdict
ann_map = defaultdict(list)
for ann in coco["annotations"]:
    ann_map[ann["image_id"]].append(ann)

# Write one .txt per image
for img_id, img_info in img_map.items():
    W, H = img_info["width"], img_info["height"]
    stem = Path(img_info["file_name"]).stem
    lines = []
    for ann in ann_map[img_id]:
        x, y, w, h = ann["bbox"]          # COCO: top-left x,y + w,h (absolute)
        cx = (x + w / 2) / W              # normalise to [0,1]
        cy = (y + h / 2) / H
        nw = w / W
        nh = h / H
        if ann.get("iscrowd", 0) or ann["category_id"] not in cat_to_cls:
            continue
        cls_id = cat_to_cls[ann["category_id"]]
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    with open(output_dir / f"{stem}.txt", "w") as f:
        f.write("\n".join(lines))

print(f"Done. Labels written to: {output_dir}")