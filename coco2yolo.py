# convert_coco_to_yolo.py
import json
from pathlib import Path

coco_json   = "C:\Project\Project_Detection\FLIR_ADAS_v2\images_thermal_val\coco.json"
output_dir  = Path("C:\Project\Project_Detection\FLIR_ADAS_v2\images_thermal_val\labels_thermal_val")
output_dir.mkdir(exist_ok=True)

with open(coco_json) as f:
    coco = json.load(f)

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
        cls_id = ann["category_id"] - 1   # COCO is 1-indexed → 0-indexed
        lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    with open(output_dir / f"{stem}.txt", "w") as f:
        f.write("\n".join(lines))

print(f"Done. Labels written to: {output_dir}")