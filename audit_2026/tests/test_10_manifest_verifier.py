"""T10: audit_tools/verify_manifest.py on a synthetic FLIR-style mini dataset with planted violations.
Checks that each planted violation is flagged (video id shared across splits, TIFF without JPEG, dimension mismatch,
exact duplicate across splits, near-duplicate across splits, box outside the image, undeclared category,
count above the 14-bit container) and that a clean dataset passes with exit code 0."""
import _common, os, sys, json, tempfile, subprocess, shutil
import numpy as np
from PIL import Image
from _common import env_info, save
TOOL = os.path.join(os.path.dirname(_common.__file__), "..", "audit_tools", "verify_manifest.py")
res = {"env": env_info()}
rng = np.random.default_rng(0)

def write_frame(d, stem, arr, jpeg=True, jpeg_dims=None):
    Image.fromarray(arr.astype(np.uint16)).save(os.path.join(d, stem + ".tiff"))
    if jpeg:
        a8 = (np.clip((arr - 1000) / 2000.0, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(a8)
        if jpeg_dims: im = im.resize(jpeg_dims)
        im.save(os.path.join(d, stem + ".jpg"), quality=90)

def make_split(root, split, specs, cats=(1, 2, 3)):
    d = os.path.join(root, split, "data"); os.makedirs(d)
    images, anns = [], []
    for i, (stem, arr, kw, boxes) in enumerate(specs):
        write_frame(d, stem, arr, **kw)
        images.append({"id": i, "file_name": f"data/{stem}.jpg", "height": arr.shape[0], "width": arr.shape[1]})
        for b in boxes:
            anns.append({"id": len(anns), "image_id": i, "bbox": b[0], "category_id": b[1], "area": b[0][2] * b[0][3], "iscrowd": 0})
    json.dump({"images": images, "annotations": anns, "categories": [{"id": c, "name": str(c)} for c in cats]}, open(os.path.join(root, split, "coco.json"), "w"))

def frame(seed, h=48, w=64):
    return np.random.default_rng(seed).integers(1000, 3000, size=(h, w)).astype(np.uint16)

def run(root, splits):
    out = os.path.join(root, "out")
    p = subprocess.run([sys.executable, "-s", TOOL, "--root", root, "--splits", *splits, "--out", out], capture_output=True, text=True)
    rep = open(os.path.join(out, "REPORT.md")).read()
    return p.returncode, rep

# clean dataset
root = tempfile.mkdtemp(prefix="gateir_manifest_clean_")
make_split(root, "train", [(f"video-A-frame-{i:06d}", frame(i), {}, [([5, 5, 10, 10], 1)]) for i in range(3)])
make_split(root, "val", [(f"video-B-frame-{i:06d}", frame(100 + i), {}, [([1, 1, 8, 8], 2)]) for i in range(2)])
rc, rep = run(root, ["train", "val"])
res["clean_exit_code"] = rc; res["clean_passes"] = rc == 0 and "Hard failures (0)" in rep
shutil.rmtree(root)

# planted violations
root = tempfile.mkdtemp(prefix="gateir_manifest_bad_")
dup = frame(7)
near = dup.copy(); near[0, 0] += 1
over = frame(8); over[3, 3] = 2 ** 14 + 5
make_split(root, "train", [
    ("video-A-frame-000001", frame(1), {}, [([5, 5, 10, 10], 1)]),
    ("video-C-frame-000001", dup, {}, []),                              # exact duplicate (also in val)
    ("video-D-frame-000001", frame(9), {}, []),                         # near-duplicate partner in val
    ("video-E-frame-000001", frame(2), {"jpeg": False}, []),            # TIFF without JPEG
    ("video-F-frame-000001", frame(3), {"jpeg_dims": (32, 24)}, []),    # JPEG dims mismatch
    ("video-G-frame-000001", over, {}, []),                             # count above 14-bit container
    ("video-H-frame-000001", frame(4), {}, [([60, 40, 10, 10], 1)]),    # box outside the 64x48 image
    ("video-I-frame-000001", frame(5), {}, [([1, 1, 5, 5], 9)]),        # undeclared category
])
make_split(root, "val", [
    ("video-A-frame-000099", frame(50), {}, []),                        # video A shared with train
    ("video-J-frame-000001", dup, {}, []),                              # exact duplicate of train C
    ("video-K-frame-000001", frame(9).copy() + 0, {}, []),              # same content as train D -> dHash near-duplicate but different sha? identical -> exact; perturb:
])
# make the K frame a near-duplicate rather than exact: add tiny noise
arrK = frame(9).astype(np.int32); arrK[::7, ::5] += 1
Image.fromarray(arrK.astype(np.uint16)).save(os.path.join(root, "val", "data", "video-K-frame-000001.tiff"))
rc, rep = run(root, ["train", "val"])
res["bad_exit_code"] = rc
flags = {
    "video_shared": "video id A appears in splits" in rep,
    "tiff_without_jpeg": "video-E-frame-000001: TIFF without a JPEG pair" in rep,
    "jpeg_dims_mismatch": "video-F-frame-000001: JPEG" in rep and "dimension mismatch" in rep,
    "exact_duplicate": "exact duplicate TIFF" in rep,
    "near_duplicate": "near-duplicate frames" in rep and "video-K-frame-000001" in rep,
    "box_outside": "video-H-frame-000001: box" in rep,
    "undeclared_category": "undeclared category id 9" in rep,
    "count_over_container": "exceeds the 14-bit container" in rep,
}
res["planted_violations_flagged"] = flags
res["report_excerpt"] = rep[:1500]
shutil.rmtree(root)
res["all_pass"] = bool(res["clean_passes"] and rc == 1 and all(flags.values()))
save("T10_manifest_verifier.json", res); print(json.dumps({k: v for k, v in res.items() if k not in ("env", "report_excerpt")}, indent=1, default=str))
