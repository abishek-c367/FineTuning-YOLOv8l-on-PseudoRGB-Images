"""T03: Compare the repo's two AP implementations (metrics.py and Student/evaluate_thermal.py) against
pycocotools on synthetic edge cases: perfect preds, empty preds, empty GT image, duplicate detections,
class confusion, low-score extras, partial overlaps. Reports AP50 and AP50-95 side by side.
"""
import _common, torch, numpy as np, json, io, contextlib
from _common import env_info, save
from metrics import compute_ap_per_class
from Student.evaluate_thermal import compute_map
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

S, NC = 512, 3
rng = np.random.default_rng(0)

def coco_eval(preds, gts, nc, iou_thr=None):
    """preds: list per image of (N,6) [x1,y1,x2,y2,conf,cls]; gts: list per image of (M,5) [cls,x1,y1,x2,y2]."""
    images = [{"id": i + 1, "width": S, "height": S} for i in range(len(gts))]
    cats = [{"id": c + 1, "name": str(c)} for c in range(nc)]
    anns, aid = [], 1
    for i, g in enumerate(gts):
        for row in g:
            c, x1, y1, x2, y2 = row
            anns.append({"id": aid, "image_id": i + 1, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "area": (x2 - x1) * (y2 - y1), "iscrowd": 0}); aid += 1
    dets = []
    for i, p in enumerate(preds):
        for row in p:
            x1, y1, x2, y2, s, c = row
            dets.append({"image_id": i + 1, "category_id": int(c) + 1, "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(s)})
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(); coco.dataset = {"images": images, "categories": cats, "annotations": anns}; coco.createIndex()
        if not dets:
            return {"AP50": 0.0, "AP50_95": 0.0}
        cd = coco.loadRes(dets)
        E = COCOeval(coco, cd, "bbox")
        E.evaluate(); E.accumulate(); E.summarize()
    return {"AP50": float(E.stats[1]), "AP50_95": float(E.stats[0]), "AP75": float(E.stats[2])}

def repo_metrics_py(preds, gts, nc):
    P = [torch.tensor(p, dtype=torch.float32) if len(p) else torch.zeros(0, 6) for p in preds]
    GB = [torch.tensor(g[:, 1:], dtype=torch.float32) if len(g) else torch.zeros(0, 4) for g in gts]
    GC = [torch.tensor(g[:, 0], dtype=torch.long) if len(g) else torch.zeros(0, dtype=torch.long) for g in gts]
    ap50 = compute_ap_per_class(P, GB, GC, nc, 0.5)
    aps = [np.mean([compute_ap_per_class(P, GB, GC, nc, t)[c] for t in np.linspace(0.5, 0.95, 10)]) for c in range(nc)]
    # metrics.py averages over ALL classes incl. absent ones (AP=0) -> replicate its mAP50 definition
    return {"AP50": float(np.mean(list(ap50.values()))), "AP50_95": float(np.mean(aps)), "per_class50": ap50}

def repo_eval_thermal(preds, gts, nc):
    AP = [{"boxes": np.array(p)[:, :4] if len(p) else np.zeros((0, 4)), "scores": np.array(p)[:, 4] if len(p) else np.zeros(0),
           "labels": np.array(p)[:, 5].astype(int) if len(p) else np.zeros(0, int)} for p in preds]
    AG = [{"boxes": np.array(g)[:, 1:] if len(g) else np.zeros((0, 4)), "labels": np.array(g)[:, 0].astype(int) if len(g) else np.zeros(0, int)} for g in gts]
    m, pc = compute_map(AP, AG, nc, np.linspace(0.5, 0.95, 10))
    return {"AP50": m["mAP50"], "AP50_95": m["mAP50_95"], "per_class50": pc}

def box(xc, yc, w, h):
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]

cases = {}
g1 = np.array([[0] + box(100, 100, 60, 80), [1] + box(300, 300, 100, 60), [2] + box(400, 120, 40, 40)], float)
cases["perfect"] = ([np.array([r[1:].tolist() + [0.9, r[0]] for r in g1])], [g1])
cases["empty_preds"] = ([np.zeros((0, 6))], [g1])
cases["empty_gt_image_with_fp"] = ([np.array([box(50, 50, 20, 20) + [0.8, 0]]), np.array([r[1:].tolist() + [0.9, r[0]] for r in g1])], [np.zeros((0, 5)), g1])
dup = np.array([r[1:].tolist() + [0.9, r[0]] for r in g1] + [r[1:].tolist() + [0.5, r[0]] for r in g1])
cases["duplicate_detections"] = ([dup], [g1])
conf = np.array([r[1:].tolist() + [0.9, (r[0] + 1) % 3] for r in g1])
cases["all_classes_wrong"] = ([conf], [g1])
shifted = np.array([(np.array(box(100 + 20, 100, 60, 80)) ).tolist() + [0.9, 0], box(300, 300 + 25, 100, 60) + [0.8, 1], box(400, 120, 40, 40) + [0.7, 2]])
cases["partial_overlap_iou_0.5_0.9"] = ([shifted], [g1])
mixed = np.array([r[1:].tolist() + [0.9, r[0]] for r in g1] + [box(200, 450, 30, 30) + [0.95, 0], box(60, 400, 30, 30) + [0.1, 1]])
cases["tp_plus_high_and_low_score_fp"] = ([mixed], [g1])
# multi-image random
gts_r, preds_r = [], []
for i in range(6):
    n = rng.integers(0, 5); g = []
    for _ in range(n):
        g.append([int(rng.integers(0, 3))] + box(*rng.uniform(80, 430, 2), *rng.uniform(20, 120, 2)))
    g = np.array(g, float) if g else np.zeros((0, 5)); gts_r.append(g)
    p = []
    for r in g:
        if rng.random() < 0.8:
            j = rng.normal(0, 8, 4); p.append((r[1:] + j).tolist() + [float(rng.uniform(0.3, 1)), r[0] if rng.random() < 0.85 else (r[0] + 1) % 3])
    for _ in range(rng.integers(0, 3)):
        p.append(box(*rng.uniform(80, 430, 2), *rng.uniform(20, 120, 2)) + [float(rng.uniform(0.05, 0.6)), int(rng.integers(0, 3))])
    preds_r.append(np.array(p, float) if p else np.zeros((0, 6)))
cases["random_6_images"] = (preds_r, gts_r)

res = {"env": env_info(), "cases": {}}
for name, (preds, gts) in cases.items():
    r = {"pycocotools": coco_eval(preds, gts, NC)}
    for k, f in [("metrics_py", repo_metrics_py), ("evaluate_thermal_py", repo_eval_thermal)]:
        try:
            r[k] = f(preds, gts, NC)
        except Exception as e:
            r[k] = f"FAIL {e!r}"
    res["cases"][name] = r
    print(name, json.dumps({k: (v if isinstance(v, str) else {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'per_class50'}) for k, v in r.items()}, default=str))
save("T03_metrics_vs_pycocotools.json", res)
