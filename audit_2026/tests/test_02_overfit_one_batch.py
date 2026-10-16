"""T02: Can the student + Ultralytics loss memorize one synthetic batch (detection only, no teacher)?
Two bright rectangles on a dark image -> boxes. Train 300 AdamW steps, decode with BOTH repo decoders
(metrics.decode_student_predictions and Student/evaluate_thermal.decode_predictions) and check IoU with GT.
"""
import _common, torch, json, math
from _common import env_info, save
torch.manual_seed(0)
from Student.yolov8_thermal import yolov8s_thermal
from Distillation import create_yolo_loss, DistillationTrainer
from metrics import decode_student_predictions, box_iou
from Student.evaluate_thermal import decode_predictions, nms_per_class

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
S = 256
img = torch.zeros(2, 1, S, S, device=dev)
gt = []  # (b, cls, xc, yc, w, h) normalized
boxes_px = {0: [(60, 60, 50, 80, 0), (170, 150, 70, 40, 1)], 1: [(128, 128, 90, 90, 2)]}
for b, lst in boxes_px.items():
    for (xc, yc, w, h, c) in lst:
        img[b, 0, yc - h // 2: yc + h // 2, xc - w // 2: xc + w // 2] = 0.9
        gt.append((b, c, xc / S, yc / S, w / S, h / S))
targets = {"batch_idx": torch.tensor([g[0] for g in gt], device=dev),
           "cls": torch.tensor([float(g[1]) for g in gt], device=dev),
           "bboxes": torch.tensor([[g[2], g[3], g[4], g[5]] for g in gt], device=dev)}

student = yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=False).to(dev)
loss_fn = create_yolo_loss(num_classes=3, model=student, require_ultralytics=True).to(dev)
opt = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=5e-4)
tr = DistillationTrainer.__new__(DistillationTrainer)
hist = []
student.train()
for step in range(300):
    out = student(img, return_features=True)
    tup = DistillationTrainer._convert_fpn_to_ultralytics_format(tr, out["predictions_ultralytics"])
    l = loss_fn(tup, targets); total = l[0].sum() if isinstance(l, (tuple, list)) else l.sum()
    opt.zero_grad(set_to_none=True); total.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), 10.0); opt.step()
    if step % 25 == 0 or step == 299:
        comp = l[1] if isinstance(l, (tuple, list)) else None
        comp = {k: round(float(v), 4) for k, v in comp.items()} if isinstance(comp, dict) else ([round(v, 4) for v in comp.tolist()] if torch.is_tensor(comp) else None)
        hist.append((step, float(total), comp))
        print(step, float(total), hist[-1][2])

student.eval()
with torch.no_grad():
    out = student(img, return_features=True)
    preds = out["predictions_ultralytics"]
    dec_a = decode_student_predictions(preds, student, S, conf_threshold=0.25, iou_threshold=0.5)
    dec_b = [nms_per_class(d, 0.5) for d in decode_predictions(preds, student.stride.to(dev), 3, conf_thres=0.25)]

def match(dets, b):
    outl = []
    for (xc, yc, w, h, c) in boxes_px[b]:
        g = torch.tensor([[xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]], device=dev, dtype=torch.float32)
        if dets.shape[0] == 0:
            outl.append({"gt_cls": c, "best_iou": 0.0, "pred_cls": None}); continue
        ious = box_iou(g, dets[:, :4])[0]
        j = int(ious.argmax())
        outl.append({"gt_cls": c, "best_iou": float(ious[j]), "pred_cls": int(dets[j, 5]), "conf": float(dets[j, 4]), "n_dets": int(dets.shape[0])})
    return outl

res = {"env": env_info(), "loss_history": hist,
       "decoder_metrics_py": {b: match(dec_a[b], b) for b in boxes_px},
       "decoder_evaluate_thermal_py": {b: match(dec_b[b], b) for b in boxes_px}}
res["verdict_loss_decreased"] = hist[-1][1] < 0.3 * hist[0][1]
res["verdict_all_gt_recovered_iou50_cls_ok"] = all(m["best_iou"] >= 0.5 and m["pred_cls"] == m["gt_cls"] for b in boxes_px for m in res["decoder_metrics_py"][b])
save("T02_overfit_one_batch.json", res)
print(json.dumps({k: v for k, v in res.items() if k != "env"}, indent=1, default=str))
