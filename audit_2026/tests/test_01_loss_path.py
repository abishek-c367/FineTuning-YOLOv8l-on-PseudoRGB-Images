"""T01: Does the student's detection-loss path actually run with the installed Ultralytics?
Reproduces exactly what Distillation.train_step does: student(x, return_features=True) in train mode
-> _convert_fpn_to_ultralytics_format -> UltralyticsYOLOLoss(v8DetectionLoss). Also tests the plain list path.
Checks: finite loss, backward works, gradients finite and non-zero on head + backbone.
"""
import _common, torch, traceback, json
from _common import env_info, save
res = {"env": env_info(), "checks": {}}
torch.manual_seed(0)
from Student.yolov8_thermal import yolov8s_thermal
from Distillation import create_yolo_loss, DistillationTrainer

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student = yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=False).to(dev)
res["student_params"] = sum(p.numel() for p in student.parameters())
res["student_params_with_vit"] = sum(p.numel() for p in yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=True).parameters())
res["student_params_default_ctor_nc3"] = sum(p.numel() for p in yolov8s_thermal(num_classes=3).parameters())
student.train()
B, S = 2, 512
x = torch.rand(B, 1, S, S, device=dev)
targets = {"batch_idx": torch.tensor([0, 0, 1], device=dev),
           "cls": torch.tensor([0., 2., 1.], device=dev),
           "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.3], [0.2, 0.2, 0.1, 0.1], [0.7, 0.6, 0.3, 0.2]], device=dev)}

out = student(x, return_features=True)
res["student_output_keys"] = list(out.keys())
preds = out["predictions_ultralytics"]
res["pred_shapes"] = [list(p.shape) for p in preds]
res["stride"] = student.stride.tolist()

try:
    loss_fn = create_yolo_loss(num_classes=3, model=student, require_ultralytics=True).to(dev)
    res["checks"]["loss_constructed"] = True
except Exception as e:
    res["checks"]["loss_constructed"] = f"FAIL: {e!r}\n{traceback.format_exc()}"
    save("T01_loss_path.json", res); raise SystemExit(1)

# (a) exact train_step path: tuple (pred_tensor, features_dict)
tr = DistillationTrainer.__new__(DistillationTrainer)
tup = DistillationTrainer._convert_fpn_to_ultralytics_format(tr, preds)
res["tuple_types"] = [type(tup[0]).__name__, type(tup[1]).__name__, list(tup[1].keys())]
for name, inp in [("tuple_path_as_in_train_step", tup), ("plain_list_path", preds)]:
    try:
        student.zero_grad(set_to_none=True)
        l = loss_fn(inp, targets)
        total = l[0] if isinstance(l, (tuple, list)) else l
        total = total.sum() if total.numel() > 1 else total
        comps = (l[1].tolist() if torch.is_tensor(l[1]) else {k: float(v) for k, v in l[1].items()}) if isinstance(l, (tuple, list)) else None
        total.backward()
        gn = {n: float(p.grad.norm()) for n, p in student.named_parameters() if p.grad is not None}
        res["checks"][name] = {"loss": float(total), "finite": bool(torch.isfinite(total)),
                               "components_box_cls_dfl": comps,
                               "n_params_with_grad": len(gn), "n_params": len(list(student.parameters())),
                               "grad_norm_stem": gn.get("backbone.stem.conv.weight"),
                               "grad_norm_P2_cls": gn.get("head.heads.P2.cls_pred.weight"),
                               "all_grads_finite": all(torch.isfinite(p.grad).all().item() for p in student.parameters() if p.grad is not None)}
    except Exception as e:
        res["checks"][name] = f"FAIL: {e!r}\n{traceback.format_exc()}"
    # the forward graph was consumed; recompute for the next variant
    out = student(x, return_features=True); preds = out["predictions_ultralytics"]
    tup = DistillationTrainer._convert_fpn_to_ultralytics_format(tr, preds)

# (b) loss with NO targets in batch (empty) must be finite
try:
    empty = {"batch_idx": torch.zeros(0, device=dev), "cls": torch.zeros(0, device=dev), "bboxes": torch.zeros(0, 4, device=dev)}
    l = loss_fn(preds, empty); total = l[0] if isinstance(l, (tuple, list)) else l
    res["checks"]["empty_targets_loss"] = {"loss": float(total.sum()), "finite": bool(torch.isfinite(total.sum()))}
except Exception as e:
    res["checks"]["empty_targets_loss"] = f"FAIL: {e!r}"

save("T01_loss_path.json", res)
print(json.dumps(res["checks"], indent=1, default=str)[:3000])
