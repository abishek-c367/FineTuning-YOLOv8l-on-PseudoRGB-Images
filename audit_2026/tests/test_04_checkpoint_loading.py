"""T04: Reproduce the notebook's checkpoint loading (Notebooks/thermal_inference.ipynb cell: 'model_state_dict' else whole dict,
strict=False) against a checkpoint written by DistillationTrainer.save_checkpoint (key 'student_state').
Also check the architecture mismatch (notebook builds yolov8s_thermal(num_classes=3) with defaults)."""
import _common, torch, copy, json, io
from _common import env_info, save
torch.manual_seed(0)
from Student.yolov8_thermal import yolov8s_thermal
trained = yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=False)
for p in trained.parameters():
    p.data.add_(torch.randn_like(p) * 0.1)   # make 'trained' weights distinguishable
ckpt = {"epoch": 5, "student_state": trained.state_dict(), "feature_loss_state": {}, "optimizer_state": {}, "scheduler_state": {}, "alpha": 0.2, "losses": None}
buf = io.BytesIO(); torch.save(ckpt, buf); buf.seek(0); checkpoint = torch.load(buf, map_location="cpu")

def notebook_load(model, checkpoint):
    if isinstance(checkpoint, dict):
        state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    else:
        state_dict = checkpoint
    return model.load_state_dict(state_dict, strict=False)

def max_abs_diff(a, b):
    return max(float((pa - pb).abs().max()) for (na, pa), (nb, pb) in zip(a.state_dict().items(), b.state_dict().items()) if pa.dtype.is_floating_point)

res = {"env": env_info()}
# (1) notebook path with the SAME architecture
m1 = yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=False); init_copy = copy.deepcopy(m1)
r = notebook_load(m1, checkpoint)
res["notebook_logic_same_arch"] = {"missing_keys": len(r.missing_keys), "unexpected_keys": r.unexpected_keys[:10], "n_unexpected": len(r.unexpected_keys),
                                   "weights_changed_vs_init": max_abs_diff(m1, init_copy) > 0, "weights_equal_trained": max_abs_diff(m1, trained) < 1e-8}
# (2) notebook path with notebook's default constructor (yolov8s_thermal(num_classes=3)) -> include_p2=False, ViT neck True
m2 = yolov8s_thermal(num_classes=3)
try:
    r2 = m2.load_state_dict(checkpoint["student_state"], strict=False)
    res["notebook_default_ctor_vs_trained_ckpt"] = {"ctor_defaults": {"include_p2": m2.include_p2, "use_transformer_neck": m2.use_transformer_neck},
                                                    "missing_keys_n": len(r2.missing_keys), "unexpected_keys_n": len(r2.unexpected_keys),
                                                    "missing_examples": r2.missing_keys[:5], "unexpected_examples": r2.unexpected_keys[:5]}
except Exception as e:
    res["notebook_default_ctor_vs_trained_ckpt"] = f"FAIL {e!r}"
# (3) correct loader from Student/evaluate_thermal.py
m3 = yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=False); m3.load_state_dict(checkpoint["student_state"])
res["evaluate_thermal_loader_equal_trained"] = max_abs_diff(m3, trained) < 1e-8
save("T04_checkpoint_loading.json", res); print(json.dumps({k: v for k, v in res.items() if k != 'env'}, indent=1, default=str))
