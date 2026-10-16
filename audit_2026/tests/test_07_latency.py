"""T07: batch=1 latency of the student variants at 512x512 on this GPU (fp32, eval, no NMS), plus the
CycleGAN generator and the Ultralytics YOLOv8-L teacher forward (training-only components), for cost accounting."""
import _common, torch, time, json
from _common import env_info, save
from Student.yolov8_thermal import yolov8s_thermal
from Cycle_gan.cyclegan import Generator
torch.backends.cudnn.benchmark = True
res = {"env": env_info(), "latency_ms_b1_512": {}, "note": "fp32, eval, no NMS/decoding, cudnn.benchmark=True, 50 warmup + 200 timed iters, batch=1, 512x512, RTX 2080 Ti with other CPU jobs running"}
dev = torch.device("cuda")
def bench(fn, n=200):
    for _ in range(50): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / n * 1000
x1 = torch.rand(1, 1, 512, 512, device=dev); x3 = torch.rand(1, 3, 512, 512, device=dev)
with torch.no_grad():
    for name, kw in [("s_P3P4P5_noViT", dict(include_p2=False, use_transformer_neck=False)), ("s_P2_noViT (trained config)", dict(include_p2=True, use_transformer_neck=False)),
                     ("s_P2_ViT", dict(include_p2=True, use_transformer_neck=True))]:
        m = yolov8s_thermal(num_classes=3, **kw).to(dev).eval()
        res["latency_ms_b1_512"][name] = {"ms": round(bench(lambda: m(x1)), 2), "params_M": round(sum(p.numel() for p in m.parameters()) / 1e6, 2)}
    g = Generator(3, 3, 9).to(dev).eval()
    res["latency_ms_b1_512"]["cyclegan_generator_(train_only)"] = {"ms": round(bench(lambda: g(x3)), 2), "params_M": round(sum(p.numel() for p in g.parameters()) / 1e6, 2)}
    try:
        from ultralytics import YOLO
        y = YOLO("yolov8l.pt").model.to(dev).eval()
        res["latency_ms_b1_512"]["yolov8l_teacher_(train_only)"] = {"ms": round(bench(lambda: y(x3)), 2), "params_M": round(sum(p.numel() for p in y.parameters()) / 1e6, 2)}
        y2 = YOLO("yolov8s.pt").model.to(dev).eval()
        res["latency_ms_b1_512"]["ultralytics_yolov8s_reference"] = {"ms": round(bench(lambda: y2(x3)), 2), "params_M": round(sum(p.numel() for p in y2.parameters()) / 1e6, 2)}
    except Exception as e:
        res["latency_ms_b1_512"]["ultralytics"] = f"FAIL {e!r}"
save("T07_latency.json", res); print(json.dumps(res["latency_ms_b1_512"], indent=1))
