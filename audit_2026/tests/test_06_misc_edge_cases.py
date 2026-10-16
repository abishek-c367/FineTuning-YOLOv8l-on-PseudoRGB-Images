"""T06: (a) metrics.DetectionMetricsAccumulator.update with exactly ONE GT box in the batch (cls.squeeze(-1) on a 1-D tensor);
(b) Student/evaluate_thermal.py evaluates with model.train() -> BatchNorm uses batch statistics and mutates running stats;
(c) class-index conventions: dataset (alphabetical) vs Teacher CLASS_NAMES vs coco2yolo.py (category_id-1) vs evaluate_thermal default names;
(d) dataset letterbox geometry for a 640x512 FLIR frame at imgsz 512."""
import _common, torch, json, copy, os, tempfile
from _common import env_info, save
torch.manual_seed(0)
res = {"env": env_info()}
from Student.yolov8_thermal import yolov8s_thermal
from metrics import DetectionMetricsAccumulator
from data.flir_dataset import FlirThermalCocoDataset, FlirCocoPaths, ultralytics_collate, _letterbox
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = yolov8s_thermal(num_classes=3, include_p2=True, use_transformer_neck=False).to(dev).eval()
x = torch.rand(2, 1, 256, 256, device=dev)
with torch.no_grad():
    preds = m(x, return_features=True)["predictions_ultralytics"]
# (a) one GT box in batch
acc = DetectionMetricsAccumulator(3, 256, 0.001)
t1 = {"batch_idx": torch.tensor([0], device=dev), "cls": torch.tensor([1.], device=dev), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]], device=dev)}
try:
    acc.update(preds, t1, m); res["one_gt_box_update"] = {"ok": True, "n_images_accumulated": len(acc._all_preds)}
except Exception as e:
    res["one_gt_box_update"] = f"{type(e).__name__}: {str(e)[:120]}"
acc2 = DetectionMetricsAccumulator(3, 256, 0.001)
t2 = {"batch_idx": torch.tensor([0, 1], device=dev), "cls": torch.tensor([1., 2.], device=dev), "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]], device=dev)}
try:
    acc2.update(preds, t2, m); res["two_gt_boxes_update"] = {"ok": True, "n_images_accumulated": len(acc2._all_preds)}
except Exception as e:
    res["two_gt_boxes_update"] = f"{type(e).__name__}: {str(e)[:120]}"
# (b) BN train-mode evaluation
m_eval = copy.deepcopy(m)
bn_before = {n: b.clone() for n, b in m_eval.named_buffers() if n.endswith("running_mean")}
with torch.no_grad():
    m_eval.eval(); p_eval = m_eval(x, return_features=True)["predictions_ultralytics"]
    m_eval.train(); p_train = m_eval(x, return_features=True)["predictions_ultralytics"]; m_eval.eval()
bn_after = {n: b for n, b in m_eval.named_buffers() if n.endswith("running_mean")}
res["eval_in_train_mode"] = {"max_abs_output_diff_P3": float((p_eval[1] - p_train[1]).abs().max()),
                             "mean_abs_output_P3": float(p_eval[1].abs().mean()),
                             "running_mean_mutated": max(float((bn_before[n] - bn_after[n]).abs().max()) for n in bn_before) > 0}
# (c) class index conventions
tmp = tempfile.mkdtemp(); os.makedirs(os.path.join(tmp, "data"))
from PIL import Image
Image.fromarray((torch.rand(512, 640) * 255).byte().numpy(), mode="L").save(os.path.join(tmp, "data", "f1.jpg"))
coco = {"images": [{"id": 1, "file_name": "data/f1.jpg", "width": 640, "height": 512}],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "bike"}, {"id": 3, "name": "car"}, {"id": 10, "name": "sign"}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 20, 30, 60]}, {"id": 2, "image_id": 1, "category_id": 2, "bbox": [100, 100, 40, 40]},
                        {"id": 3, "image_id": 1, "category_id": 3, "bbox": [200, 300, 120, 80]}, {"id": 4, "image_id": 1, "category_id": 10, "bbox": [5, 5, 10, 10]}]}
json.dump(coco, open(os.path.join(tmp, "coco.json"), "w"))
from pathlib import Path
ds = FlirThermalCocoDataset(FlirCocoPaths(Path(tmp), Path(tmp) / "coco.json"), imgsz=512)
img, fname, tgt = ds[0]
res["dataset_class_map"] = {"catid_to_classidx": ds.catid_to_classidx, "class_name_to_idx": ds.class_name_to_idx,
                            "returned_cls": tgt["cls"].flatten().tolist(), "file_name": fname, "img_shape": list(img.shape)}
res["teacher_CLASS_NAMES_in_finetune_yolov8l"] = ["person", "car", "bicycle"]
res["coco2yolo_py_rule"] = {"cls_id = category_id - 1": {"person(1)": 0, "bike(2)": 1, "car(3)": 2, "sign(10)": 9}, "filters_to_3_classes": False}
res["evaluate_thermal_default_class_names"] = ["person", "car", "bicycle"]
res["metrics_py_print_names"] = ["bicycle", "car", "person"]
# (d) letterbox geometry
out, gain, (px, py) = _letterbox(torch.rand(1, 512, 640), 512)
res["letterbox_640x512_to_512"] = {"gain": gain, "pad_left_top": [px, py], "content_hw": [round(512 * gain), round(640 * gain)], "box0_norm_xywh": tgt["bboxes"][0].tolist()}
save("T06_misc_edge_cases.json", res); print(json.dumps({k: v for k, v in res.items() if k != 'env'}, indent=1, default=str))
