# PROJECT COMPONENT MAP — GATE-IR

Status key: **DONE** = implemented and (per the collaborator's report) executed; **PARTIAL** = implemented but not exercised / not verified; **SCAFFOLD** = code exists, no weights, no training path; **PROPOSED** = slides/PDF only. Evidence labels as in `INPUTS_AND_PROVENANCE.md`. Commits: student repo `373dbdd`, zer0 repo `7948bd6`.

## 1. Training pathway (distillation) — student repo

| Stage | Claimed component | Code | State | Dataset input | Checkpoint | Eval / run command | Notes (code observation) |
|---|---|---|---|---|---|---|---|
| T0 | FLIR ADAS v2 COCO reader, letterbox, 3-class filter | `data/flir_dataset.py` (`FlirThermalCocoDataset`, `_letterbox`, `ultralytics_collate`) | DONE | `FLIR_ADAS_v2/images_thermal_train/{coco.json,data/*.jpg}`; val split `images_thermal_val` | — | — | 8-bit JPEG via PIL `.convert("L")`, /255; letterbox 640×512→512×410 content, pad 0; classes **alphabetical** (bicycle=0, car=1, person=2); `bike`→`bicycle` alias; crowd and other categories dropped. Returns `(thermal, file_name, target)`. |
| T1 | CycleGAN thermal→pseudo-RGB (unpaired) | `Cycle_gan/cyclegan.py` (9-resblock ResNet generator, PatchGAN D), `Cycle_gan/cyclegan_loader.py` (loads `G_BA.*` from `cyclegan_epoch_45.pth`), `data/dataset.py` (`UnpairedDataset`) | DONE per report; **no training script in this repo** (zer0 has `training/train_cyclegan.py` with a different architecture) | `images_rgb_train/data` (A) and `images_thermal_train/data` (B), sequential pairing | `Cycle_gan/cyclegan_epoch_45.pth` (not in repo) | `python Cycle_gan/cyclegan_loader.py` (visual check only) | Input: thermal replicated to 3ch, normalised to [-1,1]; output (x+1)/2. Trained at 512×512 *resize* (not letterbox), inference on letterboxed frames. |
| T2 | Pseudo-RGB cache | `Cycle_gan/cache_pseudo_rgb.py` (writer); `Distillation.DistillationTrainer._load_pseudo_rgb_from_cache` (reader) | DONE per notebook (`pseudo_rgb_cache/data/*.png` exists on cluster) | train split only (val cache not generated; val uses no teacher) | — | `python Cycle_gan/cache_pseudo_rgb.py` | Named by thermal stem; see `CODE_AUDIT_2026.md` C2 (consistent by virtue of FLIR's `data/` prefix; resume logic broken; live fallback dead). |
| T3 | Teacher: YOLOv8-L on pseudo-RGB, features hooked at Detect input (P3,P4,P5) | `Teacher/teacher_yolov8_feature_extractor.py` | DONE | pseudo-RGB (B,3,512,512) in [0,1] | `Teacher/yolov8l.pt` (`config.py`) — **name of the stock COCO checkpoint**; whether a fine-tuned file was substituted is unknown | — | Forward pre-hook on `Detect`; channels (256,512,512) for YOLOv8-L. Frozen. |
| T3' | Teacher fine-tuning on pseudo-RGB (optional) | `Teacher/finetune_yolov8l.py`, labels via `coco2yolo.py` | PARTIAL / UNKNOWN whether run | YOLO-format labels from `coco2yolo.py` | `OUTPUT_DIR/exp/weights/best.pt` | `python Teacher/finetune_yolov8l.py --dataset_root … --weights yolov8l.pt` | `coco2yolo.py` writes `category_id-1` for **all 15 FLIR classes** while `CLASS_NAMES=['person','car','bicycle']` (nc=3) → class ids ≥3 would be rejected by Ultralytics and ids 1/2 would be bike/car swapped vs names. See audit C8. |
| T3'' | Teacher evaluation on pseudo-RGB | `Teacher/evaluate_teacher.py` | PARTIAL (own 11-point VOC AP, conf 0.25 default) | thermal val dir + YOLO labels | — | see docstring | Not COCO mAP; conf-thresholded AP. |
| T4 | Student: YOLOv8-s-style thermal detector, 1-ch input, P2 head, optional ViT neck | `Student/yolov8_thermal.py` (`YOLOv8Thermal`, `yolov8s_thermal`) | DONE | (B,1,512,512) | `checkpoints_distill/distill_epoch_{5,10,…}.pth`, `distill_final.pth` (on cluster, not in repo) | — | Run config: `include_p2=True, use_transformer_neck=False` (`train_distillation.py:107`) → **no ViT neck in the trained model**. Strides [4,8,16,32]. Head output order `[reg(64), cls(nc)]`. Also has an unused `obj_pred` branch. |
| T5 | Detection loss (Ultralytics `v8DetectionLoss`) | `Distillation.UltralyticsYOLOLoss`, `create_yolo_loss`, `DistillationTrainer._convert_fpn_to_ultralytics_format` | DONE (version-sensitive; verified against Ultralytics 8.4.178 in T01) | targets dict `{batch_idx, cls, bboxes}` | — | — | Passes `(pred_tensor, {"boxes","scores","feats"})`; current Ultralytics `parse_output` takes element 1 → dict API. Older 8.x versions expected a list → would break. |
| T6 | Feature-mimic loss P3,P4 (MSE on L2-normalised features, 1×1 conv+BN adapters) | `Distillation.FeatureMimicLoss` | DONE | student fused P3 (128ch, 64×64), P4 (256ch, 32×32) vs teacher (256, 512) | adapters saved in `feature_loss_state` | — | α=0.2 after dividing each loss by its first-batch value (`yolo_loss_ref`, `feat_loss_ref`). Refs reset on resume. |
| T7 | Training loop, train-time metrics, per-250-step val batch, end-of-epoch val | `Distillation.train_distillation`, `metrics.py` | DONE (epoch-5 checkpoint referenced) | train + val loaders (val shuffle=False) | every 5 epochs | `python train_distillation.py` (or `sbatch python.sbatch`) | 50 epochs, batch 8, AdamW 1e-3, cosine T_max=100 (never completes), grad clip 10 on student only. Train-time mAP uses `conf≥0.001`, 101-point AP50/AP75 (no AP50-95). |
| T8 | Offline student evaluation | `Student/evaluate_thermal.py` | PARTIAL (no output in repo) | val split | loads `student_state` | `python Student/evaluate_thermal.py --checkpoint … --no-transformer` | Runs the forward in `model.train()` (BN batch statistics) — see audit C5; default `conf=0.25` truncates the PR curve; class names printed in the wrong order. |
| T9 | Inference notebook | `Notebooks/thermal_inference.ipynb` | DONE but **loads no weights** (audit C1) | one val image | `distill_epoch_5.pth` | — | `load_state_dict(checkpoint, strict=False)` on the wrapper dict; builds the model with defaults (`include_p2=False, ViT=True`) → architecture mismatch as well. |

**Train-only components:** T1, T2, T3, T3', T6. **Inference uses only T4** (+ decoding/NMS). The intended "teacher and CycleGAN excluded at inference" property holds in the code.

## 2. Inference pathway (GATE-IR gate + enhancement) — zer0 repo only

| Stage | Component | Code | State | Weights / labels | Notes |
|---|---|---|---|---|---|
| A | WeatherGate: 5 statistics (variance, min, max, soft-histogram entropy, Laplacian variance) → MLP 5-64-64-3 | `gate/weather_gate.py` | **SCAFFOLD** | none; no training script; no weather labels exist for any dataset in the project | Random-init MLP; outputs are meaningless (T08). Statistics are computed on min-max-normalised 14-bit or [0,1] input. |
| B-fog | Adaptive gamma γ = 1 + 2(mean − 0.5), clamp [0.3, 3] | `preprocessing/fog_enhancer.py` | DONE (deterministic, no params) | — | A global contrast stretch; not a physical de-fog for LWIR (matrix §D). |
| B-rain | LSRB depthwise-separable CNN predicts additive "rain mask", subtract, then LCN (sigmoid local-contrast) or OpenCV CLAHE | `preprocessing/rain_remover.py` | **SCAFFOLD** | none; no training script, no rain-mask targets | Random-init; the additive-streak model is unsupported for LWIR (matrix §D). |
| Router | Hard routing by class id, grouped batch processing | `preprocessing/weather_router.py` | DONE (plumbing) | — | — |
| C | YOLOv8-Thermal (same file as student) | `models/yolov8_thermal.py` | same as T4 | none in this repo | — |
| Demo | `main.py --mode demo/test/inference` | `main.py` | DONE (random weights) | — | Prints latency with random weights; "FPS" line multiplies by batch size. |

**Nothing in the inference pathway has been trained or evaluated.** The slides' "Forward Inference" diagram is PROPOSED only. The student repo contains no gate/enhancement code at all.

## 3. Proposed-only items (slides / summary PDF)

- Weather-specific detection accuracy under clear/fog/rain: no experiment, no labels, no dataset with real thermal weather labels in the project.
- "Real-time" deployment: no measurement with trained weights; T07 gives raw forward latency on an RTX 2080 Ti for cost accounting only.
- M3FD, C3I, SWVI: listed on the slides; no loaders, no downloads, no results.
- ViT neck: implemented, **not** used in the trained configuration.

## 4. Mapping of the brief's "claimed components" to reality

| Brief says | Reality |
|---|---|
| "thermal → CycleGAN → pseudo-RGB → YOLOv8-L teacher → features; thermal → YOLOv8-s student; mimic + detection loss" | Implemented as described; executed at least to epoch 5 on an RTX A6000 per notebook output; no metrics available. |
| "raw thermal → Clear/Fog/Rain gate → identity / gamma / LSRB+LCN → thermal detector" | Plumbing only; gate and LSRB untrained; no weather labels. |
| "YOLOv8-Thermal with ViT neck" | Trained without the ViT neck. |
| "inference excludes teacher and CycleGAN" | True in code. |
