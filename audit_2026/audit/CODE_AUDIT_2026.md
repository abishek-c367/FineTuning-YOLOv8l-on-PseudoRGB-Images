# CODE AUDIT 2026 — `abishek-c367/FineTuning-YOLOv8l-on-PseudoRGB-Images` @ `373dbdd` and `zer0-data/GATE-IR` @ `7948bd6`

Environment for every tested item: Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0, **Ultralytics 8.4.178**, pycocotools, RTX 2080 Ti (GPU 1), 15 Oct 2026. Test scripts: `~/gateir_audit/tests/test_0{1..8}_*.py`; raw outputs: `~/gateir_audit/results/T0*.json` (each carries `env` with versions and commit hashes). No dataset, weights or logs from the collaborator were available; everything below is static analysis plus synthetic-input tests. **The Ultralytics version used on the cluster is unknown; findings labelled "version-dependent" may differ there.**

Severity: **S1** wrong results silently; **S2** crash / blocks a code path; **S3** quality or reporting defect; **S4** cosmetic.

## 1. Confirmed defects

### C1 — S1 — Inference notebook loads no weights and builds the wrong architecture
- **Evidence:** `Notebooks/thermal_inference.ipynb` cell "Load the Distilled Checkpoint": `state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint`, then `load_state_dict(state_dict, strict=False)`. `DistillationTrainer.save_checkpoint` (`Distillation.py:722-738`) writes the key **`student_state`**. The notebook also constructs `yolov8s_thermal(num_classes=3)` with defaults (`include_p2=False, use_transformer_neck=True`, `Student/yolov8_thermal.py:785-786`) whereas training used `include_p2=True, use_transformer_neck=False` (`train_distillation.py:107`).
- **Test T04:** notebook logic on a trainer-format checkpoint → 369 missing keys, 7 unexpected keys (`epoch, student_state, …`), **weights identical to random init**, prints "✓ Checkpoint loaded successfully". With the notebook's default constructor, even `student_state` would give 13 missing (transformer) and 30 unexpected (P2 head) keys. The notebook's reported parameter count (22,137,772) equals the default-constructor model (T01: 22.14 M), confirming the mismatch.
- **Impact:** every visualisation produced by that notebook came from a randomly initialised network. Any qualitative claim based on it is void.
- **Fix:** use `Student/evaluate_thermal.load_student_from_checkpoint` (which is correct: T04 `evaluate_thermal_loader_equal_trained = true`) and store the architecture flags inside the checkpoint. Patch `patches/0001-notebook-loader.patch`.
- **Regression test:** T04.

### C2 — S2/S3 — Pseudo-RGB cache writer: no `mkdir` for `data/`, broken resume; loader fragile
- **Evidence:** `Cycle_gan/cache_pseudo_rgb.py:102` creates only `CACHE_DIR`; lines 175–206 derive `cache_name = splitext(file_name)[0]+'.png'`; FLIR's `file_name` is `data/video-…jpg`, so the PNG goes under `CACHE_DIR/data/`. Lines 166–181 compare `'data/x.png'` against `os.listdir(CACHE_DIR)` entries (`'data'`), so the skip test never matches. `Distillation.py:337-349` lists `<cache>/data/*.png`; `_load_pseudo_rgb_from_cache` (403–464) joins `<cache>` with the prefixed name.
- **Test T05:** `writer_save_without_mkdir_data = FileNotFoundError` (first run fails unless `data/` pre-exists); `writer_resume_skip_detects_saved_file = false`; `loader_finds_file_written_by_writer = ok` (paths coincide); a dataset without the `data/` prefix → `FileNotFoundError`.
- **Impact:** not a correctness bug for the FLIR run (lookup is by filename, so shuffling is safe); it is a reproducibility trap (fresh run crashes; interrupted caching restarts from zero).
- **Fix:** `os.makedirs(dirname(save_path), exist_ok=True)`; compare against a recursive listing; derive the loader's directory from the writer's constant. Patch `0002-cache-paths.patch`.

### C3 — S2 — Live CycleGAN fallback is dead code; no-cache mode returns `None`
- **Evidence:** `Distillation.py:481-493`: the fallback body is inside a triple-quoted string. `_generate_pseudo_rgb` returns `None` when `use_cache` is False; `train_step` then calls `self.teacher(None, …)`.
- **Test T05:** `no_cache_fallback_returns = None`; a cache miss raises `FileNotFoundError` (not silent). `validate_step` (`Distillation.py:702`) passes the thermal tensor where filenames are expected → `TypeError` (T05), but `validate_step` is commented out of the training loop (`Distillation.py:858, 920`) so it never runs.
- **Impact:** training without a complete cache crashes immediately; nothing silent. The brief's phrasing "passes None into the teacher" is accurate but the consequence is a crash at step 1.
- **Fix:** restore the fallback; make `validate_step` take filenames. Patch `0003-live-fallback.patch`.

### C4 — S3 — Train-time metric accumulator crashes on batches with exactly one GT box (silently skipped)
- **Evidence:** `metrics.py:367` `gt_cls_all = targets["cls"].squeeze(-1)`; `ultralytics_collate` already returns `cls` as 1-D `(M,)`; when M = 1, `squeeze(-1)` yields a 0-d tensor and `gt_cls_all[mask]` raises.
- **Test T06:** `one_gt_box_update = IndexError: too many indices for tensor of dimension 0`; two boxes OK. The training loop wraps the call in `try/except` and prints `[metrics] accumulator.update failed`, so such batches are dropped from train-time mAP.
- **Impact:** monitoring metrics only; biased toward object-dense batches. **Fix:** `.reshape(-1)`. Patch `0004-metrics-squeeze.patch`.

### C5 — S1 (for any reported number) — Offline evaluation runs the network in `train()` mode
- **Evidence:** `Student/evaluate_thermal.py:482-484`: `model.train(); raw_output = model(images, return_features=True); model.eval()`. In train mode BatchNorm uses batch statistics and updates running stats; the comment says this is needed to get the Ultralytics-format list, but `forward(…, return_features=True)` already returns `predictions_ultralytics` in eval mode (`Student/yolov8_thermal.py:914-928`).
- **Test T06:** eval-mode vs train-mode outputs on the same random batch differ by up to 2.86 (mean |output| 0.043 at P3); `running_mean_mutated = true`.
- **Impact:** any mAP from this script depends on batch composition and is not the deployed model's mAP. Also `--conf 0.25` default truncates the PR curve (AP under-estimate), `--include-p2` is always True (`action="store_true", default=True`), and `--no-transformer` must be passed or loading fails; class names are printed in the wrong order (dataset is alphabetical: bicycle, car, person).
- **Fix:** remove `model.train()`, default `conf=0.001`, correct names, and report pycocotools mAP. Patch `0005-eval-mode.patch`.

### C6 — S3 — Three different class-index conventions in one repository
- **Evidence (T06):** dataset/collate/student: `{bicycle:0, car:1, person:2}` (alphabetical sort of allowed names, `data/flir_dataset.py:86`); teacher fine-tune `CLASS_NAMES=['person','car','bicycle']` (`Teacher/finetune_yolov8l.py:21`); `coco2yolo.py:32` writes `category_id−1` for **all** FLIR categories (person 0, bike 1, car 2, …, sign 9) with no filtering; `evaluate_thermal.py` default names `['person','car','bicycle']`; `metrics.py:485` prints alphabetical names (correct for the student).
- **Impact:** (i) if the teacher was fine-tuned with `coco2yolo.py` labels and `nc=3`, every image containing any class id ≥ 3 is rejected by Ultralytics' label check and ids 1/2 are bike/car under the names car/bicycle; (ii) per-class AP printouts for the student are mislabelled. Feature-mimic distillation itself is unaffected (it never uses teacher logits).
- **Fix:** one `CLASSES = ("person","car","bicycle")` constant with an explicit FLIR category map used by every script. Patch `0006-class-map.patch`.

### C7 — S3 — Detection head lacks class-prior bias initialisation
- **Evidence:** `P2DetectionHead.__init__` (`Student/yolov8_thermal.py:603-637`) leaves `cls_pred` bias at PyTorch default; Ultralytics' `Detect.bias_init` sets it to `log(5/nc/(640/s)^2)`.
- **Test T01:** initial `cls_loss` = 2742 (box 2.26, dfl 4.21; batch 2, 512², nc 3); gradient norm on `head.heads.P2.cls_pred.weight` = 2667 vs 142 on the stem. **T02** shows the toy problem still converges, so this is a quality/stability issue, not a blocker; with real data and `lr=1e-3`, expect a noisy first epoch and a large, uninformative `yolo_loss_ref` (the first-batch normaliser in `train_step`).
- **Fix:** bias init as in Ultralytics; patch `0007-head-bias-init.patch`.

### C8 — S3 — Loss-normalisation and schedule details that change on resume
- `yolo_loss_ref`/`feat_loss_ref` are set from the first batch seen and are not saved in the checkpoint (`Distillation.py:636-647, 729-737`), so the effective α changes when resuming. `CosineAnnealingLR(T_max=100)` with 50 epochs never reaches `eta_min`. `clip_grad_norm_` covers the student only, not the adapters. The unused `obj_pred` branch (8 tensors) receives no gradient (T01: 223/231) and wastes parameters. None of these is wrong; all are undocumented choices that must be stated in a methods section.

### C9 — S3 (zer0 repo) — Gate and rain remover are untrained; gamma rule ignores contrast
- **Evidence:** no training script, labels or weights for `WeatherGate` or `RainRemover` in either repo (grep). `FogEnhancer.compute_adaptive_gamma` uses only the mean (`preprocessing/fog_enhancer.py:84-94`).
- **Test T08:** random-init gate assigns class 1 (fog) to clear, fog and rain synthetic inputs; on a low-contrast "fog" input with mean 0.49 the gamma rule gives γ = 0.987 (no change) — LWIR fog is a *contrast* loss, not a mean shift, so the rule does not even address the stated degradation; random-init `RainRemover` changes the image by 0.15 mean absolute (noise).
- **Impact:** the inference pathway in the slides has no trained component; `main.py` latency/FPS lines come from random weights.

## 2. Version sensitivity (tested)

- **T01:** the exact `train_step` path — tuple `(pred_tensor, {"boxes","scores","feats"})` — works with Ultralytics 8.4.178 because `v8DetectionLoss.parse_output` takes element 1 and `loss()` indexes `preds["boxes"]`. The **plain-list path fails** on 8.4.178 (`list indices must be integers or slices, not str`). Older 8.x releases expected a list of feature maps; the code would then fail the other way. Conclusion: the repository only runs on dict-API Ultralytics; pin the version (`ultralytics==8.4.178` reproduces here).
- Stride/anchor conventions: `make_anchors(preds["feats"], stride)` with `stride=[4,8,16,32]` and head order P2→P5 are consistent; both repo decoders and the loss agree (T02: all GT recovered at IoU ≥ 0.995, correct classes, after 300 steps; loss 5498 → 1.94).

## 3. Not bugs (false alarms from the static review)

- Cache ↔ loader directory mismatch: **not a correctness bug** for FLIR (C2); shuffled batches are handled by filename.
- Detection-head channel order `[reg, cls]`: **correct** for `v8DetectionLoss` (T01/T02).
- "Teacher gets `None`": only when no cache exists; the run that was done had a cache. Crash, not silent.
- `metrics.py` / `evaluate_thermal.py` AP: **agree with pycocotools** on random multi-image data (AP50 0.5668/0.566 vs 0.5668; AP50-95 0.2409/0.2399 vs 0.2409, T03). Two boundary quirks only (perfect-recall saturation at 0.9901; IoU-exactly-0.5 counted as a miss by `evaluate_thermal.py`).
- Letterbox geometry and box transform: **correct** (T06).
- Inference excludes teacher and CycleGAN: **true**.

## 4. Latency (T07, indicative only)

Batch 1, 512×512, fp32, eval, no NMS, cudnn.benchmark, RTX 2080 Ti with other CPU jobs running: student P3–P5 25.2 ms; **trained config (P2, no ViT) 29.7 ms**; P2+ViT 35.5 ms; CycleGAN generator 35.9 ms and YOLOv8-L 42.7 ms (training only); Ultralytics YOLOv8s reference 26.0 ms. The student (21.5 M params) is roughly twice the size of Ultralytics YOLOv8s (11.2 M) because the custom head uses two 3×3 convs at full width per branch and the P2 branch is wide; "YOLOv8-small" is a misnomer. No "real-time" claim can be made or refuted without an optimised (FP16/TensorRT) measurement on the target device, which does not exist.

## 5. Post-patch verification (branch `audit/2026-10`, same environment)

- T01: initial `cls_loss` 2742 → **5.19** (box 2.25, dfl 4.21); gradient norm on `P2.cls_pred.weight` 2667 → **0.80**; loss finite, backward OK.
- T02: still memorises the toy batch (loss 23.8 → 1.46 in 300 steps, all boxes recovered at IoU > 0.99, both decoders agree).
- T04: unchanged (the fixed notebook now uses `student_state` with `strict=True`; the test characterises the *original* notebook logic and still shows it loads nothing).
- T06: single-GT-box batch now accumulates (`one_gt_box_update: ok`).
- T05 **fails on the patched branch by design**: it replays the original writer/loader logic inline and expects `None` from the no-cache path; with patch 0003 that path now raises `ValueError` when no thermal tensor is supplied. Treat T05 as a characterisation test of commit `373dbdd`.

## 6. Patches and tests

`~/gateir_audit/patches/000{1..7}-*.patch` apply to `student_repo` at `373dbdd` (`git apply`). They are also committed on the branch `audit/2026-10`. Tests: T01–T08 in `tests/`; run with `CUDA_VISIBLE_DEVICES=1 python -s test_0N_*.py` from `tests/`.
