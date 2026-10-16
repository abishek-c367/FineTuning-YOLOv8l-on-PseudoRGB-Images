# REPRODUCTION PROTOCOL — GATE-IR distillation baseline

Goal: reproduce arm **B1** (the collaborator's CycleGAN-teacher distillation run) and **B0/B0'** on this machine, with manifests, pinned versions and logs, before any method work. Nothing below has been run beyond the smoke tests (T01–T08). Steps marked **[APPROVAL]** exceed the "bounded smoke test" budget and need the user's go-ahead.

## 1. Environment (done)

```
~/gateir_audit/venv  (Python 3.10.12)
torch 2.6.0+cu124, torchvision 0.21.0, ultralytics 8.4.178, kornia, pycocotools, torchmetrics
nvidia driver 535.309.01 (CUDA 12.2 runtime compat), 2x RTX 2080 Ti 11 GB
```
Pin file to generate: `~/gateir_audit/venv/bin/pip freeze > audit/requirements-lock.txt` (do after the collaborator confirms their versions; if they used an older Ultralytics, test T01 against it first).

## 2. Inputs to obtain (blocked on the collaborator / user)

| Item | Source | Size | Action |
|---|---|---|---|
| FLIR ADAS v2 archive | https://www.flir.com/oem/adas/adas-dataset-form/ (registration, terms) | ~17 GB zip | **[APPROVAL]** user downloads; place under `~/gateir_data/FLIR_ADAS_v2/` (quarantined, never committed) |
| `cyclegan_epoch_45.pth` | collaborator | ~100 MB | required for B1; without it B1 is a new CycleGAN training (**[APPROVAL]**, ~1–2 GPU-days) |
| Teacher weights actually used | collaborator (stock `yolov8l.pt` or fine-tuned) | 87 MB | Ultralytics will download stock `yolov8l.pt` automatically (done in T07) |
| `checkpoints_distill/*.pth`, training stdout, `eval_metrics.json` | collaborator | — | needed for Gate 1 ("reproduce within tolerance") |
| M3FD, UNIRI-TID, ZUT-FIR-ADAS | public links in `PRIOR_ART_NOVELTY_MATRIX.md` §C | 2–10 GB each | **[APPROVAL]**, only for EX |

## 3. Data manifests and split integrity (first thing after data arrives; CPU only, minutes)

1. `python audit_tools/make_manifest.py <root>` → `manifests/flir_v2_<split>.csv` with path, SHA-256, width, height, n_boxes per class, `video_id` parsed from `video-<id>-frame-<n>`.
2. Assert: no `video_id` appears in more than one of train / val / `video_thermal_test`. If the official stills overlap by video, carve a **video-grouped** val split from train and keep the official val as a second dev set; never touch `video_thermal_test` until the matrix is frozen.
3. Verify annotation integrity: boxes inside the image, categories present, counts per class vs the dataset card (person/car/bicycle).
4. Record FLIR's licence text verbatim in `manifests/LICENSE_FLIR.txt`.

## 4. Smoke pilot on a quarantined subset (bounded, no approval needed once data exists)

- 200 train frames from ≤ 10 videos, 50 val frames from 3 other videos; imgsz 512; 2 epochs; batch 8; GPU 1.
- Run B0' (α = 0) and, if the CycleGAN checkpoint exists, cache + B1. Check: loss finite, `yolo_loss_ref` magnitude, mAP rising above 0, checkpoint reload equality (T04 logic), pycocotools eval on val.
- This pilot sets nothing in the pre-registration.

## 5. Full runs **[APPROVAL]** (each ≥ 2 h)

| Run | Command (from `student_repo` on branch `audit/2026-10`) | Est. time |
|---|---|---|
| Cache | `python Cycle_gan/cache_pseudo_rgb.py` (after patch 0002) | ~1 h, ~10 GB |
| B1 | `CUDA_VISIBLE_DEVICES=1 python train_distillation.py --epochs 50 --batch 8 --alpha 0.2 --seed S` | ~10 h/seed |
| B0' | same with `--alpha 0` (patch adds `--seed`, `--alpha 0` short-circuit) | ~8 h/seed |
| B0 | `yolo detect train model=yolov8s.yaml data=flir3.yaml ch=1 imgsz=512 epochs=50 hsv_h=0 hsv_s=0 seed=S` | ~6 h/seed |
| Eval | `python audit_tools/eval_coco.py --ckpt … --split val` (pycocotools, conf 0.001, NMS 0.7) | minutes |

Logs go to `~/gateir_audit/results/runs/<arm>_seed<S>/` with `stdout.log`, `config.json`, `git_commit.txt`, `pip_freeze.txt`, `metrics.json`.

## 6. Reproduction tolerance (Gate 1)

B1 counts as reproduced if our 3-seed mean mAP50 on the official val split is within the across-seed SD (or ±2 mAP points, whichever is larger) of the collaborator's logged value **computed with the same evaluator**. If the collaborator's numbers come from `evaluate_thermal.py` (train-mode BN, conf 0.25, see C5) they must be recomputed with pycocotools from their checkpoint before comparison.

## 7. Blockers today

- No data, no weights, no logs on this machine → Gate 0 not passed.
- Disk: 76 GB free; FLIR + cache + checkpoints ≈ 30 GB; OK but monitor.
- GPU 0 serves the display; use GPU 1 (`CUDA_VISIBLE_DEVICES=1`). Other CPU-bound jobs (Loyola, TopoQSPR) are running; use `num_workers=4` max.
