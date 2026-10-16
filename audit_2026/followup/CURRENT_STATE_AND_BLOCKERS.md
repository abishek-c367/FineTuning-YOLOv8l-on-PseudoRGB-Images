# CURRENT STATE AND BLOCKERS (16 Oct 2026)

Written in response to the follow-up research roadmap of 16 Oct 2026 (Gate 0 and Gate 1). Every statement below is either **verified on this machine** (V), **reported by the collaborator and not reproduced** (R), or **unknown** (U).

## 1. Exact commits (V)

| Repository | Branch | Commit | Content |
|---|---|---|---|
| `abishek-c367/FineTuning-YOLOv8l-on-PseudoRGB-Images` | `main` | `373dbdd` | collaborator's code as audited; untouched |
| same | `audit/2026-10` (pushed; PR #2 open to `main`) | `cc95761` | `main` + 7 fix commits (`bb1d4ca`..`3f6f1ff`) + `audit_2026/` (`efc4487`, `cc95761`) |
| same | `followup/2026-10` | this branch | `audit/2026-10` + `audit_2026/followup/`, reconciled experiment matrix, test T05b |
| `zer0-data/GATE-IR` | `main` | `7948bd6` | weather-gate / enhancer scaffold (Aryan Sood, Jan 2026) |

The 2026-10-15 audit documents refer to these commits. No other branch of either repository was inspected. `Dataset_Alignment` (merged PR #1 by the collaborator) is contained in `main`.

## 2. Data and weights on this machine (V)

| Item | State | Consequence |
|---|---|---|
| FLIR ADAS v2 archive (thermal TIFF/JPEG, COCO json, RGB) | **absent** (search of the filesystem on 16 Oct found no FLIR, ADAS or thermal image data) | no detector experiment of any track can start; licence/terms not on file |
| `cyclegan_epoch_45.pth` | **absent** | Track A arm 4 and the collaborator's B1 configuration cannot be run; retraining a CycleGAN is 1 to 2 GPU-days and would not reproduce his run |
| Teacher weights actually used (stock `yolov8l.pt` or fine-tuned) | **unknown** (U); stock `yolov8l.pt` downloaded by Ultralytics for tests | if a fine-tuned teacher was used, its labels/class order are a separate unknown (audit C6) |
| `checkpoints_distill/*.pth`, training stdout, `eval_metrics.json`, config, requirements | **absent** | nothing in the project slides or summary can be checked; Gate 1 "reproduce within tolerance" cannot be entered |
| GPU | 2 × RTX 2080 Ti 11 GB; GPU 0 hosts the display and runs another user's job (≈ 0.5 GB, 25 %); GPU 1 idle | GPU 1 only |
| CPU / RAM | 16 cores, 62 GB; other users' CPU jobs (Loyola scripts, a joblib pool) running | `num_workers ≤ 4`; do not disturb |
| Disk | **68 GB free** (93 % used) on 16 Oct, down from 76 GB on 15 Oct | FLIR (≈ 33 GB) + checkpoints fit; a pseudo-RGB cache (≈ 10 GB) and any CycleGAN training would be tight; monitor |

## 2b. FLIR ADAS v2 contents, verified from the vendor page on 16 Oct 2026 (published fact)

Teledyne FLIR's dataset page (oem.flir.com, "FREE Teledyne FLIR Thermal Dataset for Algorithm Training", opened 16 Oct 2026) lists: 26,442 annotated frames, 520,000 boxes, 15 categories; thermal as **14-bit TIFF (no AGC)** and **8-bit JPEG (AGC applied)**; RGB 8-bit JPEG (9,233 train/val RGB images); 7,498 video frames at 24 Hz with a "1:1 match between thermal and visible frames"; elsewhere the same page says "16-bit pre-AGC frames", so the container bit depth is a day-1 header check (matrix §0). No licence text is on the page; it is behind a login and the Terms of Use link. Consequences: the RAW-penalty paper's premise (paired count-level and AGC frames) holds; a literature agent's remark that the raw frames are "not public" is wrong and is not carried forward; pixel alignment of RGB and thermal is not stated anywhere opened, so Track A arm 5 (genuinely paired visible teacher) stays conditional.

## 3. Tested versus only reported (V / R)

**Tested on synthetic batches, both on the original code and on branch `audit/2026-10` (16 Oct re-run, logs in `results/rerun_patched/`):**

- Loss and backward pass run on Ultralytics 8.4.178 through the tuple path used by `train_step`; the plain-list path raises a TypeError on this version (version sensitivity, unchanged by the patches). (T01)
- A synthetic batch is overfit and both decoders recover the planted boxes; with the head bias initialisation the initial loss drops from ≈ 5498 to ≈ 24. (T02)
- The repository's AP matches pycocotools except at perfect recall and at the IoU = 0.5 boundary. (T03)
- The original notebook's loading logic leaves 369 keys unloaded and 7 unexpected, i.e. inference on random weights; it also builds the untrained default architecture; the patched loader reproduces the trained state exactly. (T04)
- Original cache writer/loader path defects (C2, C3) are characterised by T05; the patched code passes T05b (recursive listing, resume/skip under `data/`, mkdir, loader resolution, cache-miss FileNotFoundError, no-cache ValueError, live fallback shape and [0,1] range). T05 fails by design on the patched branch and is kept as the characterisation of the original.
- Single-GT-box crash, train-mode BatchNorm drift in evaluation, three class conventions. (T06)
- Batch-1 512 px fp32 latency on RTX 2080 Ti: student with P2 and no ViT 29.7 ms, P3 to P5 25.2 ms, P2 + ViT 35.5 ms, CycleGAN generator 35.9 ms, YOLOv8-L 42.7 ms, Ultralytics YOLOv8-s 26.0 ms. (T07)
- The zer0 weather gate has no trained weights in the repository; its outputs are those of a random initialisation. (T08)

**Reported by the collaborator, not reproduced (R):** every number in the project slides and summary PDF (mAP, per-class AP, "real-time", small-object gains). The evaluator that produced them is unknown; if it was `Student/evaluate_thermal.py` as committed, the numbers were computed with BatchNorm in training mode, conf 0.25 and the wrong class-name order (audit C5) and are not comparable to pycocotools.

**Not established by anything on this machine:** a reproducible thermal-only baseline; any teacher/student comparison; any weather-routing result; any small-object or real-time claim.

## 4. Gate status

| Gate | Status | Blocker |
|---|---|---|
| 0 Access and provenance | **blocked** | FLIR ADAS v2 archive and licence record; collaborator's weights, logs, config; written agreement on authorship and roles with the zer0 scaffold author |
| 1 Regression audit | **done** (smoke level) | nothing; the seven patches are tested (T01 to T08, T05b); they are not evidence of detector performance |
| 2 Reproducible baseline and pilot | **blocked** by Gate 0 | data |
| 3 Stop or scale | not reached | — |

## 5. Request list to the collaborator (verbatim, ready to send)

1. FLIR ADAS v2: confirm which archive version you downloaded (file name, size, date) and send the licence/terms you accepted; we will download our own copy under the same terms.
2. `cyclegan_epoch_45.pth` (and the CycleGAN training config and log).
3. The teacher checkpoint actually used in `train_distillation.py` runs (stock `yolov8l.pt` or your fine-tuned file), and the class order it was trained with.
4. `checkpoints_distill/` (at least the final epoch of the run reported in the slides), with the stdout log, the exact command line and `pip freeze`.
5. The script or notebook that produced the mAP numbers in the slides, and its raw output (`eval_metrics.json` or equivalent).
6. The train/val/test split definition you used (file lists or the COCO json files), so that video overlap can be checked.
7. Aryan Sood's role in the distillation code and whether he is a co-author of anything derived from it.
