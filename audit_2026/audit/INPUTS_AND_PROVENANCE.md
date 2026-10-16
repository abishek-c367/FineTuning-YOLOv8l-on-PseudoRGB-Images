# INPUTS AND PROVENANCE — GATE-IR audit (15 Oct 2026)

Status labels used throughout the audit: **PDF claim**, **README claim**, **code observation**, **static concern**, **tested result**, **external published result**, **hypothesis**, **inference**.

## 1. Documents supplied

| File | Pages | Content | Evidence status |
|---|---|---|---|
| `~/Downloads/Project_Slides.pdf` | 6 | IIT Roorkee slides "Object detection using Cycle-GAN generated pseudo RGB thermal images". Lists 6 references, 4 datasets (FLIR ADAS, M3FD, C3I, SWVI), a "Data Distillation" training diagram (thermal → CycleGAN → YOLOv8 teacher; thermal → YOLOv8-small student; feature-mimic + YOLO loss) and a "Forward Inference" diagram (thermal → Weather Inf. Gate → Clear / Foggy(gamma) / Rainy(deblurring) → YOLOv8-small). | PDF claim. No numbers, no results. |
| `~/Downloads/GATE-IR_Summary.pdf` | 2 | Problem statement + 3-stage proposal (WeatherGate MLP on variance/entropy/Laplacian; adaptive gamma for fog; LSRB + LCN for rain; YOLOv8-s with ViT neck) + CycleGAN/YOLOv8-L teacher distillation. Text identical to the student repo README. | PDF claim. No results. |
| `~/Downloads/Research_Proposal.pdf` | 3 | Separate AI-generated-image forensics proposal (random compression/resizing, DDIM noise/denoise at random t, ResNet-50/ViT-B encoder, SimCLR NT-Xent loss with same-set positives). Quotes "99%+ clean / 70–75% resized / 60% JPEG-75" with no citation. | PDF claim, unimplemented, no citation for the numbers. Treated as a separate project (see `SECOND_PROJECT_FEASIBILITY.md`). |
| Research brief (markdown, not included here) | — | The audit brief (derived from a prior ChatGPT review). Its recommendations are counter-checked in `RECOMMENDATIONS_COUNTERCHECK.md`. | Instruction document, not evidence. |

## 2. Repositories (cloned read-only to `~/gateir_audit/repos/`)

| Repo | Clone dir | HEAD commit | Commits | Dates | Author(s) | License | Fork? |
|---|---|---|---|---|---|---|---|
| `abishek-c367/FineTuning-YOLOv8l-on-PseudoRGB-Images` | `student_repo` | `373dbdd7c2ff6c8e29f93bcbbb4fe4880c65b519` | 6 (plus branch `Dataset_Alignment`, README-only diff) | 2026-05-18 → 2026-06-17 | Abishek Chaudhari (`abishek_c@mfs.iitr.ac.in`, as `abishek-lgtm` / `abishek-c367`) | none declared | no (GitHub API: repo created 2026-03-28, not a fork, 1 contributor) |
| `zer0-data/GATE-IR` | `zer0_repo` | `7948bd6aa8fe9139a8779f09190e821d900969f6` | 11 | 2026-01-04 → 2026-01-22 | Aryan Sood (`aryansood054@gmail.com`) | README says "MIT License" but no LICENSE file; GitHub API reports no license | no (1 fork, 1 star) |

### 2.1 Relationship between the two repositories (code observation)

- `student_repo/Student/yolov8_thermal.py` (1106 lines) vs `zer0_repo/models/yolov8_thermal.py` (1053 lines): only **96 differing lines** (whitespace-insensitive `diff`). The student version changes the default `include_p2` from True to False and adds the `return_features=True` branch that returns `predictions_ultralytics` in training mode. Everything else (backbone, PAN neck, P2 head, TransformerNeck, docstrings) is identical.
- `student_repo/Distillation.py` is a modified copy of `zer0_repo/training/distillation.py`: same classes (`ChannelAdapter`, `SpatialAdapter`, `FeatureMimicLoss`, `ResponseDistillationLoss`, `DistillationTrainer`, `UltralyticsYOLOLoss`, `DistillationOnlyLoss`, `create_yolo_loss`), same `MockModel` self-test; the student version adds the pseudo-RGB cache, the metrics accumulator, and the Ultralytics-format conversion.
- `student_repo/Cycle_gan/cyclegan.py` is a different (shorter, line-commented) CycleGAN than `zer0_repo/training/cyclegan.py`.
- The student repo's notebooks and `Student/evaluate_thermal.py` defaults hard-code `/home/aryan_s2/Detection_with_Distillation/…` (Aryan Sood's account) and the inference notebook reports an **NVIDIA RTX A6000**; `python.sbatch` uses `~/abishek_env` on a SLURM cluster. `cache_pseudo_rgb.py` has commented-out code referencing `/home/aryan_s2/...`.
- **Inference:** the two repositories are one team's codebase (zer0-data = design/scaffold, Jan 2026; abishek-c367 = the trained distillation branch, May–June 2026), run on shared infrastructure. Nothing here suggests misconduct. **Open question for Abishek (must be answered before any manuscript):** who wrote which module, what is Aryan Sood's role, is there an existing manuscript or thesis, and what authorship is expected.

### 2.2 What the repositories do *not* contain

- No trained weights (`.pt`/`.pth` are git-ignored): no CycleGAN checkpoint (`Cycle_gan/cyclegan_epoch_45.pth` referenced in `config.py`), no teacher weights, no student checkpoints (`checkpoints_distill/distill_epoch_5.pth` is referenced in the inference notebook as existing on the cluster).
- No training logs, no `results.json`, no metrics of any run, no SLURM output.
- No FLIR ADAS v2 data (git-ignored), no M3FD, C3I or SWVI loaders. `zer0_repo/tools/setup_datasets.py` only prints download instructions.
- No training script for the `WeatherGate` MLP or the `RainRemover` LSRB in either repo (`grep` over `training/`, `tools/`, `main.py`: no references). No weather labels anywhere. **The gate and the rain remover are untrained scaffolding** (see `CODE_AUDIT_2026.md`, T08).
- No unit tests. No pinned environment (`zer0_repo/requirements.txt` has only lower bounds; the student repo has no requirements file). The notebook kernel was Python 3.12.11; the Ultralytics version is unknown.

## 3. Datasets and weights on this machine

Searched `/home`, `/mnt`, `/media`, `/data`, `/datasets` (depth ≤ 4) for `flir|m3fd|llvip|kaist|c3i|swvi|thermal|infrared`: **nothing found**. FLIR ADAS v2 requires registration at Teledyne FLIR (https://www.flir.com/oem/adas/adas-dataset-form/); it cannot be fetched by me. See `REPRODUCTION_PROTOCOL.md` for what must be obtained.

## 4. Compute on this machine (observed 2026-10-15 21:04)

- 2 × NVIDIA GeForce RTX 2080 Ti (11 GB each), driver 535.309.01 (CUDA 12.2). GPU 0 hosts the display (≈0.6 GB used); GPU 1 idle.
- 16 CPU cores, 62 GB RAM, **76 GB free disk** (92% full).
- Other user jobs running and not to be disturbed: `selection_resampling.py`, `structural_checks.py`, `hp_counts.py` (Loyola venv, ~100% CPU each) and a 16-worker joblib pool (`topoqspr` venv, niced).
- Audit environment: `~/gateir_audit/venv` (Python 3.10, PyTorch cu124 wheels, Ultralytics 8.4.178 — exact versions recorded in `results/*.json` under `env`).

## 5. Items that must come from the collaborator before Gate 0 can pass

1. Training logs (stdout of `train_distillation.py`, incl. the line `Loss references set — yolo: …, feat: …`) and the Ultralytics / PyTorch versions in `abishek_env`.
2. The checkpoints `checkpoints_distill/distill_epoch_*.pth` and `distill_final.pth`, the CycleGAN checkpoint, and the exact teacher weights used (`Teacher/yolov8l.pt` = stock COCO weights, or a fine-tuned file?).
3. Whether `Teacher/finetune_yolov8l.py` was ever run, on which labels (`coco2yolo.py` output?), and its `val_best` metrics.
4. Any evaluation output of `Student/evaluate_thermal.py` (`Student/results/eval_metrics.json`).
5. The FLIR ADAS v2 archive and its license acceptance; M3FD / C3I / SWVI if they were actually obtained.
6. Authorship/roles: Aryan Sood's contribution and whether a manuscript or thesis already exists.
