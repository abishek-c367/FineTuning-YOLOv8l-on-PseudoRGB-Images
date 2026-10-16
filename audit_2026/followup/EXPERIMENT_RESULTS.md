# EXPERIMENT RESULTS (16 Oct 2026)

Rule: a row appears here only for a completed run with an immutable configuration file, a seed, a checkpoint SHA-256, the evaluator used and the comparison it belongs to. Everything else is **NOT RUN**. Smoke tests on synthetic batches are listed separately and are not detector experiments.

## 1. Detector experiments

| Track / cell | Config | Seed | Checkpoint hash | Metric | Status |
|---|---|---|---|---|---|
| Track A arm 1, no teacher (B0') | — | — | — | — | **NOT RUN** (no FLIR ADAS v2 on this machine) |
| Track A arm 2, thermal-trained teacher | — | — | — | — | **NOT RUN** |
| Track A arm 3, COCO RGB teacher on grey-replicated thermal | — | — | — | — | **NOT RUN** |
| Track A arm 4, CycleGAN pseudo-RGB teacher (B1) | — | — | — | — | **NOT RUN** (also blocked: `cyclegan_epoch_45.pth` not received) |
| Track B artefact study | — | — | — | — | **NOT RUN** |
| Track C P2 / neck comparison | — | — | — | — | **NOT RUN** |
| Track D enhancement oracle | — | — | — | — | **NOT RUN** |
| Track E temporal consistency | — | — | — | — | **NOT RUN** |
| RAW-penalty matrix C1 to C17, X1, X2, O1, L0, PIL | — | — | — | — | **NOT RUN** |
| Collaborator's reported B1 run | reported in the project summary PDF | unknown | none received | reported by the collaborator, evaluator unknown (possibly `evaluate_thermal.py` in train mode, audit C5) | **NOT REPRODUCED**; no log, checkpoint or config received |

No number from the project slides or summary PDF has been reproduced. No number in `paper/DRAFT.md` is a result; all are marked `[PENDING EXPERIMENT]`.

## 2. Smoke and regression tests (synthetic batches, not detector experiments)

Raw outputs with environment and commit hashes: `results/T0*.json` (original code, 15 Oct 2026) and `results/rerun_patched/` (branch `audit/2026-10`, 16 Oct 2026). Environment: Python 3.10.12, torch 2.6.0+cu124, ultralytics 8.4.178, RTX 2080 Ti, GPU 1.

| Test | What it establishes | Original code | Patched branch |
|---|---|---|---|
| T01 loss path | loss/backprop runs on the pinned Ultralytics; parameter counts (student 21.5 M with P2, no ViT) | pass | see `rerun_patched/SUMMARY.txt` |
| T02 one-batch overfit | loss decreases on a synthetic batch; both decoders recover the planted boxes (IoU > 0.99) | pass (5498 → 1.94) | pass with bias init (23.8 → 1.46) |
| T03 metrics vs pycocotools | repository AP matches pycocotools except two edge cases | pass with noted exceptions | see summary |
| T04 checkpoint loading | notebook loader loads zero trained weights (C1); patched loader equals trained state | characterises defect | pass |
| T05 cache paths | characterises the original cache writer/loader (C2, C3) | characterises defect | **fails by design** (superseded by T05b) |
| T05b cache paths, patched | recursive listing, resume/skip under `data/`, mkdir, loader resolution, cache-miss and no-cache behaviour, live fallback shape/range | n/a | **pass** (`results/T05b_cache_paths_patched.json`, `all_pass: true`) |
| T06 edge cases | single-GT-box crash (C4), train-mode eval drift (C5), class maps (C6) | characterises defects | see summary |
| T07 latency | batch-1 512 fp32 latency table | pass (measurement) | unchanged |
| T08 zer0 gate | weather gate has no trained weights; outputs are random-init | characterises scaffold | unchanged |
| T09 preprocessing factor isolation (16 Oct, review P0) | U/V/W/Z differ by exactly one declared operation each; rounding error ≤ half a step; float adds no levels; full-range 8-bit collapses the scene band (78 vs 256 levels on the synthetic scene); causal A1 unchanged by future frames; affine folds exactly into conv1 | n/a | **pass** (`results/T09_preproc_factor_isolation.json`, `all_pass: true`) |
| T10 manifest verifier (16 Oct, review P1) | clean synthetic dataset passes; eight planted violations (shared video id, TIFF without JPEG, JPEG dimension mismatch, exact duplicate, near-duplicate, box outside image, undeclared category, count above the 14-bit container) each flagged, exit 1 | n/a | **pass** (`results/T10_manifest_verifier.json`, `all_pass: true`) |

T09 and T10 are static tests on synthetic arrays; they establish nothing about detector performance. These tests show that the code paths execute and that the patches change the behaviours they target. They do not show that any configuration trains to a useful detector on FLIR ADAS v2.
