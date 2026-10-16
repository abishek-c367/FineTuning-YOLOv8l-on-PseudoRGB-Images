# EXPERIMENT MATRIX (pre-registration draft) — GATE-IR

**Status: DRAFT, not frozen.** Freeze after (i) the collaborator's artefacts arrive, (ii) the FLIR split is verified for video overlap, (iii) the pilot on a quarantined subset confirms the pipeline runs end-to-end. The pilot never sets thresholds or the primary metric.

## 0. Fixed protocol (applies to every arm)

- Data: FLIR ADAS v2 thermal, 3 classes (person, car, bicycle) with explicit FLIR category map; `images_thermal_train` for training, `images_thermal_val` for model selection, **`video_thermal_test` frozen for the final test and opened once**. Video-id grouping verified; manifest with SHA-256 per file.
- Input: 8-bit AGC JPEG, letterbox 512, pad 0 (as in the repo). (Optional secondary: 640.)
- Student architecture for all KD arms: the repo's `yolov8s_thermal(include_p2=True, use_transformer_neck=False)` **and** stock Ultralytics `yolov8s.yaml` with `ch=1` (so results are comparable to the literature); same augmentation (Ultralytics defaults with hsv_h = hsv_s = 0), same epochs (50), batch, optimizer, imgsz, seeds {0,1,2} (5 if time allows).
- Metrics: pycocotools mAP50-95 (primary), AP_small, AP50, per-class AP; batch-1 latency on RTX 2080 Ti fp32 and FP16 (TensorRT only if available), including enhancement + routing time.
- Uncertainty: paired cluster bootstrap over videos (1000 reps) + across-seed SD. All seeds reported.
- Decision rule for "X helps": paired bootstrap 95% CI of ΔmAP50-95 excludes 0 **and** |Δ| > across-seed SD of the baseline.

## 1. Arms

| ID | Arm | Purpose | Compute (RTX 2080 Ti, est.) | Status |
|---|---|---|---|---|
| B0 | Stock Ultralytics YOLOv8s (ch=1) trained from scratch and from COCO init (first conv averaged) | Reality check; literature-comparable baseline | 2 × 3 seeds × ~6 h | not run |
| B0' | Repo student (P2, no ViT), detection loss only (α = 0) | Architecture baseline for KD arms | 3 seeds × ~8 h | not run |
| B1 | Repo student + CycleGAN pseudo-RGB → COCO YOLOv8-L teacher, feature mimic P3/P4, α = 0.2 (reproduces the collaborator's run) | Reproduce the original work | needs `cyclegan_epoch_45.pth` + cache (~10 GB) ; 3 seeds × ~10 h | blocked (weights) |
| B2 | = B0' (no KD, identical compute) | Is KD useful at all? | shared with B0' | — |
| B3a | Thermal-trained YOLOv8-L teacher (train on FLIR thermal, 3 classes) → same student, same mimic loss | Fair same-modality KD control | teacher 3 × ~12 h + student 3 × ~10 h | not run |
| B3b | COCO YOLOv8-L fed **grayscale-replicated thermal** (no CycleGAN) → same student | Isolates the translator's contribution | 3 × ~10 h | not run |
| B3c | COCO YOLOv8-L fed CycleGAN pseudo-RGB but with a **shuffled** pairing (features from a different frame) | Negative control: does the mimic loss act as a generic regulariser? | 3 × ~10 h | not run |
| B4 | Enhancement actions applied uniformly at test time to the best B0 model: identity / gamma / LSRB(untrained, random) / LCN / CLAHE / fixed gamma 0.7 and 1.4 | Does any enhancement help detection at all? | inference only, minutes | not run |
| B5 | **Oracle per-frame action** (labels used offline) over the action set of B4; report headroom H and the fraction of frames where any non-identity action wins | Go/no-go for the whole routing programme | inference only | not run |
| B6 | Routing controls: (a) run-all + max-mean-confidence pick; (b) random action; (c) the zer0 5-statistic MLP gate trained on *oracle labels* from the train split; (d) a small CNN router on oracle labels | Fair router comparison | hours | conditional on B5 showing H > seed SD |
| P1 | **Primary**: teacher-input ablation {B2, B3a, B3b, B3c, B1} on identical student/compute | Does unpaired CycleGAN pseudo-RGB transfer anything specific? | = B1 + B3 | conditional on B1 reproducing |
| P2 | Mechanistic ablation of P1's winner: distill P2 only / P3 only / P4 only / P3+P4; α ∈ {0.05, 0.2, 1} | What carries the signal | 3 × 4 × ~10 h | conditional |
| EX | External evaluation on M3FD (IR, 6→3 class map) and, for weather, UNIRI-TID / ZUT-FIR-ADAS person-only AP | Generalisation; the only real-weather evidence available | inference | needs data access |
| EFF | End-to-end batch-1 latency/memory for B0, B0', B1 winner, and the full gate+enhancer+detector chain | Deployment accounting | minutes | partial (T07, random weights) |

## 2. Hypotheses, predictions, falsifiers

- **H_KD (primary):** "An unpaired CycleGAN pseudo-RGB teacher improves the thermal student beyond a no-teacher baseline and beyond a same-modality thermal teacher." Prediction: mAP(B1) > mAP(B2) and mAP(B1) ≥ mAP(B3a) by the decision rule. Falsified if B1 ≈ B2, or if B3b (grayscale-replicated input, no translator) ≈ B1 (the translator is irrelevant), or if B3c ≈ B1 (the mimic loss is a generic regulariser).
- **H_ENH (gate for routing):** "At least one cheap enhancement action improves detection on a non-trivial subset of frames." Prediction: H (B5) > across-seed SD of B0 and the non-identity-wins fraction > 10%. Falsified otherwise, in which case all routing work stops and the negative result is reported.
- **H_ROUTE (only if H_ENH survives):** "A learned router recovers ≥ 50% of the oracle headroom at < 2 ms routing cost." Falsified if the run-all + confidence baseline (B6a) matches it, or if routing cost exceeds the latency of simply running the identity + best fixed action.

## 3. Budget

Rough total for B0, B0', B3a/b/c, P1 with 3 seeds at ~8–12 h per run on one RTX 2080 Ti: ~400–500 GPU-hours (~3 weeks on one GPU, ~10 days on both). Storage: FLIR ADAS v2 (~16 GB extracted), pseudo-RGB cache (~10 GB), checkpoints (~0.1 GB each). The disk has 76 GB free; feasible but tight. Any arm > 2 h requires explicit approval before launch.

## 4. Planned failure cases to report

- B1 does not reproduce because the CycleGAN checkpoint is unavailable → report B3b as the fallback "teacher on thermal" arm.
- LSRB is untrained → B4 includes it as a random-weight control only; a trained rain remover needs rain targets that do not exist for LWIR.
- No fog/rain frames in FLIR → the word "weather" appears only in the EX section.
