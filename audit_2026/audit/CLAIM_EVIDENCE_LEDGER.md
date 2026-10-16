# CLAIM–EVIDENCE LEDGER — GATE-IR (15 Oct 2026)

Each narrative claim that appears in the slides, the summary PDF, the READMEs, the brief, or this audit is listed with its evidence class and pointer. Classes: **PDF claim**, **README claim**, **code observation**, **static concern**, **tested result**, **external published result**, **hypothesis**, **inference**.

| # | Claim | Class | Evidence pointer | Status |
|---|---|---|---|---|
| L1 | "CycleGAN conversion, detection and teacher–student distillation have been experimented" (collaborator's message) | PDF claim / collaborator report | Notebook cell output shows `checkpoints_distill/distill_epoch_5.pth` loaded on an RTX A6000 (`Notebooks/thermal_inference.ipynb`) | Partially corroborated: a run reached ≥ 5 epochs; no metrics, logs or weights seen |
| L2 | "Complete weather-gated GATE-IR pipeline has not been experimentally validated" | collaborator report + code observation | zer0 repo: untrained gate/LSRB, no labels (`CODE_AUDIT` C9, T08) | Confirmed |
| L3 | "YOLOv8-Thermal adds a Vision Transformer neck" (GATE-IR_Summary §3.3, README) | PDF/README claim | `train_distillation.py:107` `use_transformer_neck=False` | **Not true for the trained model** |
| L4 | "Detect small/distant objects more accurately with a modified architecture" (Summary §2) | PDF claim | No evaluation exists; P2-on-YOLOv8-FLIR-v2 published by YOLO-APDM (matrix A10) | Unsupported; not novel |
| L5 | "Transfer knowledge from RGB models … improving detection accuracy without large labelled thermal data" (Summary §3.4) | PDF claim | No B1-vs-B2 comparison exists; CGDet/FreqKD/InfraNet (A2, A4, A9) do this with stronger controls | Unsupported; routine |
| L6 | "Achieve all of this in real-time" (Summary §2) | PDF claim | T07: 29.7 ms batch-1 fp32 on RTX 2080 Ti for the student alone, random weights, no NMS; no target device | Unsupported; "real-time" undefined |
| L7 | "LSRB uses depthwise separable convs, ~8–9× faster" (Summary §3.2) | PDF claim | Correct arithmetic for 3×3, 64-ch; irrelevant since LSRB is untrained (C9) | Arithmetic OK, no detection evidence |
| L8 | "Fog scatters IR radiation reducing contrast; rain reduces temperature gradients" (Summary §1) | PDF claim | Physics literature: fog → contrast loss toward path radiance, droplet-size dependent (matrix §D); rain in LWIR = volumetric attenuation, no streak evidence | Fog claim defensible as contrast loss; rain-streak model not supported |
| L9 | Teacher is "YOLOv8-Large pre-trained on RGB" | README claim / code | `config.py:3` `Teacher/yolov8l.pt`; fine-tune script exists, unknown if run | Unknown which weights were used |
| L10 | The distillation code trains a detector (loss path works) | tested result | T01 (finite loss/grads), T02 (memorises toy batch, both decoders recover boxes) on Ultralytics 8.4.178 | Confirmed on this version |
| L11 | The inference notebook shows the trained student's detections | README/notebook claim | T04: notebook loads no weights (C1) | **Refuted** |
| L12 | Cache writer and loader disagree on layout (brief §3.2) | static concern | T05 | Partially: consistent for FLIR; mkdir/resume defects |
| L13 | Cache-missing training passes None to the teacher (brief §3.1) | static concern | T05 | Confirmed as crash |
| L14 | metrics.py / evaluate_thermal.py AP computations are wrong (brief §3.5) | static concern | T03 vs pycocotools | Largely refuted; two boundary quirks; eval-mode bug C5 is the real issue |
| L15 | Every GATE-IR component has a published precedent | external published result | Matrix A1, A2, A4, A9, A10, A11, B1–B6 (links) | Confirmed by verified sources |
| L16 | FLIR ADAS v2 has no per-frame weather labels | external (dataset page) | Matrix §C | Confirmed (official page); press material mentions fog/rain only |
| L17 | Real-weather thermal evaluation is possible only on UNIRI-TID / ZUT-FIR-ADAS / DENSE | external | Matrix §C | Confirmed; single-class, different cameras |
| L18 | Thermal-Det, TherA (CVPR 2026), the IPT 2026 review and the Sensors 2026 KD survey exist as cited in the brief | external | Matrix header | Confirmed |
| L19 | Student latency "≈ YOLOv8s" | inference | T07: 29.7 vs 26.0 ms; 21.5 M vs 11.2 M params | Student is ~2× the parameters of YOLOv8s |
| L20 | The two repositories belong to one team | inference | `INPUTS_AND_PROVENANCE` §2.1 (96-line diff; `/home/aryan_s2` paths; A6000 in notebook) | Strong inference; needs the collaborator's confirmation |
| L21 | Adaptive gamma addresses fog | PDF claim | T08: γ depends on mean only; low-contrast mean-0.5 image unchanged | Refuted for the stated degradation |
| L22 | An unpaired pseudo-RGB teacher adds information beyond a thermal teacher | hypothesis (H_KD) | No experiment; FreqKD/InfraNet use paired RGB; CGDet unclear | Open; the one question with room (matrix §E.4) |
| L23 | Cheap enhancement has oracle headroom on thermal detection | hypothesis (H_ENH) | No experiment | Open; go/no-go for routing |
| L24 | The 2ε regret bound and the DPI are contributions | brief's suggestion | `MATH_STAT_REVIEW` §2–3 | Refuted as contributions |
| L25 | Forensics proposal (DDIM + contrastive) is novel | PDF claim | `SECOND_PROJECT_FEASIBILITY` (DRCT ICML 2024) | Refuted |

## Added 16 Oct 2026 (independent method review)

| # | Claim | Class | Evidence pointer | Status |
|---|---|---|---|---|
| L26 | "C1 − C2 isolates 8-bit rounding" (matrix v2, draft rev. 2) | design claim | Review P0: A0 was unclipped at float and clipped at 8-bit, so the contrast bundled clipping. Fixed: A0 clipped at both precisions; clipping measured by C21 (A0u); `preproc_controls.py` + T09 show each step is one operation | **Was wrong; corrected** |
| L27 | "C2 − C14 is the quantization container term" | design claim | Both arms are 8-bit; the difference is range allocation / mapping (plus saturation). Relabelled; no "quantization" wording unless Δ2a's own CI supports it | **Relabelled** |
| L28 | "float32 input preserves more information than 8-bit" | design claim | T09: U has no more distinct levels than the counts; the intervention is the absence of rounding after a specified map | Reworded in abstract and matrix |
| L29 | "C10 − C12 isolates the pretraining mismatch (Lynred)" | design claim | Review P0: compares two maps inside one initialisation. Fixed: H3(a) relabelled map sensitivity; H3(b) DiD within 3-ch with new cell C22 | **Was over-claimed; corrected** |
| L30 | "A1 per-video affine is deployable" | design claim | Offline whole-video statistics are non-causal; A1 now defined causal (`arm_A1_causal`, T09) | Corrected |
| L31 | "Track D is inference-only, < 1 GPU-day" (follow-up 16 Oct) | planning claim | Requires a trained detector (calibration run C13 seed 0, ≈ 12 GPU-h) and enhancer tuning on development videos | **Overstated; corrected** in PUBLICATION_OPTIONS |
| L32 | "Pseudo-RGB plus cross-modal KD into a thermal-only student is novel" | novelty claim | Yoon, Park, Paik, IEEE Access 2026 (DOI 10.1109/ACCESS.2025.3650060; abstract verified via Consensus: FLIR Aligned + LLVIP, MMD + response distillation, 40.4 % mAP, thermal-only inference) | **Not novel**; only the four-arm teacher-modality control question remains open; Yoon's ablation rows (reviewer: baseline 38.0, response-KD 36.8, MMD 39.4) UNVERIFIED, full text not accessible from this machine |
| L33 | Bootstrap: "no seed pairing across cells is assumed" (draft rev. 2 §4.2) | design claim | Seeds are paired by index; joint seed draw adopted; independent draw kept as sensitivity | Corrected |
