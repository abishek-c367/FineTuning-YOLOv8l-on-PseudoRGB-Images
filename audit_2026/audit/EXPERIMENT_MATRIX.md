# EXPERIMENT MATRIX (pre-registration draft) : RAW-penalty decomposition on FLIR ADAS v2

**Status: DRAFT, not frozen. Reconciled 2026-10-16 to `paper/DRAFT.md` revision 2 (§3 to §6, §10), then amended the same day after the independent method review (P0 mechanism attribution, P0 pretraining attribution, P1 statistics): A0 is clipped at both precisions so that C1 − C2 is a pure rounding intervention; the clipping intervention has its own cell C21 (A0u); C2 − C14 is relabelled range allocation at 8 bits; H3(b) is a difference-in-differences within the 3-channel architecture with the new cell C22; the bootstrap pairs seeds by index; A1 is causal. See `followup/REVIEW_RESPONSE_2026-10-16.md`.** The 2026-10-15 version of this file was written against revision 1 of the draft and the council record and disagreed with revision 2 on seven points. Revision 2 is authoritative; the council record (`paper/COUNCIL_RECORD.md`) is a historical record of the decision and is not edited. The seven reconciled points:

| # | Item | Council record / matrix v1 | Draft rev. 2 (now adopted here) | Where in the draft |
|---|---|---|---|---|
| 1 | Primary comparator for H1 | A3 (vendor curve, isotonic) | **A2 per-frame affine** (A3 and A5 secondary) | §4.5, §10 item 2, App. A item 3 |
| 2 | Non-inferiority margin δ | pooled across-seed SD of the primary cells, fixed on val | **δ = 1.0 mAP50-95, fixed now** | §4.4, App. A item 11 |
| 3 | Epochs | 50 | **100** (library default), val slope over the last 20 epochs reported | §3.4, §10 item 6, App. A item 8 |
| 4 | Photometric augmentation | one brightness/contrast jitter in the mapped domain | **none** (`hsv_h = hsv_s = hsv_v = 0`); jitter-on replicates J1/J2 optional | §5.2 |
| 5 | Seed counts | 3, raised to 5 for primary cells by the pilot | **fixed in advance**: 5 for C1 and C5, 3 elsewhere; the pilot sets nothing | §4.3, §5.5 |
| 6 | H2 positive control | Δ̂2 ≥ 5 mAP threshold | **no threshold**; control passes if L2 > δ; split into rounding (C1 − C2) and container (C2 − C14) terms | §4.5, §10 item 16 |
| 7 | Cell list and budget | C1 to C13 (+X1, X2, O1), ≈ 235 to 265 GPU-h at 50 epochs | **C1 to C17 in three tiers, 49 runs, ≈ 590 GPU-h at 100 epochs**, pre-registered drop order; per-run time measured by a calibration run (C13 seed 0) before the freeze | §3.7, §6.1 |

Freeze after (i) the FLIR ADAS v2 archive is on disk and the day-1 checks pass, (ii) the split is verified for video overlap and TIFF/JPEG stem pairing, (iii) the pilot on a quarantined video-grouped subset of `images_thermal_val` confirms every path runs end to end and fixes the batch size, (iv) the calibration run has measured the per-run time. The pilot sets no threshold, metric, seed count or margin.

Statistics rules: `audit/MATH_STAT_REVIEW.md` §4 and draft §4. Prior-art rows the arms tie to: draft §2.1.

## 0. Fixed protocol (applies to every cell)

- Data: FLIR ADAS v2 thermal, 3 classes (person, car, bicycle) with an explicit FLIR category map; `images_thermal_train` for training, `images_thermal_val` for monitoring and the pilot, **`video_thermal_test` frozen and opened in two registered stages** (pixel stage: manifest, K_test, DR tercile boundaries, A3 curves, m_t, S1 gate; label stage: once, after every budgeted run). Video-id grouping verified across all splits; manifest with SHA-256, bit depth, max count and video id per file.
- Input: the paired 14-bit TIFF `[container bit depth UNVERIFIED until day 1]`, mapped per arm (§1), letterbox 512, pad 0 in the mapped domain (library default fill overridden in every arm). Every arm, including the 8-bit arms and the vendor JPEG, is fed as a float32 tensor through one patched loader; 8-bit arms take values in {k/255}. Deterministic loader-equivalence test (draft §5.2) before any run.
- Detector: stock Ultralytics `yolov8s.yaml`, version 8.4.178 pinned; `ch=1` from scratch; `ch=3` from scratch on grey-replicated input (channel control); COCO `yolov8s.pt` 3-ch on grey-replicated input at the library's own 8-bit-grey input scale. The audited repository's custom student is not used.
- Training: identical for every cell. Library defaults for loss weights, optimiser, schedule, warm-up, EMA (logged verbatim per run); **100 epochs**; batch `[PENDING: pilot, 11 GB]`; geometric augmentation at library defaults (mosaic, scale, translate, flip; values logged); **no photometric augmentation** (`hsv_h = hsv_s = hsv_v = 0`, no jitter), because any per-image affine jitter is itself a per-frame map. Final-epoch EMA weights; no best-on-val selection. Val mAP50-95 curve released per run; slope over the final 20 epochs in the main table.
- Metrics: pycocotools mAP50-95 (primary), AP_small, AP50, per-class AP; conf 0.001, NMS IoU 0.7, max 300 det; eval mode with a test that BN running statistics are not mutated (audit C5). Batch-1 latency on RTX 2080 Ti fp32, including the map (descriptive column).
- Uncertainty: **two-way paired bootstrap (video × seed)**, B = 1000; one set of video index draws shared by every cell and seed; seeds are paired by index across arms (same seed → same data order, augmentation stream, initialisation), and each replicate draws seed indices **jointly** for the cells of a contrast from their common index set (5 for C1 vs C5; indices 0–2 when a 3-seed cell is involved, the extra seeds of a 5-seed cell entering only its SD and a sensitivity CI). The independent per-cell draw is reported as a sensitivity check. Across-seed SD reported for every cell. K_test and frame count stated beside every CI. Unit test: identity index set reproduces the full-set mAP exactly.
- Margin: **δ = 1.0 mAP50-95**, fixed (draft §4.4); a design choice, defended as such, not a demonstrated practically important difference. The calibration run's predicted h_test is written into the OSF record beside δ. Rules are CI-based only; failure to reject a difference is never reported as equivalence. Non-inferiority of X to Y: L > −δ. Inferior: U < −δ. Inconclusive otherwise, and always when the CI half-width h > δ (pre-registered third outcome).
- Label-free covariates per frame: DR(x) = q99(x) − q1(x) (terciles fixed at the pixel stage); isotonic vendor curve g_x, fit residual r_x, clipped fraction f_x, 8-bit rounding error e_x; frame-to-frame curve movement m_t (histogram-weighted RMS, draft §4.5 S1).

## 1. Factors and arms

**A (adaptivity; dataset statistics from `images_thermal_train` only, 10k sampled pixels per frame, fixed seed):**

| A | Map | Per-frame params | Ties to |
|---|---|---|---|
| A0 | dataset-wide affine (x − q1_D)/s_D, **clipped to [0,1] at both precisions**; round at 8-bit | 0 | the "fixed affine" of the title; float vs 8-bit differ by rounding alone |
| A0u | the same affine **unclipped**, float only | 0 | clipping intervention (C21); folds exactly into conv1 (T09) |
| A1 | per-video affine, **causal** (expanding-window percentiles over frames 0..t; `arm_A1_causal`) | per video | FFC / gain drift control; offline whole-video form is pilot diagnostic only |
| A2 | per-frame affine: 1-99 % stretch, clipped to [0,1] at both precisions | 2 | **primary comparator**; N-AGC RAW-stretch and TCNet clip before their 8-bit step |
| A3 | vendor curve recovered per frame by isotonic (PAVA) fit of JPEG value on TIFF count, applied at count resolution; r_x logged | monotone curve | secondary comparator; relabelled "global component of the vendor map" under F2 |
| A4 | vendor AGC JPEG / 255 | (vendor) | descriptive tie-in to TCNet / Fieldscale / N-AGC; never a primary comparator (Bhowmik 2022) |
| A5 | learned per-frame mapper (TCNet re-implementation; N-AGC if code is released), trained jointly on the detection loss, retrained per arm | network | secondary comparator, 3 seeds |
| R | TCNet's full-range linear scaling x/(2^14 − 1) | 0 | the published RAW container; H2 container term; H3(c) |

**Q (precision):** float32 vs 8-bit rounding Q_8(v) = round(255·v)/255 of the identical, already clipped, mapped image (`requant8` refuses unclipped input, so clipping can never hide inside the precision factor). Float32 adds no information beyond the counts (T09: no new levels); the intervention is the absence of rounding after a specified map. Transforms: `audit_tools/preproc_controls.py`.

**I (initialisation):** scratch 1-ch; scratch 3-ch (channel control); COCO-pretrained 3-ch.

## 2. Cells (draft §6.1; run in tier order, tier 1 complete before any tier-2 run starts)

| ID | A | Q | I | Seeds | Role | Tier | Est. GPU-h (12 h/run, PENDING calibration) | Status |
|---|---|---|---|---|---|---|---|---|
| C1 | A0 | float | scratch 1-ch | 5 | H1 arm; H2; H3(b),(d); H4 | 1 | 60 | not run |
| C5 | A2 | float | scratch 1-ch | 5 | **H1 primary comparator**; H3(b); H4 | 1 | 60 | not run |
| C2 | A0 | 8-bit | scratch 1-ch | 3 | H2 rounding term; H4b | 1 | 36 | not run |
| C14 | R | 8-bit | scratch 1-ch | 3 | H2 container term; positive control | 1 | 36 | not run |
| C10 | A0 | float | COCO 3-ch | 3 | H3(a),(b),(c) | 1 | 36 | not run |
| C12 | A2 | float | COCO 3-ch | 3 | H3(a) comparator | 1 | 36 | not run |
| C9 | A5 | float | scratch 1-ch | 3 | H1 secondary; H4a | 2 | 36 + mapper `[PENDING]` | blocked: mapper re-implementation |
| C7 | A3 | float | scratch 1-ch | 3 | H1 secondary; H4a; S1 | 2 | 36 | not run |
| C13 | A4 | 8-bit native | scratch 1-ch | 3 | descriptive tie-in; S1; **seed 0 = calibration run (pre-freeze)** | 2 | 36 | not run |
| C16 | A4 | 8-bit native | COCO 3-ch | 3 | H3(c): the published AGC setting | 2 | 36 | not run |
| C17 | R | float | COCO 3-ch | 3 | H3(c): the published RAW setting (float reading) | 2 | 36 | not run |
| C15 | A0 | float | scratch 3-ch | 3 | H3(d) channel control; H3(b) DiD | 2 | 36 | not run |
| C22 | A2 | float | scratch 3-ch | 3 | H3(b) DiD partner of C15 (review, 16 Oct) | 2 | 36 | not run |
| C21 | A0u | float | scratch 1-ch | 3 | clipping intervention Δ_C (review, 16 Oct) | 2 | 36 | not run |
| C3 | A1 | float | scratch 1-ch | 3 | slow camera-state control; H4a; F6 | 3 | 36 | not run |
| C6 | A2 | 8-bit | scratch 1-ch | 3 | H4b DiD; N-AGC RAW-stretch without JPEG | 3 | 36 | not run |
| C8 | A3 | 8-bit | scratch 1-ch | 3 | C13 − C8 = JPEG + local component (descriptive) | 3 | 36 | not run |

Tier 1: 22 runs, ≈ 265 GPU-h. All tiers: 55 runs (43 scratch + 12 pretrained), ≈ 660 GPU-h, ≈ 14 days on two RTX 2080 Ti or 28 on one (GPU 0 hosts the display; availability `[UNVERIFIED]`).

Optional, after tier 3, in this order: C11 (A0, 8-bit, COCO), C18 (A3, float, COCO), C19 (z-scored float counts, COCO; Lynred's exact input), C23 (A0u, float, COCO; pure input-scale bracket), C4 (A1, 8-bit, scratch), J1/J2 (C1 and C5 with photometric jitter on, 3 seeds; F7 sensitivity), C20 (COCO weights into a 1-ch stem `[UNVERIFIED library support]`), L0 (C13 through the native uint8 loader, 1 seed). X1/X2 (augmentation off) and O1 of matrix v1 are absorbed into J1/J2 and C14.

Pilot (PIL): quarantined video-grouped subset of `images_thermal_val` (≈ 200 train frames from ≤ 10 videos, 50 held-out frames from 3 other videos, 2 epochs, < 2 h): confirms the float loader, every map, the 8-bit path, both 3-ch paths and the mapper run end to end; pycocotools, loader-equivalence, bootstrap-identity and A3-rounding tests pass; fixes the batch size. Never a source of any registered number.

**Drop rule (pre-registered):** if the calibration run implies the full budget exceeds the window, tiers are dropped from the bottom (tier 3, then tier 2 from C15 upward); tier 1 is never cut; epochs and seed counts are never reduced to fit; dropped cells are listed as "not run (budget)".

## 3. Hypotheses and decision rules (draft §4.5 to §4.7; frozen before the label stage)

Notation: Δ̂ point estimate; [L, U] two-way bootstrap 95 % CI; h its half-width; δ = 1.0.

- **H1 (primary, non-inferiority).** Δ1 = μ(C1) − μ(C5). Non-inferior if L1 > −δ; inferior if U1 < −δ; inconclusive otherwise (always if h > δ). Secondaries, same rule, same table: C1 vs C7 (A3) and C1 vs C9 (A5). Discordance rule: the headline follows H1 only; a secondary that is inferior or inconclusive narrows the claim's scope and never reverses it.
- **H2 (requantization terms; planted positive control).** Δ2a = μ(C1) − μ(C2) (**rounding** within the dataset band; the arms differ by the rounding operation alone); Δ2b = μ(C2) − μ(C14) (**range allocation at fixed 8-bit output**; bundles any saturation difference between the maps, named accordingly); Δ2 = μ(C1) − μ(C14). Δ_C = μ(C21) − μ(C1) (**clipping**) reported beside them, outside the chain. Positive control passes if L2 > δ; no equivalence from H1/H4 is interpreted as equality unless it has passed. No threshold on any term. Attribution rule: "requantization, not adaptivity" is supported if L2 > max(|L1|, |U1|); weakened if the CIs overlap; withdrawn if U2 < δ. The word "quantization" is used for Δ2a only if its own CI excludes 0; if Δ2a is small and Δ2b large the effect is reported as range allocation (mapping) at 8 bits and the title changes before submission.
- **H3 (initialisation; Lynred isolated; published setting reproduced).** (a) Δ3 = μ(C10) − μ(C12): map sensitivity under pretrained initialisation; on its own it compares two maps within one initialisation. Lynred effect confirmed only if U3 < −δ **and** (b) holds. (b) difference-in-differences within the 3-channel architecture: Δ_DiD = (μ(C15) − μ(C22)) − (μ(C10) − μ(C12)), shared video and seed draws; pretraining × map interaction supported if the CI excludes 0 (replaces Δ1 − Δ3, which confounded channel count with initialisation). (c) published setting: ΔP = μ(C16) − μ(C17) with its two brackets, μ(C10) − μ(C17) being the pure input-scale term. (d) channel control: μ(C15) − μ(C1) with CI inside [−δ, δ]; otherwise factor I is reported as "initialisation plus channel count" and H3(a) is read against C15. H3 fails if L3 > −δ: non-replication of Lynred (arXiv 2407.04484) on FLIR.
- **H4 (adaptivity buys nothing at float; DR interaction only at 8 bits).** (a) TOST-style equivalence family over the ten pairs among C1, C3, C5, C7, C9: equivalent if [L, U] ⊂ [−δ, δ]; different if entirely outside; inconclusive otherwise; uncorrected, descriptive. (b) DR difference-in-differences: D_8 = G(C2) − G(C6) predicted negative with CI excluding 0; D_f = G(C1) − G(C5) predicted inside [−δ, δ]; G_c = mAP_c(low tercile) − mAP_c(high tercile); tercile membership fixed per frame.
- **S1 (descriptive, gated).** Gate at the pixel stage: m_t > max(r_t, r_{t−1}) on ≥ 10 % of consecutive test-frame pairs. Gate fails → "FLIR's shipped AGC is temporally damped on this split", no flicker claim. Gate passes → for C1, C7, C13 with per-arm, per-seed val-F1 operating point: flicker rate, per-object confidence SD along GT tracks of length ≥ 5, dose-response against m_t. Track ids from annotations if present `[UNVERIFIED]`, else greedy IoU ≥ 0.5 linking (label-derived, stated).

Multiplicity: one primary contrast (H1, C1 vs C5). H2 and H3(a to d) are supporting contrasts with their own rules. H1 secondaries, H4a, H4b and S1 are descriptive, uncorrected, never headline. Minimum detectable effect: δ, the across-seed SD of each cell and h are reported next to every null or equivalence statement.

## 4. Budget and resources (draft §3.7)

Per-run time `[PENDING: calibration run C13 seed 0]`; the only prior figure is the reproduction protocol's ≈ 6 GPU-h for 50 epochs of stock YOLOv8-s on this machine, extrapolated to ≈ 12 GPU-h at 100 epochs. Storage: ≈ 17 GB TIFF + ≈ 16 GB JPEG `[unverified until extracted]`; no cached float frames; A3 tables ≈ 64 kB/frame (order 1 GB); checkpoints ≈ 0.1 GB each, last epoch only, ≈ 5 GB. Isotonic fits: scikit-learn on 10k sampled pixels per frame, CPU, hours. Loader (16-bit TIFF decode, float mosaic, per-frame statistics) with ≤ 4 workers on the shared CPU may be the bottleneck; the calibration run measures it. **Any run > 2 h requires explicit approval before launch. GPU 1 only. Other users' CPU jobs are not to be disturbed.**

Timeline (draft §9): week 1 data verification, manifests, day-1 checks, K_test, A3 curves and label decision, pilot, batch size, calibration run started; week 2 calibration complete, freeze and OSF registration; weeks 3 onward tiered runs; then pixel stage, label stage once, analysis, writing.

## 5. Day-1 checks and kill conditions

1. TIFF present and stem-paired with JPEG for every frame of every split; bit depth and max count read from headers. Hard kill for the primary split if TIFFs are missing or unpaired for `video_thermal_test` → fallback F1.
2. No video id in more than one split; otherwise carve a video-grouped val from train (official val kept as a second monitoring set).
3. Annotation integrity: boxes inside the image, categories present, per-class counts vs the dataset card; exact (SHA-256) and near-duplicate (dHash) frames across train / pilot / val / test; TIFF header mode and max count; JPEG pairing by stem **and** dimensions; all by `audit_tools/verify_manifest.py` (test T10), exit 1 on any hard failure. The CycleGAN checkpoint's training list, if ever received, is checked against the held-out videos.
4. Licence text recorded verbatim; release plan (manifests, hashes, code, tone curves, detections; no images) checked against it `[UNVERIFIED]`.
5. Per-frame r_x, f_x, e_x distributions on train; A3 label rule (F2) applied before the freeze.
6. Loader-equivalence test passes; bootstrap identity test passes; A3 rounding test passes.

## 6. Planned failure cases to report (draft §7)

- F1 TIFFs missing/unpaired for the video split → whole-video holdout from the stills as the frozen test set; pre-registered deviation.
- F2 vendor map not a global monotone curve → A3 relabelled "global component of the vendor map"; r_x reported as a camera finding; C1 vs C5 remains primary.
- F3 few video clusters → K_test stated in every CI; h > δ → "inconclusive".
- F4 learned mapper unavailable or > 11 GB → re-implementation labelled "unreproduced numbers"; if it still fails, C9 dropped and the claim narrowed to classical maps.
- F5 H3 fails → non-replication of Lynred on FLIR; the title loses "and pretraining mismatch".
- F6 A1 superior to A0 by the H4a rule → mechanism statement adds "after a per-video affine".
- F7 J1/J2 differ from C1/C5 by > δ → augmentation implicated; claim restricted to "under geometric augmentation only".
- F8 H2 positive control fails → no equivalence statement interpreted; quantization attribution withdrawn; pretraining term still reported from C10/C12/C16/C17.
- F9 S1 gate fails → damped AGC reported as the measurement; no flicker claim.
- F10 H1 fails → paper rewritten around the confirmed adaptivity requirement; tables unchanged.
- F11 licence forbids per-frame curve release → per-video summaries only.
- F12 budget window shorter than the calibration-run extrapolation → drop rule (§2) applied; dropped cells listed.
