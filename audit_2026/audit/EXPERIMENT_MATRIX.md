# EXPERIMENT MATRIX (pre-registration draft) : RAW-penalty decomposition on FLIR ADAS v2

**Status: DRAFT, not frozen.** Rewritten 2026-10-15 to match `paper/DRAFT.md` and the council decision in `paper/COUNCIL_RECORD.md` §6. It supersedes the earlier KD / weather-routing matrix (B0 to EX, P1, P2); the oracle-headroom protocol note from that draft (seed-split correction of a per-frame oracle) is kept as an internal protocol improvement and is not an arm here. Freeze after (i) the FLIR ADAS v2 archive is on disk and the day-1 checks pass, (ii) the split is verified for video overlap and TIFF/JPEG stem pairing, (iii) the pilot on a quarantined video-grouped subset of `images_thermal_val` confirms the float loader, the maps, the 8-bit path, the pretrained path and the learned mapper run end to end. The pilot never sets thresholds or the primary metric; it decides only the seed count for the two primary cells.

Statistics rules: `audit/MATH_STAT_REVIEW.md` §4. Prior-art rows the arms tie to: `paper/COUNCIL_RECORD.md` §3.1.

## 0. Fixed protocol (applies to every arm)

- Data: FLIR ADAS v2 thermal, 3 classes (person, car, bicycle) with an explicit FLIR category map; `images_thermal_train` for training, `images_thermal_val` for monitoring and for fixing the non-inferiority margin, **`video_thermal_test` frozen for the final test and opened once**. Video-id grouping verified across all three splits; manifest with SHA-256, bit depth, max count and video id per file.
- Input: the paired 14-bit TIFF `[container bit depth UNVERIFIED until day 1]`, mapped per arm (§1), letterbox 512, pad 0 in the mapped domain. Every arm, including the 8-bit arms and the vendor JPEG, is fed as a float32 tensor through one patched loader; the 8-bit arms take values in {k/255}. (Amended from the earlier matrix, which fixed the input to the 8-bit AGC JPEG.)
- Detector for all arms: stock Ultralytics `yolov8s.yaml`, version 8.4.178 pinned; `ch=1` from scratch, or COCO `yolov8s.pt` with 3 channels on grey-replicated input for the initialisation factor. The audited repository's custom student is not used.
- Training: same augmentation for every arm (Ultralytics geometric defaults with hsv_h = hsv_s = 0, plus one brightness/contrast jitter in the mapped domain, logged), same epochs (50), batch `[PENDING from pilot, 11 GB]`, optimizer, schedule and imgsz; seeds {0,1,2} (plus {3,4} for the two primary cells). Final-epoch EMA weights (`last.pt`); no best-on-val selection.
- Metrics: pycocotools mAP50-95 (primary), AP_small, AP50, per-class AP; conf 0.001, NMS IoU 0.7, max 300 det; evaluation in eval mode with a test that running statistics are not mutated (audit C5). Batch-1 latency on RTX 2080 Ti fp32 (FP16/TensorRT only if available), including the map.
- Uncertainty: paired cluster bootstrap over videos (B = 1000, identical resampled video index sets for every arm and seed, duplicated frames re-indexed) + across-seed SD. All seeds reported, including diverged runs. The number of video clusters K and the frame count are stated beside every CI.
- Decision rule for "X differs from Y": paired bootstrap 95% CI of the seed-averaged ΔmAP50-95 excludes 0 **and** |Δ| > δ, where δ is the pooled across-seed SD of the two primary cells on `images_thermal_val`, fixed in the OSF record before the test split is opened. Non-inferiority of X to Y: CI lower bound > −δ. If the CI half-width h exceeds δ the statement is "inconclusive" (a pre-registered third outcome).
- Label-free covariates computed for every frame: per-frame 14-bit dynamic range DR(x) = q99(x) − q1(x) (terciles fixed on the test set before unblinding); per-frame isotonic AGC transfer function g_x and its fit residual r_x; frame-to-frame map movement m_t; clipped-pixel fraction under 8-bit requantization.

## 1. Factors and arms

**A (adaptivity, applied to the 14-bit TIFF; dataset statistics from `images_thermal_train` only, 10k sampled pixels per frame, fixed seed):**

| A | Map | Per-frame parameters | Ties to |
|---|---|---|---|
| A0 | dataset-wide affine (x − m_D) / s_D, m_D dataset median, s_D dataset 1-99% span; no clipping at float, clip + round at 8-bit | 0 | foldable into the first conv; the "fixed affine" of the title |
| A1 | per-frame median offset (x − med(x)) / s_D | 1 | FFC-drift control |
| A2 | per-frame 1-99% clip to [0,1] | 2 | N-AGC "RAW-stretch" before its JPEG step; TCNet "clip" |
| A3 | camera AGC applied losslessly: per-frame isotonic (PAVA) fit of JPEG value on TIFF count from the paired files, 10k stratified pixels, applied to the TIFF in float; r_x logged | monotone curve | the primary comparator; FLIR AGC without JPEG loss |
| A4 | vendor AGC JPEG / 255 | (vendor) | descriptive tie-in to TCNet / Fieldscale / N-AGC tables; never the primary comparator (Bhowmik 2022) |
| A5 | learned per-frame mapper (TCNet re-implementation from the published description; N-AGC if code is released), trained jointly on the detection loss, retrained per arm | network | the learned reference, 3 seeds |

**Q (precision):** float32 vs 8-bit uniform requantization Q_8(u) = round(255 clip(u,0,1)) / 255 of the identical mapped image.

**I (initialisation):** ch=1 YOLOv8-s from scratch vs COCO-pretrained 3-ch YOLOv8-s on grey-replicated input.

## 2. Cells

| ID | A | Q | I | Seeds | Purpose | Compute (RTX 2080 Ti, est.) | Status |
|---|---|---|---|---|---|---|---|
| C1 | A0 | float | scratch | 5 | H1 arm; H2; H3(b); H4a | 5 × ~6 h = 30 | not run |
| C2 | A0 | 8-bit | scratch | 3 | H2 planted positive control; H4b | 18 | not run |
| C3 | A1 | float | scratch | 3 | FFC-drift control; H4a; falsifier | 18 | not run |
| C4 | A1 | 8-bit | scratch | 3 | H4b | 18 | not run |
| C5 | A2 | float | scratch | 3 | classical per-frame map at float; H4a | 18 | not run |
| C6 | A2 | 8-bit | scratch | 3 | H4b; N-AGC RAW-stretch without JPEG | 18 | not run |
| C7 | A3 | float | scratch | 5 | H1 comparator; H4a; S1 | 30 | not run |
| C8 | A3 | 8-bit | scratch | 3 | H4b; vendor map without JPEG | 18 | not run |
| C9 | A5 | float | scratch | 3 | H1 secondary comparator; H4a | 18 + mapper overhead `[PENDING]` | blocked on mapper re-implementation |
| C10 | A0 | float | COCO 3-ch | 3 | H3(a), H3(b) | 3 × ~5 h = 15 | not run |
| C11 | A0 | 8-bit | COCO 3-ch | 3 | H3 descriptive 2×2 | 15 | not run |
| C12 | A3 | float | COCO 3-ch | 3 | H3(a) comparator | 15 | not run |
| C13 | A4 | 8-bit (native) | scratch | 3 | descriptive tie-in; S1 | 18 (not in the council's 235 GPU-h) | not run |
| X1 | A0, augmentation off | float | scratch | 1 | R7 sanity | 6 (not in the council figure) | not run |
| X2 | A3, augmentation off | float | scratch | 1 | R7 sanity | 6 (not in the council figure) | not run |
| O1 | x / (2^14 − 1), i.e. TCNet's RAW row | 8-bit | scratch | 1 | optional: reproduce the published RAW cell's container | 6 | optional, unbudgeted |
| L0 | A4 through the float loader vs A4 through the native uint8 loader | n/a | scratch | pilot | loader is not a confound | pilot only | not run |
| PIL | pilot on a quarantined video-grouped subset of `images_thermal_val` (≈200 train frames from ≤10 videos, 50 held-out frames from 3 other videos, 2 epochs) | all paths | both | 1 | end-to-end check; CI half-width for the 3-vs-5 seed decision | < 2 h | not run |

Remaining pretrained cells (A1, A2, A5 × COCO) are optional and run only if time allows.

## 3. Hypotheses, predictions, falsifiers (frozen before `video_thermal_test` is opened)

Notation: Δ̂ seed-averaged point estimate; [L, U] paired cluster-bootstrap 95% CI; δ the pre-fixed margin (§0).

- **H1 (primary, non-inferiority).** Δ1 = mAP50-95(C1) − mAP50-95(C7). Non-inferior if L > −δ. **Falsified** if Δ̂1 < −δ and U < 0. Inconclusive otherwise. The same rule is applied and reported alongside for C1 − C9 (A0 vs the learned mapper). Falsifier in full: A0-float (and A1-float) trail A3-float or A5-float by more than δ with the paired CI excluding 0, concentrated in the low-DR tercile. Then per-frame histogram adaptivity is a real detector requirement at float precision, TCNet / N-AGC are justified, and the paper becomes the first multi-seed, CI-bearing, precision-controlled confirmation of that.
- **H2 (precision main effect; planted positive control).** Δ2 = mAP50-95(C1) − mAP50-95(C2). Confirmed if Δ̂2 ≥ 5 and L > 0. No null from H1 or H4 is interpreted as evidence of equivalence unless H2 is confirmed; the CI half-width h and δ are reported next to every null as the minimum detectable effect. If L > 0 but Δ̂2 < 5: quantization is real but smaller than the published RAW penalty, title attribution weakened. If the CI covers 0: quantization is not the mechanism; reported.
- **H3 (initialisation × precision interaction; Lynred isolated).** (a) Δ3 = mAP50-95(C10) − mAP50-95(C12), δ_pre = pooled val across-seed SD of C10 and C12. Confirmed if Δ̂3 < −δ_pre and U < 0. (b) The CI of (Δ1 − Δ3) excludes 0. C11 completes the 2×2 at A0 descriptively. **Falsified** if the pretrained net shows no float penalty: reported as a non-replication of Lynred (arXiv 2407.04484) on FLIR; the scratch cells still identify quantization vs adaptivity.
- **H4 (adaptivity buys nothing at float; DR interaction only at 8 bits).** (a) All ten pairwise seed-averaged differences among C1, C3, C5, C7, C9 have |Δ̂| ≤ δ; CIs reported uncorrected and labelled a descriptive family. (b) Low-minus-high DR-tercile mAP gap is negative for C2 (A0-8bit) and non-negative for C6 and C8 (A2-8bit, A3-8bit); tercile gaps of all float cells within δ of one another. Tercile subsets use the same video-cluster bootstrap with tercile membership fixed per frame.
- **S1 (descriptive secondary, gated).** Gate: on `video_thermal_test`, m_t > max(r_t, r_{t−1}) on ≥ 10% of consecutive frame pairs. Gate fails → reported as "FLIR's shipped AGC is temporally damped on this split"; no flicker claim. Gate passes → for C1, C7, C13 with per-arm, per-seed operating point chosen on val by maximal F1: flicker rate (GT object detected at t−1 and t+1 but not t, over interior track frames), per-object confidence SD along GT tracks of length ≥ 5, and the dose-response of both against m_t. GT tracks from annotation track ids if present `[UNVERIFIED]`, else greedy IoU ≥ 0.5 linking (label-derived, stated). Cluster-bootstrap CIs labelled descriptive; video count stated.

Multiplicity: one primary contrast (H1, C1 vs C7). H2 and H3 are single supporting contrasts. H4a/H4b and S1 are descriptive families, uncorrected, never headline.

## 4. Budget

Council-budgeted cells C1 to C12: 31 scratch runs × ~6 h + 9 pretrained runs × ~5 h ≈ 235 GPU-h ≈ 5 to 6 days on two RTX 2080 Ti. With C13, X1, X2: ≈ 265 GPU-h. Storage: ~17 GB TIFF + ~16 GB JPEG, no pseudo-RGB cache, cached mapped arrays `[size PENDING; may be regenerated on the fly if disk is tight]`, checkpoints ~0.1 GB each; 76 GB free at audit time. Isotonic fits: scikit-learn on ~10k sampled pixels per frame, CPU, all splits, hours. Any arm > 2 h requires explicit approval before launch. Use GPU 1; `num_workers ≤ 4` (shared CPU).

Timeline: week 1 data verification + day-1 checks + pilot; week 2 freeze + OSF pre-registration (δ, tercile boundaries, decision rules, analysis code); weeks 3 to 4 runs; weeks 5 to 7 single opening of `video_thermal_test`, analysis, writing.

## 5. Day-1 checks and kill conditions

1. TIFF present and stem-paired with JPEG for every frame of every split; bit depth and max count read from headers. **Hard kill for the primary split** if TIFFs are missing or unpaired for `video_thermal_test` → fallback: hold out whole videos from the stills by file-name video id as the frozen test set, reported as a pre-registered deviation.
2. No video id in more than one split; if the official still splits overlap, carve a video-grouped val from train (official val kept as a second monitoring set).
3. Annotation integrity: boxes inside the image, categories present, counts per class vs the dataset card.
4. Licence text recorded verbatim; release plan (manifests, hashes, code, tone curves, detections, no images) checked against it `[UNVERIFIED]`.
5. Per-frame r_x distribution on train: if the global isotonic fit does not reproduce the JPEG (large r_x), A3 is relabelled "global component of the vendor map" before the freeze and the C1 vs C9 contrast is declared the adaptivity carrier.

## 6. Planned failure cases to report

- F1 TIFFs missing/unpaired for the video split → stills-video-holdout fallback, pre-registered deviation.
- F2 vendor map not a global monotone curve (FLIR DDE / local contrast) → A3 relabelled; r_x distribution reported as a camera finding; A0 vs A5 carries the adaptivity claim.
- F3 few video clusters → K stated in every CI; pilot h decides 3 vs 5 seeds for C1/C7; h > δ → "inconclusive".
- F4 learned mapper code unavailable or > 11 GB → re-implementation labelled "unreproduced numbers"; if it still fails, A5 dropped, claim narrowed to classical maps, A2 stands in only for the shape of the DR interaction.
- F5 H3 fails → non-replication of Lynred on FLIR; title loses "and pretraining mismatch".
- F6 A1 beats A0 by > δ → "zero-parameter" becomes "one-parameter".
- F7 X1/X2 differ from C1/C7 by > δ in A3's favour → augmentation implicated; equality claim restricted to "under standard augmentation".
- F8 H2 fails → quantization attribution weakened or withdrawn; pretraining term reported from C10 to C12.
- F9 S1 gate fails → damped AGC reported as the measurement; no flicker claim.
- F10 H1 fails → paper rewritten around the confirmed adaptivity requirement; tables unchanged.
- F11 licence forbids per-frame curve release → per-video summaries only.
