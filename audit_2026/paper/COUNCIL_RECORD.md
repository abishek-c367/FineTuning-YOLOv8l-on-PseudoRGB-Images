# COUNCIL RECORD — single-paper decision (2026-10-15)

Chair's record of the final council round. Inputs: two surviving candidates with verified
novelty checks and refutation reviews, three refuted-earlier candidates, and three lens
scorecards. Output: ONE paper to develop, with the design changes required to make it
survive its own refutation, and the risks that remain.

Project context: `/home/mpcrlab/gateir_audit/` (audit of the GATE-IR student repo; statistics
rules in `audit/MATH_STAT_REVIEW.md` §4; prior-art matrix in `audit/PRIOR_ART_NOVELTY_MATRIX.md`;
verified harness results T01–T08 in `results/`). Hardware: 2x RTX 2080 Ti (11 GB), ~76 GB free
disk. Pinned stack: Ultralytics 8.4.178, pycocotools-agreeing `metrics.py` after patches 0005/0006.

---

## 1. Decision

**Chosen paper: Candidate 1, rebuilt around the reviewer's strongest fix.**

**Working title.** *Float precision, not per-frame adaptivity: the "RAW penalty" in 14-bit LWIR
object detection is quantization and pretraining mismatch, and no tone map beats a fixed affine
once both are controlled.*

**One-sentence claim (to be filled only from results).** On a video-clustered 3-class FLIR ADAS
v2 protocol with >=3 seeds, a zero-parameter dataset-wide affine fed as float32 to a
from-scratch 1-channel YOLOv8-s is non-inferior (margin = across-seed SD) to the camera's own
per-frame AGC applied losslessly and to classical and learned per-frame tone maps retrained per
arm; the published "RAW penalty" (TCNet 37.9 vs 53.5) reappears only when the same map is
requantized to 8 bits or fed to a COCO-pretrained 3-channel backbone.

**Why one paper, and why this one.** Both survivors were refuted on contribution/identifiability,
not feasibility. Candidate 1's defects are all repairable on the same dataset for ~55 extra
GPU-h (add a detector-initialisation factor; make the AGC comparator lossless; give the learned
reference seeds; demote flicker). Candidate 2's defect is a confound between modality and
degradation state whose repair needs degraded-LWIR data (ZUT-FIR-ADAS 95 GB vs 76 GB free,
UNIRI-TID one 13-minute static-camera fog session) that the three earlier refutations already
showed cannot support recording-clustered inference. All three lenses ranked Candidate 1 first
(21/23/21 vs 19/18/20). The chair concurs and adds: Candidate 1 is the only one of the two whose
negative outcome is still a table a thermal-camera integrator would cite.

---

## 2. Candidates considered

### 2.1 Candidate 1 — "The 'RAW penalty' is quantization, not missing adaptivity" (merged: physics + edge)

**Mechanism as submitted.** Every per-frame tone map (vendor plateau-equalisation AGC, per-frame
clip, TCNet, N-AGC) exists for an 8-bit container. A 14-bit LWIR scene occupies a few hundred to
~2000 counts; fixed linear scaling into 8 bits collapses it to ~5–30 grey levels and destroys the
sub-NETD gradients that small/far targets live on. That quantization, not missing adaptivity, is
what the published "RAW 14-bit" rows measure. Fed as float, a dataset-wide affine (subtract
dataset median, divide by dataset 1–99 % span) carries the full raw information, preserves
absolute apparent-temperature ordering across frames, and is absorbed by the first conv+BN of a
1-channel detector. The only physical reason for per-frame adaptivity is offset drift between
flat-field corrections, which needs one parameter (per-frame median), not a histogram remap.
Per-frame AGC also has an unmeasured temporal cost: object grey level becomes a function of scene
context, which should surface as confidence volatility and single-frame drop-outs.

**Design as submitted.** 2 factors: adaptivity A in {A0 fixed dataset affine; A1 per-frame median
offset; A2 per-frame 1–99 % clip; A3 plateau-equalisation reimplementation; A4 vendor AGC JPEG;
A5 TCNet/N-AGC learned reference} x precision Q in {float32, 8-bit uniform requantization}.
Stock yolov8s ch=1, 512 letterbox, 50 epochs, 3-class FLIR ADAS v2, 3–5 seeds. Label-free
covariates: per-frame 14-bit dynamic range terciles; per-frame isotonic AGC transfer function
(JPEG value on TIFF count), whose frame-to-frame movement is the flicker dose.

**Hypotheses as submitted.** (i) mAP(A0-float) − mAP(A4) >= −0.5 with 95 % CI excluding any loss
larger than the across-seed SD; (ii) A0/A1/A2/A3-float within seed SD of each other; (iii)
A0-8bit >= 5 mAP50-95 below A0-float; (iv) DR-tercile interaction has opposite signs for A0-8bit
vs A2/A3-8bit and vanishes at float; (v) flicker rate and per-object confidence SD lower for
A0-float than A4/A2, with dose-response to AGC-map movement.

**Falsifier as submitted.** A0/A1-float trail A3-float or A4 by more than seed SD with paired CI
excluding 0, concentrated in low-contrast frames: adaptivity is genuinely needed, TCNet/N-AGC are
justified, practitioner message "the 8-bit AGC path is fine".

### 2.2 Candidate 2 — "Cross-Fitted Oracle Headroom" (stats)

**Mechanism as submitted.** Router papers justify adaptive preprocessing by a per-image oracle
computed with one trained detector; the argmin over K actions on realised per-frame losses
harvests seed-level prediction churn (winner's curse). Cross-fit: choose the action with seed s,
score with seed s' != s. H_cross estimates the seed-stable headroom; H_naive − H_cross is churn.
Sham action set (null perturbations) as positive control. Prediction: LWIR classical-enhancer
headroom is all churn (AGC already stretched contrast); RGB fog retains a stable component.

**Design as submitted.** FLIR ADAS v2 3-class, YOLOv8-s ch=1, 5 seeds, zer0 action set
{identity, adaptive gamma, fixed gamma 0.7/1.4, CLAHE, LCN, unsharp}; RGB arm YOLOv8-s on
VOC-fog/RTTS with IA-YOLO/GDIP filter bank, 3 seeds. P1–P4 plus sham control. ~60–80 GPU-h.

### 2.3 Refuted earlier (not re-scored)

| Candidate | Threat | Reason it died |
|---|---|---|
| LWIR fog mediation test (ZUT-FIR-ADAS 16-bit + UNIRI-TID) | MED | ZUT has 752 clear-weather annotations (one tiny clear recording) as the entire reference class; UNIRI-TID is 8-bit, static camera, one 13-min fog session; mediator is a noisy proxy so a non-zero direct effect is uninterpretable; 95 GB archive vs 76 GB disk; delta thin vs Rivera Velázquez 2022, WiSE-OD 2026, N-AGC 2025. |
| Range-structured LWIR fog augmentation (Beer–Lambert + AGC emulation) | MED | UNIRI-TID fog shot at <=50 m by design (far bin structurally empty), one site per weather, site confounded with weather; ZUT fog-recording count unpublished; sigma prior contradicted by the only real-fog evidence; frame-constant control is undone by any per-frame normalisation; oracle E is a straw bound (T08). |
| Radiometric-privileged distillation (14-bit teacher -> 8-bit AGC student) | HIGH | Own go/no-go gate predicted to fail by TCNet/Fieldscale; AGC-randomisation augmentation absorbs the delta and is absent; effect size inside seed SD; identical-architecture LUPI KD already published (Bartolo 2026, Shi 2025). |

---

## 3. Verified closest works

### 3.1 For Candidate 1

| Work | Venue / date | Link | What it occupies | What it leaves open |
|---|---|---|---|---|
| Thermal Chameleon (TCNet) — Lee, Kim, Cho, Kim | IEEE RA-L 2024; arXiv 2410.18340 (24 Oct 2024) | https://arxiv.org/abs/2410.18340 | Same dataset and baseline family (RAW /(2^14−1), min-max, 1–99 % clip, HE, FLIR AGC, Fieldscale, TCNet). YOLO-X mAP: AGC 53.5, min-max 50.8, clip 49.1, RAW 37.9, TCNet 56.0. Source of the quoted 37.9 vs 53.5. | RAW-row dtype never stated; failure attributed to "narrow distribution", quantization never named; single-run main tables; per-image eval, no temporal metric; code usage withheld. |
| N-AGC — Javidnia | ICCVW 2025 (MIRA), CVF open access | https://openaccess.thecvf.com/content/ICCV2025W/MIRA/html/Javidnia_Task-Driven_Neural_Adaptive_Gain_Control_16-bit-to-8-bit_Thermal_Tone_Mapping_for_ICCVW_2025_paper.html | FLIR v1+v2 16-bit TIFFs, video-test partition (3,742 frames) as held-out set. RAW-stretch (=A2, JPEG Q=90) 65.1/67.8 AP50 vs vendor 75.3/77.2 vs N-AGC 90.1/86.3. Intro states the quantization mechanism in words. | Detector trained once on vendor AGC and frozen (RAW row is domain shift, not a retrained arm); single run; no float arm; temporal consistency explicitly future work. |
| Fieldscale — Gil, Jeon, Kim | IEEE RA-L 2024; arXiv 2405.15395 | https://arxiv.org/abs/2405.15395 | Exact 3-class FLIR protocol; YOLOX-M: AGC 48.4, Minmax 47.0, Fieldscale 49.1 (global min-max already within 1.4 mAP of AGC, single run). | All 8-bit; no RAW/float arm; no seeds/CIs; flicker listed as future work. |
| Lynred pipeline — Karam, Matias, Breniere, Chanussot | arXiv 2407.04484 (5 Jul 2024) | https://arxiv.org/abs/2407.04484 | Direct counter-claim: uint16->fp32 min-max / z-score "renders the images unusable with our pretrained neural networks"; dedicated tone mapper indispensable. | Prose only, no table; COCO-pretrained 3-ch YOLOv4-Tiny; in-house data. This is the pretraining mechanism the new factor targets. |
| Grimming, McIntosh, Mahalanobis, Driggers | OSA Continuum 4(2):529, 2021, DOI 10.1364/OSAC.404600 | https://doi.org/10.1364/OSAC.404600 | Fast R-CNN on pre-AGC 14-bit TIFF with saturation-based linear rescale "slightly better" than AGC JPEG. | Single run, 8-bit output, no quantization or temporal analysis; detail seen only via snippet. |
| Lost in Compression — Bhowmik, Barker, Gaus, Breckon | arXiv 2205.08002 (May 2022) — not opened | https://arxiv.org/abs/2205.08002 | JPEG quality alone degrades FLIR detection, especially small objects. | Reason A4-JPEG cannot be the primary comparator. |

### 3.2 For Candidate 2

| Work | Venue / date | Link | Overlap |
|---|---|---|---|
| Cameron, Hoos, Leyton-Brown — Bias in Algorithm Portfolio Performance Evaluation | IJCAI 2016 | https://www.ijcai.org/Abstract/16/107 | Canonical origin: per-instance oracle over randomized members is optimistically biased; fix by averaging independent runs. |
| Bae — Sampling Luck Masquerades as Allocation Gain | arXiv 2608.13087 (Aug 2026) | https://arxiv.org/abs/2608.13087 | Pre-registered audit with instance-wise null and holdout scoring; gain collapses to ~0. |
| Chen — What Does a Routing Oracle Measure Under Stochastic Decoding? | arXiv 2607.03436 (Jul 2026) | https://arxiv.org/abs/2607.03436 | Hindsight oracle vs single-commit ceiling; the "premium" is the churn component. |
| Awad et al. — Revisiting Underwater Image Enhancement for Object Detection | J. Imaging 12(1):18, 2026 | https://pmc.ncbi.nlm.nih.gov/articles/PMC12842599 | Published uncorrected per-image enhancement oracle (single detector, single seed). |
| Shah, Li — Holdout Best-of-N | arXiv 2610.08719 (Oct 2026) | https://arxiv.org/abs/2610.08719 | Exact unbiasedness theory: split selection from scoring; one-seed cross-fit is a lower bound on seed-stable headroom. |
| Manfredi, Wang — Shift Equivariance in Object Detection | ECCVW 2020; arXiv 2008.05787 | https://arxiv.org/abs/2008.05787 | H_naive > 0 under single-pixel shifts already documented; the sham set is a control, not a result. |

---

## 4. Refutations (verbatim substance)

### 4.1 Candidate 1 — refuted as submitted

1. **Identifiability.** Three mechanisms can produce a RAW penalty: (a) bit-depth posterization,
   (b) input-scale mismatch against a COCO-pretrained backbone (Lynred), (c) missing per-frame
   adaptivity. The 2x2 crosses (a) with (c) and fixes the detector at ch=1-from-scratch with BN,
   which absorbs any affine by construction, so (b) is excluded by design. TCNet's RAW row is
   "linear /(2^14−1)", the natural phrasing for a float tensor into a pretrained YOLO-X; if so,
   the true explanation is (b) and the headline is false while every test passes.
2. **Expected defaults.** (ii) float maps tie is what a BN detector with Ultralytics augmentation
   should do; (iii) 8-bit posterization of a ~300–2000-count scene losing >=5 mAP is textbook;
   (i) a fixed affine matching vendor AGC saves 0 ms over an AGC that already runs on-camera.
   The consequential comparator (TCNet/N-AGC, claiming +2.5 to +20 mAP over AGC) is the arm
   relegated to one seed.
3. **Flicker.** High prior of a null for an uninteresting reason (Tau 2 AGC is temporally damped;
   FLIR JPEG may be filtered software AGC); video_thermal_test has few exchangeable video
   clusters for the paired CIs (MATH_STAT_REVIEW §4.1).
4. **JPEG confound.** A4-vs-A0 is a JPEG-Q comparison (Bhowmik 2022) unless A3 is a faithful
   lossless reproduction of FLIR's proprietary plateau AGC.

**Strongest fix (reviewer):** add detector initialisation as a third factor (COCO-pretrained
3-ch YOLOv8s on grey-replicated input, A0-float / A0-8bit / vendor, 3 seeds, ~55 GPU-h); replace
A4-JPEG with A3 := per-frame isotonic tone curve recovered from paired TIFF/JPEG and applied
losslessly to the TIFF; promote TCNet/N-AGC retrained-per-arm to 3 seeds; demote flicker to a
descriptive secondary gated on verified map movement; report cluster counts.

### 4.2 Candidate 2 — refuted as submitted

1. **Confounded headline.** LWIR-vs-RGB verdict is a two-cell comparison in which modality is
   perfectly confounded with degradation state (clear FLIR vs hazy VOC-fog/RTTS), detector input
   (ch=1 vs ch=3) and action set. IA-YOLO's defog filter is the analytic inverse of the model
   used to synthesise VOC-fog, so P4 passes by construction. The LWIR null cannot support "stop
   building LWIR routers" because those routers target fog/rain/low-light that FLIR lacks.
2. **Estimator mis-stated.** H_cross with one selecting seed is a lower bound on seed-stable
   headroom (Shah & Li), not unbiased; with an integer per-frame loss the selector is noisy, so a
   null P2 is uninterpretable without a planted positive control.
3. **Residual.** Statistical idea is Cameron 2016 / Bae / Chen / Shah–Li; relative to audit B5
   the delta is a protocol correction, not a paper.

**Strongest fix (reviewer):** 2x2 modality x real degradation with ZUT-FIR-ADAS / UNIRI-TID as
LWIR-degraded cells, leave-one-seed-out cross-fitting, planted invertible-degradation control.
Chair's note: the LWIR-degraded cell is exactly the data the two fog refutations (§2.3) showed
cannot support recording-clustered inference, so this fix is not reachable with available data.

---

## 5. Scores

Three lenses, each scoring novelty / rigor / feasibility / impact out of 10.

| Candidate | Lens 1 | Lens 2 | Lens 3 | Mean total |
|---|---|---|---|---|
| 1. RAW penalty is quantization (+ temporal AGC cost) | 5 / 5 / 7 / 4 = **21** | 5 / 5 / 7 / 6 = **23** | 5 / 5 / 7 / 4 = **21** | **21.7** |
| 2. Cross-fitted oracle headroom | 4 / 4 / 6 / 5 = 19 | 4 / 4 / 7 / 3 = 18 | 4 / 4 / 7 / 5 = 20 | 19.0 |

Top pick, all three lenses: Candidate 1 (Lens 2 explicitly conditional on adopting the
strongest fix). Points of agreement across lenses: the precision x adaptivity crossing, the
per-arm-retrained float affine, multi-seed CIs and the AGC-transfer measurement are unoccupied;
the title attribution is unidentifiable without the pretraining factor; flicker is a secondary;
Candidate 2's confound is structural and its repair is data-blocked.

---

## 6. The paper as it will be developed

### 6.1 Single mechanism

A per-frame tone map does two separable things: it *adapts* (maps the scene's occupied count
range to the container) and it *quantizes* (writes 8 bits). Detectors are additionally
*pretrained* on 8-bit RGB statistics. The published RAW penalty is the sum of the quantization
and pretraining terms; the adaptivity term is ~0 for a from-scratch 1-channel detector whose
first conv+BN absorbs a dataset-wide affine. The design is a 3-factor crossing that puts each
term in its own cell.

### 6.2 Factors and arms

- **A (adaptivity, applied to the 14-bit TIFF):**
  A0 dataset-wide affine (median, 1–99 % span; zero parameters at inference);
  A1 per-frame median offset only (the FFC-drift control);
  A2 per-frame 1–99 % clip (= N-AGC's RAW-stretch);
  **A3 camera AGC applied losslessly**: per-frame isotonic fit of JPEG value on TIFF count from
  the paired TIFF/JPEG, applied to the TIFF (no JPEG loss; report fit residual per frame);
  A4 vendor AGC JPEG (descriptive only, kept to tie to the published tables);
  A5 learned per-frame mapper (TCNet re-implementation from the N-AGC description; N-AGC if its
  code is released), retrained per arm, **3 seeds**.
- **Q (precision):** float32 vs 8-bit uniform requantization of the identical mapped image.
- **I (initialisation):** ch=1 YOLOv8-s from scratch vs COCO-pretrained 3-ch YOLOv8-s on
  grey-replicated input with Ultralytics' standard normalisation.

Full cells: A0–A3 x Q x I(scratch), 3 seeds (5 for A0 and A3); A5 x float x scratch, 3 seeds;
A0-float, A0-8bit, A3-float x I(pretrained), 3 seeds. Remaining pretrained cells optional.
Stock yolov8s, 512 letterbox, 50 epochs, 3-class FLIR ADAS v2, identical augmentation
everywhere (brightness/contrast jitter applied in the mapped domain, logged).

### 6.3 Pre-registered hypotheses (frozen before video_thermal_test is opened)

- **H1 (primary, non-inferiority).** mAP50-95(A0-float, scratch) − mAP50-95(A3-float, scratch)
  >= −SD_seed with paired video-cluster bootstrap 95 % CI (1000 reps, same resampled video index
  set for every arm) excluding a loss larger than SD_seed. Reported alongside A0 vs A5.
- **H2 (precision main effect).** A0-8bit <= A0-float − 5 mAP50-95 (scratch), CI excluding 0.
- **H3 (initialisation x precision interaction).** The A0-float − A0-8bit gap is >= 5 for
  scratch; for pretrained, A0-float itself trails A3-float by more than SD_seed (the Lynred
  failure reproduced and isolated), so the "RAW penalty" at float is a pretraining effect.
- **H4 (adaptivity buys nothing at float).** A0/A1/A2/A3/A5-float (scratch) all within SD_seed
  of each other; DR-tercile interaction present for the 8-bit arms (opposite signs for A0-8bit
  vs A2/A3-8bit) and absent at float.
- **S1 (descriptive secondary, gated).** Gate: the per-frame isotonic AGC curve moves between
  consecutive frames of video_thermal_test by more than its fit residual on >= 10 % of frame
  pairs. If the gate passes: flicker rate and per-object confidence SD for A0-float vs A3-float vs
  A4, with the number of video clusters stated and CIs reported as descriptive, plus the
  dose-response plot against map movement. If the gate fails, that is reported as the
  measurement ("FLIR's shipped AGC is temporally damped on this split") and no flicker claim is
  made.

### 6.4 Falsifier

H1 fails if A0-float (and A1-float) trail A3-float or A5-float by more than SD_seed with the
paired CI excluding 0, concentrated in low-DR frames. Then per-frame histogram adaptivity is a
real detector requirement at float precision, TCNet/N-AGC are justified, and the paper becomes
the first multi-seed, CI-bearing, precision-controlled confirmation of that (every prior row is
single-run and dtype-unstated). H3 fails if the pretrained net also shows no float penalty, in
which case Lynred's prose result does not replicate on FLIR and the paper says so.

### 6.5 Grafts from runners-up

- From Candidate 2: the **planted positive control** logic. Before any claim of "no difference"
  is made, the pipeline must show it can detect a known difference of the pre-registered size:
  the 8-bit requantization cell (H2) is that control, and a pre-declared minimum detectable
  effect (SD_seed and the bootstrap CI half-width on the pilot subset) is reported next to every
  null. Also adopted: identical resampled video index sets across every arm and seed, and
  reporting of the exchangeable-unit count for every CI.
- From the refuted distillation candidate: the observation that the 14-bit TIFF makes
  **tone-map randomisation augmentation** free. Not an arm; recorded as a one-paragraph
  limitation/future-work note, because it tests invariance, not the mechanism.
- Not grafted: Candidate 2's cross-fitted oracle, sham action set, RGB arm and router P3. They
  answer a different question (is per-image routing headroom real?) and would reintroduce a
  second mechanism into the title.

### 6.6 Budget

Scratch cells: 4 A-levels x 2 Q x 3 seeds = 24 runs + 4 extra seeds (A0, A3 to 5) + A5 x 3 =
31 runs x ~6 h ≈ 190 GPU-h. Pretrained cells: 3 arms x 3 seeds x ~5 h ≈ 45 GPU-h. Total
≈ 235 GPU-h ≈ 5–6 days on two 2080 Ti. Disk: ~17 GB TIFF + ~16 GB JPEG, no pseudo-RGB cache,
within 76 GB. Isotonic fits: sklearn on ~10k sampled pixels per frame, CPU.

Timeline: week 1 data verification + quarantined pilot (images_thermal_val subset only);
week 2 freeze protocol, OSF preregistration; weeks 3–4 runs; weeks 5–7 analysis and writing.
video_thermal_test opened exactly once.

Venue: IEEE Sensors Journal or Infrared Physics & Technology (primary); CVPR PBVS 2027 or RA-L
as alternatives.

---

## 7. Risks and kill conditions

| Risk | Severity | Handling |
|---|---|---|
| 14-bit TIFFs missing for video_thermal_test or not stem-paired with JPEGs | Hard kill for the primary split | Verify day 1. Fallback: hold out whole videos from stills (filenames encode video id), reported as a pre-registered deviation. N-AGC used this partition, so coverage is probable. |
| Vendor map is not a global monotone curve (FLIR DDE/local contrast) so the isotonic A3 does not reproduce the JPEG | Medium | Report per-frame fit residual; if residual is large, A3 is relabelled "global component of the vendor map" and A4 stays descriptive; the A0-vs-A5 contrast carries the adaptivity claim. |
| Few video clusters in video_thermal_test | Medium | State cluster count in every CI; pilot CI half-width decides whether 3 or 5 seeds for A0/A3; non-inferiority margin fixed at SD_seed before unblinding. |
| TCNet/N-AGC code unavailable or does not fit 11 GB | Medium | Re-implement from the paper description, label as "re-implementation, unreproduced numbers"; if it still fails, A2 stands in only for the shape of the DR interaction, never as the learned reference, and the claim is narrowed to classical maps. |
| Pretrained 3-ch cell shows no float penalty (H3 fails) | Low–medium | Reported as a non-replication of Lynred on FLIR; the scratch cells still identify quantization vs adaptivity. |
| Frame-to-frame offset drift in pre-AGC counts | Low | A1 is the control; if A1 beats A0 by > SD_seed the "zero-parameter" wording is changed to "one-parameter". |
| Augmentation absorbing the adaptivity term | Low | Identical augmentation across arms, logged; an augmentation-off replicate of A0/A3 at 1 seed as a sanity check. |
| FLIR ADAS v2 licence text unverified for derived data release | Low | Release manifests, hashes, code and tone curves; no images. |
| Flicker secondary is null for a damped-AGC reason | Accepted | Gated; a failed gate is itself the measurement and is reported without a claim. |

---

## 8. What the chair rejected and why (so it is not relitigated)

- Keeping the two-claim title (quantization + temporal cost). A merged title with two claims
  and one weak secondary reads as a menu. The flicker study is demoted and gated.
- Treating A4-JPEG as the primary comparator. It is a JPEG-quality comparison.
- Running the learned mapper at one seed. The only comparator a practitioner cares about must
  carry the same error bars as the arms it is compared against.
- Developing Candidate 2 as a companion short paper. Its data repair is blocked; its estimator
  is known; and folding its oracle machinery into this paper would reintroduce a second
  mechanism. The audit's B5 protocol note (seed-split correction) is kept as an internal
  protocol improvement, not a paper.
