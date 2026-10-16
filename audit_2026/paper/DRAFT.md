# Decomposing the "RAW penalty" in LWIR object detection: a pre-registered test of quantization, pretraining mismatch and per-frame adaptivity on FLIR ADAS v2

**Status: pre-registration draft, revision 2 (after internal hostile review), no experiment run. Every numerical result is `[PENDING EXPERIMENT]`. Every element whose truth has not been checked on this machine is marked `[UNVERIFIED]`. Prior-art numbers are quoted only from sources opened during the audit (see References) and are single-run figures as published. Design parameters chosen in this draft and not settled by the council record are listed in Appendix A.**

Conditional title, to be used only if H1, H2 and H3 all hold as pre-registered: *Float precision, not per-frame adaptivity: the "RAW penalty" in LWIR object detection is quantization and pretraining mismatch, and a fixed affine is non-inferior to the per-frame tone maps tested once both are controlled.* The present title asserts no outcome.

Authors: lead author (mathematics / statistics, ML and imaging) and thermal-detection collaborator. Author order, contribution statement and the role of the second repository's author are open items (`audit/INPUTS_AND_PROVENANCE.md` §2.1, §5) and must be settled before submission.

Source of this draft: `paper/COUNCIL_RECORD.md` (decision of 2026-10-15), the internal hostile review of revision 1 (responses in §10), `audit/PRIOR_ART_NOVELTY_MATRIX.md`, `audit/MATH_STAT_REVIEW.md`, `audit/EXPERIMENT_MATRIX.md` (to be re-synchronised with this revision), `audit/CODE_AUDIT_2026.md`, `audit/REPRODUCTION_PROTOCOL.md`.

---

## One-sentence claim (to be filled only from results)

On a video-clustered 3-class FLIR ADAS v2 protocol with 3 to 5 seeds per cell, a dataset-wide affine with no per-frame parameters, fed as float32 to a from-scratch 1-channel YOLOv8-s, is `[non-inferior / inferior / inconclusive, PENDING EXPERIMENT]` within a pre-registered margin of 1.0 mAP50-95 to a per-frame affine retrained under the same configuration, and `[PENDING EXPERIMENT]` to the camera's own global tone curve and to a learned per-frame mapper; the published "RAW penalty" (TCNet: 37.9 vs 53.5 mAP, single run, metric definition and input dtype unrecorded) is `[reproduced / not reproduced, PENDING EXPERIMENT]` by 8-bit requantization of the same counts, measured as separate clipping, rounding and range-allocation interventions, and by feeding a float affine to a COCO-pretrained 3-channel backbone, a map-sensitivity effect under pretrained initialisation that is attributed to pretraining only through a scratch-3-channel difference-in-differences.

---

## Abstract

Thermal object detectors on FLIR ADAS v2 are, in every work the audit catalogued, trained on the 8-bit automatic-gain-controlled (AGC) image the dataset ships rather than on the radiometric counts in the paired TIFF. Three papers on this dataset report that a function of the counts costs accuracy relative to the vendor AGC image (TCNet: 37.9 vs 53.5 mAP; N-AGC: 65.1 and 67.8 vs 75.3 and 77.2 AP50; Fieldscale: 47.0 vs 48.4 mAP for a global min-max map), and the larger gaps are used to justify learned per-frame tone mappers. The gap conflates three separable mechanisms: (a) 8-bit requantization after a given mapping, which we separate into three interventions, each with its own cell: clipping to the mapped range, rounding to 8 bits within a dataset-matched container, and range allocation (writing a scene occupying a few hundred to roughly two thousand counts into the full 14-bit container before rounding); float32 adds no information beyond the counts, so the intervention under test is the absence of requantization after a specified map, not float precision as such; (b) mismatch between a float input at an unfamiliar scale and a backbone pretrained on 8-bit RGB statistics; and (c) missing per-frame adaptivity. No published RAW row states the input dtype, all are single-run, and the metric definitions differ across them. We pre-register a three-factor crossing on the paired TIFF and JPEG frames of FLIR ADAS v2: adaptivity (dataset-wide affine, per-video affine as a slow camera-state control, per-frame affine, the vendor's global tone curve recovered per frame by isotonic regression and applied at count resolution, the vendor JPEG as a descriptive tie-in, a learned per-frame mapper retrained per arm, and TCNet's full-range linear scaling as the container cell), precision (float32 versus 8-bit requantization of the identical mapped image), and initialisation (1-channel from scratch, 3-channel from scratch as a channel control, and COCO-pretrained 3-channel on grey-replicated input), with the published setting (vendor JPEG and full-range float counts into the pretrained backbone) run as its own cells. The primary pre-registered contrast is a non-inferiority test of the dataset-wide affine against the per-frame affine at float precision, from scratch, with a fixed margin of 1.0 mAP50-95 and a two-way (video by seed) paired bootstrap. The requantization cells are the planted positive control: the pipeline must resolve the combined rounding-plus-range-allocation effect (dataset affine at float versus full-range 8-bit) with a confidence-interval lower bound above the margin before any equivalence is interpreted; the two parts and the clipping term are reported separately and no part is labelled a quantization effect unless its own confidence interval supports it. Results: `[PENDING EXPERIMENT]` for the primary contrast, `[PENDING EXPERIMENT]` for the clipping, rounding and range-allocation terms, `[PENDING EXPERIMENT]` for the initialisation by precision interaction and the published-setting cells, `[PENDING EXPERIMENT]` for the equivalence of float arms, and `[PENDING EXPERIMENT]` for the gated temporal-stability secondary. Whatever the outcome, the paper supplies a multi-seed, confidence-interval-bearing, precision-controlled table for this question; to our knowledge, within the works verified during the audit, no such table exists `[a systematic literature search for tone-mapping and bit-depth studies on this dataset is PENDING, see §2.3]`. We release manifests, hashes, code, per-frame tone curves (licence permitting) and the full pre-registration; no images.

---

## 1. Introduction

### 1.1 Setting

A long-wave infrared (LWIR) microbolometer camera such as the FLIR Tau 2 used to record FLIR ADAS v2 produces integer counts per pixel; the dataset card describes the TIFFs as 14-bit, N-AGC [2] refers to the same files as 16-bit TIFFs, and the container bit depth and the occupied count range are `[UNVERIFIED: to be read from the TIFF headers and pixel maxima on day 1]`. The counts are a monotone function of apparent temperature up to camera state: a per-frame offset that drifts between flat-field corrections (FFC), a gain that depends on focal-plane-array and shutter temperature, a discrete gain state (the Tau 2 has high- and low-gain modes that rescale counts `[UNVERIFIED for the dataset recordings]`), and pixel-wise residual non-uniformity. The dataset ships, for each frame, a TIFF of counts and an 8-bit AGC JPEG. Whether that JPEG is the camera's on-board AGC output or a software AGC applied to the TIFF afterwards, and which of the Tau 2 AGC modes (plateau equalisation, linear, manual, with or without digital detail enhancement, DDE) was in use, is not documented in the dataset materials the audit saw `[UNVERIFIED]`. Every detector on this dataset that the audit catalogued (`audit/PRIOR_ART_NOVELTY_MATRIX.md` §A) trains on the JPEG.

### 1.2 The open problem, concretely

Three recent papers on the same dataset compare detectors trained on the vendor AGC image with detectors trained on some function of the counts:

- TCNet [1] (YOLO-X, 3-class FLIR, single run): AGC 53.5, min-max 50.8, 1-99 percent clip 49.1, "RAW" (counts divided by 2^14 minus 1) 37.9, TCNet 56.0 "mAP". The 37.9 versus 53.5 gap is attributed to a "narrow distribution". The dtype of the RAW tensor, whether "mAP" is COCO mAP50-95 or mAP50, and whether the YOLO-X backbone was COCO-pretrained are not recorded in the audit `[UNVERIFIED]`.
- N-AGC [2] (FLIR v1 and v2 16-bit TIFFs, video-test partition held out, single run): RAW-stretch (per-frame 1-99 percent clip written to JPEG at quality 90) 65.1 and 67.8 AP50 versus vendor AGC 75.3 and 77.2 versus N-AGC 90.1 and 86.3. The audit recorded that the detector was trained once on vendor AGC and frozen; that reading is hard to reconcile with a frozen detector scoring higher on N-AGC output than on the AGC images it was trained on, so either N-AGC retrains the detector or the setting was mis-transcribed `[UNVERIFIED: the paper must be re-read before submission; the CVF page returned HTTP 403 on 2026-10-15]`. Which of the two numbers in each pair belongs to v1 and which to v2 is likewise `[UNVERIFIED]`. The introduction states the quantization mechanism in words but does not test it.
- Fieldscale [3] (YOLOX-M, exact 3-class FLIR protocol, single run): AGC 48.4, "global min-max" 47.0, Fieldscale 49.1. All arms are 8-bit; there is no float arm. Whether "global" means per image or dataset-wide is not recorded `[UNVERIFIED]`.

A fourth source, the Lynred pipeline paper [4], makes the opposite-sign observation in prose only: feeding uint16 counts converted to float32 by min-max or z-score normalisation to their COCO-pretrained 3-channel YOLOv4-Tiny "renders the images unusable", so a dedicated tone mapper is "indispensable". A fifth, Grimming et al. [5], found a saturation-based linear rescale of the pre-AGC TIFF "slightly better" than the AGC JPEG for Fast R-CNN, single run, 8-bit output.

These rows do not agree with each other. The 8-bit-versus-8-bit comparisons (Fieldscale's global min-max, TCNet's min-max and clip) put a stretch-then-quantize map within 1.4 to 4.4 points of the vendor AGC, so the published evidence that per-frame *adaptivity* matters at 8 bits is a small, single-run effect. The large gaps appear only in rows that change the precision or the input scale as well: TCNet's RAW row (full-range scaling, dtype unstated, backbone pretraining unrecorded), N-AGC's RAW row (JPEG re-encoding plus a possibly frozen detector), and Lynred's float input to a pretrained network. Fieldscale's min-max row is the one published row that changes adaptivity alone, and its 1.4-point gap is therefore the strongest existing evidence on the adaptivity question; it is single-run, 8-bit, and metric-unrecorded, and this paper's H4 is its multi-seed, float-precision counterpart.

A per-frame tone map does two separable things at once. It *adapts*: it maps the scene's occupied count range, typically a few hundred to about two thousand counts `[council estimate, UNVERIFIED until Figure 1 is produced from the manifests]`, into the container. And it *quantizes*: it writes 8 bits. The quantization step for a full-range container is 2^14 / 256 = 64 counts; for a container matched to a band of s counts it is s / 256 counts. Whether a step of that size destroys detectable structure depends on the camera's counts-per-kelvin gain relative to its noise-equivalent temperature difference (NETD), which is not public for these recordings `[UNVERIFIED]`; the mechanism is therefore stated as "the 8-bit step is larger than the smallest count difference the detector could exploit", and it is tested, not assumed, by the precision factor. A detector pretrained on 8-bit RGB images adds a third effect: a float input whose scale and contrast statistics do not resemble the pretraining data can fail for reasons that have nothing to do with the thermal scene. Among the published RAW rows, only Fieldscale's min-max row changes exactly one of these three things; the rows with large gaps change at least two, and none reports a seed SD or a confidence interval.

### 1.3 What we test

We pre-register a design that gives each mechanism its own cell. The adaptivity factor A ranges over a dataset-wide affine with no per-frame parameters (A0), a per-video affine that removes slow camera-state changes and nothing else (A1), a per-frame affine equal to N-AGC's RAW-stretch and TCNet's clip before their 8-bit step (A2), the vendor's global tone curve recovered per frame from the paired TIFF and JPEG and applied at count resolution (A3), the vendor JPEG itself as a descriptive tie-in to the published tables (A4), a learned per-frame mapper retrained per arm with seeds (A5), and TCNet's full-range linear scaling (R). The precision factor Q is float32 versus 8-bit uniform requantization of the identical mapped image. The initialisation factor I is a 1-channel YOLOv8-s from scratch, a 3-channel YOLOv8-s from scratch on grey-replicated input (channel-count control), and a COCO-pretrained 3-channel YOLOv8-s on grey-replicated input at the input scale the library itself uses for 8-bit grey images.

The mechanism we propose is that the published RAW penalty is the sum of a quantization term (Q, with a container part and a rounding part) and a pretraining term (I), and that the adaptivity term (A) is within the margin at float precision for a from-scratch detector, because the first convolution and batch normalisation absorb any dataset-wide affine, and because the physical reasons for frame-to-frame rescaling of counts are slow camera-state changes rather than per-frame scene content. The design can refute each part separately (§4.6, §7).

### 1.4 Contributions (claims, not results)

1. A three-factor decomposition of the "RAW penalty" (adaptivity, precision, initialisation) in which each term is estimated in its own cell as a retraining intervention that changes exactly one declared preprocessing operation (clipping, rounding, range allocation, per-frame adaptivity, initialisation), the operations being unit-tested in `audit_tools/preproc_controls.py` (test T09) rather than asserted, on paired TIFF and JPEG frames of a public dataset, with all arms retrained from the same configuration, and with the published setting (vendor JPEG and full-range float counts into a COCO-pretrained backbone) run as cells of the same design.
2. A comparator for the camera's own tone curve that does not pass through JPEG: a per-frame isotonic transfer function recovered from the paired TIFF and JPEG and applied to the TIFF at count resolution, with a per-frame fit residual decomposed against an estimated JPEG-error floor, so that the residual above the floor measures how much of the vendor map is not a global monotone curve.
3. A pre-registered non-inferiority test with a fixed margin, a two-way (video by seed) paired bootstrap, a planted positive control (the full-range requantization cell) and a minimum detectable effect reported next to every null.
4. A multi-seed, CI-bearing table for the classical and learned tone-map baselines of [1], [2], [3], retrained per arm under identical augmentation; to our knowledge, within the works verified in the audit, no such table exists (`[search PENDING, §2.3]`).
5. A label-free covariate pair, the per-frame dynamic range and the per-frame isotonic AGC transfer function, released with the manifests (licence permitting), which makes the vendor map's frame-to-frame movement measurable on this dataset.

**Why the outcome matters either way.** If H1 and H2 hold, the RAW rows that motivate learned per-frame mappers are mis-specified baselines: the comparison a mapper paper should make is against float counts through a fixed affine, retrained, and the "+2.5 to +20 mAP over AGC" figures in [1] and [2] are not evidence of a per-frame adaptivity requirement. The practical recipe for any integrator with host access to the count stream (the dataset's own TIFFs show that such access exists) is then to train on float counts with a dataset constant and to skip the tone mapper. If H1 fails, the paper is the first CI-bearing, precision-controlled confirmation that per-frame adaptivity is a detector requirement at float precision, which is the premise every mapper paper assumes and none has tested. The two outcomes are not symmetric in consequence: the first invalidates a baseline row in a growing method literature; the second validates it.

---

## 2. Related work and claim boundary

### 2.1 Verified closest works

| Work | What it occupies | What it leaves open (and this paper fills) |
|---|---|---|
| TCNet, Lee et al., IEEE RA-L 2024 [1] | Same dataset and baseline family (RAW divided by 2^14 minus 1, min-max, 1-99 percent clip, HE, FLIR AGC, Fieldscale, TCNet). Source of the 37.9 vs 53.5 figure. | RAW-row dtype never stated; failure attributed to "narrow distribution", quantization never named; backbone pretraining and metric definition not recorded in the audit; single-run main tables; per-image evaluation with no temporal metric; code usage withheld. |
| N-AGC, Javidnia, ICCVW 2025 [2] | FLIR v1 and v2 16-bit TIFFs, video-test partition (3,742 frames per the audit's reading `[UNVERIFIED]`) as held-out set. Intro states the quantization mechanism in words. | The audit's reading that the detector was trained once on vendor AGC and frozen is inconsistent with the quoted numbers (§1.2) `[UNVERIFIED]`; single run; no float arm; temporal consistency explicitly future work. |
| Fieldscale, Gil et al., IEEE RA-L 2024 [3] | Exact 3-class FLIR protocol; global min-max within 1.4 mAP of AGC, single run. The strongest existing evidence that adaptivity alone is a small effect at 8 bits. | All 8-bit; no RAW or float arm; no seeds or CIs; per-image versus dataset-wide min-max not recorded; flicker listed as future work. |
| Lynred pipeline, Karam et al., arXiv 2024 [4] | Direct counter-claim: uint16 to fp32 min-max or z-score input is unusable with their pretrained networks; dedicated tone mapper indispensable. | Prose only, no table; COCO-pretrained 3-channel YOLOv4-Tiny; in-house data. This is the pretraining mechanism our factor I targets; its exact input (z-score float) is an optional cell (§6.1). |
| Grimming et al., OSA Continuum 2021 [5] | Fast R-CNN on pre-AGC TIFF with saturation-based linear rescale "slightly better" than AGC JPEG. | Single run, 8-bit output, no quantization or temporal analysis. `[Detail seen only via snippet during the audit; the publisher page could not be loaded on 2026-10-15.]` |
| Lost in Compression, Bhowmik et al., arXiv 2022 [6] | JPEG compression at six levels degrades thermal infrared detection for three detectors; retraining on compressed images recovers much of the loss; tiny and small objects are more sensitive than medium and large ones (abstract, opened 2026-10-15). | The reason the vendor JPEG (A4) cannot be the primary comparator: a JPEG-versus-no-JPEG contrast is confounded with the map. `[Body not opened; cited for the abstract-level claim only.]` |

### 2.2 Claim-to-prior-art boundary

We state explicitly what this paper does not claim.

- We propose no tone-mapping method and no detector architecture. A0, A1 and A2 are affine maps; A3 is the camera's own curve; A5 is a re-implementation of a published learned mapper; R is TCNet's RAW row. Nothing in the arms is new.
- We do not claim that A0 is *better* than any per-frame map, and we do not claim that *no* tone map can beat it. The primary hypothesis is non-inferiority within a fixed margin against the per-frame maps actually tested, on one camera, one dataset, one detector family and one input size. A superiority outcome would be reported but was not pre-registered as a goal.
- We do not claim anything about adverse weather, other cameras, other bit depths or other detector families. The dataset has no per-frame weather labels (`audit/PRIOR_ART_NOVELTY_MATRIX.md` §C) and all frames come from one camera model.
- We do not claim that a fixed affine saves latency over an AGC that already runs on-camera. The latency table is descriptive and applies only where float counts are already streamed to the host (§8.3).
- We do not propose a new statistical estimator. The two-way cluster bootstrap, the fixed non-inferiority margin and the equivalence rules are standard; the contribution is in applying them pre-registered to a question where every prior row is single-run.
- The quantization mechanism itself is stated in N-AGC's prose [2]. Our contribution is to test it in cells where it is the only thing that changes.
- The oracle-headroom and cross-fitted selection machinery considered elsewhere in the audit is deliberately excluded; it answers a different question (is per-image routing headroom real?) and would put a second mechanism into the title.

### 2.3 Open literature item

The prior-art matrix (`audit/PRIOR_ART_NOVELTY_MATRIX.md` §A to §D) was built for the knowledge-distillation and weather-routing questions of the audited project and contains no row on tone mapping, bit depth or training on 16-bit thermal counts; the six works in §2.1 come from the council's search (`paper/COUNCIL_RECORD.md` §3.1), whose method is not recorded. Every "first" and "to our knowledge" in this draft is therefore conditional on a systematic search (arXiv, Semantic Scholar, IEEE Xplore; terms: thermal / LWIR / infrared with tone mapping / AGC / bit depth / 16-bit / radiometric, with object detection) that is `[PENDING]` and must be completed and logged before the OSF freeze.

---

## 3. Method

### 3.1 Notation and formal setup

Let a frame be a count image x in {0, ..., 2^b minus 1}^(H x W), with b = 14 per the dataset card and H = 512, W = 640 `[UNVERIFIED: b, H, W to be read from the files]`, with paired vendor AGC image j in {0, ..., 255}^(H x W) shipped with the dataset (its provenance relative to the TIFF is `[UNVERIFIED]`, §1.1), and label set y (boxes with classes in {person, car, bicycle}). Frames are grouped into videos v(x); frames of one video share scene, camera state, time and weather, so the exchangeable unit is the video, not the frame (`audit/MATH_STAT_REVIEW.md` §4.1).

A *tone map* is a function M from counts to real images that may depend on frame-level parameters phi(x):

M_a(x) = S_a(x; phi_a(x)),

where S_a is the map's functional form and phi_a(x) its per-frame parameters. The map *adapts* to the extent that phi_a depends on x; it has zero per-frame parameters if phi_a is constant across frames. All maps below are defined so that the nominal scene band lands in [0, 1] at float precision; values outside [0, 1] are retained at float precision and clipped only by the precision operator.

A *precision operator* Q_q acts on the mapped image. Q_float is the identity (float32 tensor). Q_8 is uniform 8-bit requantization:

Q_8(u) = round(255 clip(u, 0, 1)) / 255,

applied elementwise. The 8-bit arm therefore sees a float tensor whose values lie in {k/255 : k = 0, ..., 255}; it passes through the identical loader, augmentation and network as the float arm. The only difference between the two precision cells of an arm is the value set, and the per-frame fraction of pixels clipped by Q_8 is logged for every arm so that saturation and rounding can be distinguished descriptively.

An *initialisation* theta_0 is one of: random with 1 input channel (scratch-1ch); random with 3 input channels, the mapped image replicated to 3 channels (scratch-3ch, the channel-count control); or COCO-pretrained with 3 input channels, the mapped image replicated to 3 channels (pretrained-3ch). For the pretrained cells the input affine is pre-registered as the identity on the mapped image in [0, 1], because that is exactly the scale the library feeds the same weights when it loads an 8-bit grey image (uint8 divided by 255, replicated); any other choice would make the pretrained cell's outcome a function of an arbitrary constant. Lynred's exact input, a z-scored float image, is an optional cell (§6.1).

The detector f_theta is Ultralytics YOLOv8-s (version 8.4.178, pinned by the audit) with the stock configuration; the network is identical across arms except for the input channel count and initialisation that factor I prescribes.

### 3.2 The decompositions the cells estimate

Write the test-set metric of a trained detector as m(A, Q, I). The published contrasts cannot be written as one identity because their dtype is unstated; the design brackets them with two decompositions, each estimated from its own cells with seeds.

**Scratch decomposition (the mechanism at the detector that absorbs affines).** With R the full-range scaling x / (2^14 minus 1), and A0 the dataset affine clipped to [0, 1] at both precisions (§3.3),

m(A2, float, scratch) minus m(R, 8-bit, scratch)
= [m(A2, float, scratch) minus m(A0, float, scratch)]       (adaptivity term, delta_A; H1; both arms clipped)
+ [m(A0, float, scratch) minus m(A0, 8-bit, scratch)]       (rounding term within the dataset band, delta_Q1; H2; the two arms differ by the rounding operation alone)
+ [m(A0, 8-bit, scratch) minus m(R, 8-bit, scratch)]        (range-allocation term at fixed 8-bit output, delta_Q2; H2; bundles any saturation difference between the two maps and is named accordingly).

A fourth intervention sits outside the chain because A2 is itself a clipped map: the clipping term delta_C = m(A0u, float, scratch) minus m(A0, float, scratch), where A0u is the same affine without the clip (cell C21). R-8bit is the published RAW row's container (every scene written into the full 14-bit range before 8-bit rounding); A0-8bit is the same rounding with the container matched to the dataset band; A0-float removes the rounding; A2-float adds per-frame adaptivity. Each bracket changes exactly one declared operation (`audit_tools/preproc_controls.py`, verified by test T09 on synthetic counts). The identity is algebraic; the names on the brackets are claims about retraining interventions, not about information content, and each is interpreted only through its own confidence interval (§4.5).

**Pretrained decomposition (the published setting).** With A4 the vendor JPEG at its native 8 bits,

m(A4, 8-bit native, pretrained) minus m(R, float, pretrained)
= [m(A4, 8-bit native, pretrained) minus m(A0, float, pretrained)]   (vendor path versus fixed float affine at pretrained init; H3)
+ [m(A0, float, pretrained) minus m(R, float, pretrained)]            (input-scale term, delta_I; H3).

A0u-float and R-float carry identical information (R is an affine rescaling of the unclipped A0u), while the registered pretrained cell uses the clipped A0 (C10, in [0, 1], the library's own input range); the second bracket is therefore an input-scale-plus-clipping effect unless the optional A0u-pretrained cell (C23) is run, and is labelled so. It is a map-sensitivity effect under pretrained initialisation; it is attributed to pretraining itself only through the scratch-3-channel difference-in-differences of H3(b). Within those limits it is an effect that a from-scratch batch-normalised network would absorb and a pretrained network may not. The first bracket is the vendor path against the float affine at the published initialisation. The scratch-3ch cell (A0-float) separates channel count from initialisation within factor I. The claim in the conditional title is that delta_A is within the margin, delta_Q1 + delta_Q2 is large, and the float penalty at pretrained initialisation is a delta_I effect.

### 3.3 Arms of the adaptivity factor

All statistics marked "dataset" are computed on `images_thermal_train` only, from a fixed random sample of 10,000 pixel locations per frame, with the seed recorded. q_p denotes the p-th percentile; m_D the dataset median; s_D = q_99,D minus q_1,D the dataset 1-99 percent span. One formula per arm is used in both precision cells.

- **A0, dataset-wide affine (zero per-frame parameters).**
  S_0(x) = (x minus q_1,D) / s_D. The dataset's 1-99 percent band maps to [0, 1]; the dataset median lands at (m_D minus q_1,D) / s_D. Clipped to [0, 1] at both precisions, so that the float and 8-bit cells differ by the rounding operation alone; the clipping intervention is measured separately by the unclipped arm **A0u** (S_0 without the clip, float only, cell C21), and the clipped fraction is logged per frame. The same (q_1,D, s_D) are used at test time. The unclipped map is a constant affine and folds exactly into the first convolution's weights and bias on interior pixels (`fold_affine_into_conv`, test T09); the clip is a pointwise nonlinearity on the pixels outside the dataset band only, so a folded deployment corresponds to A0u. The value 0 is the bottom of the dataset band, which is what the loader's fill value 0 means in every arm (§3.4).
- **A1, per-video affine (the slow camera-state control).**
  S_1(x) = (x minus q_1,v(x)) / s_v(x), with the percentiles computed from the pooled sampled pixels of all frames of the frame's video v(x) within its split (train stills of that video for training frames; all frames of a test video for test frames). The map changes between videos but not between frames of one video, so it removes slow camera-state changes (FFC offset drift, gain drift with FPA temperature, a gain-state switch between recordings) without adapting to per-frame scene content. Residual pixel-wise non-uniformity is removed by no global map and is a common term across every arm. A1 is run in its **causal** form: frame t of a video is mapped with the percentiles of the pooled samples of frames 0 to t only (expanding window, `arm_A1_causal`), which is deployable; the offline whole-video statistic (`arm_A1_offline`) is an upper bound and is used only as a pilot diagnostic. Test T09 verifies that the causal map of frame t is unchanged by any later frame. For training stills, the per-video statistic is computed over the train stills of that video, which are not a temporal sequence; this is stated. A within-video gain-state switch would appear as bimodality of the per-frame median within a video and is checked on day 1 `[UNVERIFIED whether any occurs]`.
- **A2, per-frame affine (two per-frame parameters; the primary comparator).**
  S_2(x) = clip((x minus q_1(x)) / (q_99(x) minus q_1(x)), 0, 1). Clipped at both precisions (the 2 percent of pixels outside the frame's own 1-99 band are clipped and logged), so A2-float and A2-8bit differ by rounding alone; A2-8bit equals N-AGC's RAW-stretch [2] before its JPEG step and TCNet's "clip" [1]. This is the per-frame tone map whose float version is unambiguous: it adapts offset and gain to each frame, carries the full count information, and depends on no assumption about the vendor map and on no re-implementation.
- **A3, the vendor's global tone curve at count resolution (secondary comparator).**
  For each frame, fit a monotone non-decreasing step function g_x by isotonic regression of the JPEG value on the count,
  g_x = argmin over monotone g of (1/n) sum_{i=1}^{n} (j[p_i] minus g(x[p_i]))^2,
  on n = 10,000 pixel locations p_i sampled stratified over the frame's count quantiles, solved by pool-adjacent-violators. The fitted function is piecewise constant with plateaus whose values are block means of 8-bit targets, so applied as a step function it carries at most the information of the vendor's 8-bit image; this is why revision 1's "lossless" A3-float was not a float arm (§10, item 2). We instead take the knots (mean count of each plateau, fitted value of each plateau) and interpolate them with a strictly increasing piecewise-linear curve g~_x; S_3(x) = g~_x(x) / 255 in float32. Within each plateau's count range the map is strictly increasing, so count differences below the vendor's 8-bit step survive at float precision; the curve's *shape* is the vendor's, its *resolution* is the count's. Outside the vendor's occupied range (plateaus fitted at 0 or 255) the curve is extended flat, because the vendor clipped those counts and the clip is part of the vendor map; the clipped fraction is logged. The interpolation is an assumption (that the vendor curve is smooth between plateaus) and is stated as such; A3-8bit = Q_8(S_3) differs from the step-function fit by at most the rounding of the interpolated values, and this is verified on day 1 by comparing Q_8(S_3) to round(g_x(x)) / 255 `[test, PENDING]`.
  *Fit residual and its JPEG floor.* The per-frame residual r_x = RMS over the full frame of (j minus g_x(x)) mixes JPEG error, spatially local vendor processing (DDE or local contrast) and non-monotone operations. To separate them we estimate a JPEG-error floor per frame: j^_x = round(g_x(x)) is re-encoded with the quantisation tables read from the paired JPEG file `[UNVERIFIED: tables readable from the files and constant across the dataset]`, decoded to j^'_x, and f_x = RMS(j^'_x minus j^_x). The excess e_x = sqrt(max(r_x^2 minus f_x^2, 0)) is the residual not attributable to JPEG error at this content and quality. A3 keeps the label "vendor global tone curve" if the median e_x over `images_thermal_train` is at most one 8-bit grey level (the quantisation unit; design choice, Appendix A); otherwise A3 is relabelled "the global monotone component of the vendor map" before the freeze (failure case F2) and the full distributions of r_x, f_x and e_x are reported as a finding about the camera. A3 uses, at deployment, information the camera already produces (its own AGC output); it uses no labels.
- **A4, vendor AGC JPEG (descriptive only).**
  S_4(x) = j / 255. This arm ties our table to the published ones; it differs from A3-8bit by JPEG compression plus whatever the global fit misses, so the A4 minus A3-8bit difference is a descriptive estimate of the JPEG-plus-local-map effect. It is never the primary comparator [6]. Its first seed, run before the freeze, is the calibration run (§4.2).
- **A5, learned per-frame mapper, retrained per arm (3 seeds).**
  S_5(x) = M_psi(x), a convolutional mapper with output in [0, 1]^(H x W), trained jointly with the detector, re-implemented from the published description of TCNet [1] (or N-AGC [2] if its code is released). The audit recorded no architectural detail, loss composition, parameter count or FLOPs for either mapper `[UNVERIFIED]`; the architecture, parameter count, FLOPs, memory at the chosen batch size and the training objective of the re-implementation are `[PENDING]` and will be written into the OSF record before any run. The re-implementation is labelled "re-implementation, unreproduced numbers". If it does not fit in 11 GB alongside YOLOv8-s, the fallback is F4.
- **R, full-range linear scaling (the published RAW container).**
  S_R(x) = x / (2^14 minus 1), TCNet's RAW row [1]. At 8 bits (scratch) it is the container cell of the scratch decomposition; at float (pretrained) it is the published RAW setting under the float reading of TCNet's dtype.

### 3.4 Detector, objective and training

All arms train the same network with the Ultralytics 8.4.178 detection objective,

L(theta) = lambda_box L_CIoU + lambda_cls L_BCE + lambda_dfl L_DFL,

with the loss weights, optimiser, learning-rate schedule, warm-up and EMA at the Ultralytics defaults for the pinned version, recorded verbatim in each run's configuration file `[the numerical defaults are not restated here to avoid misquoting; they are logged from the library at run time]`. Fixed across every arm: image size 512 letterbox; fill value 0 in the mapped domain for letterbox padding and mosaic canvas (the library's default fill is a mid-grey constant `[UNVERIFIED for the pinned version]` and is overridden in every arm); epochs = 100, the library default `[council planning used 50; changed after review, see §10 item 6]`; batch size `[PENDING: set by the pilot for 11 GB, then fixed]`; geometric augmentation at library defaults (mosaic, scale, translate, horizontal flip; numerical values logged); *no photometric augmentation* in any arm (`hsv_h = hsv_s = hsv_v = 0`, and no added brightness or contrast jitter), because any per-image affine jitter is itself a randomised per-frame map that would erase the distinction between A0, A1 and A2 (revision 1's jitter is removed; a jitter-on sensitivity replicate is an optional cell, §6.1). For A5 the mapper parameters psi are optimised jointly with theta by the same objective.

Convergence is checked but not used for selection: for every run the validation mAP50-95 curve is released, and the slope over the final 20 epochs is reported in the main table so that under-training is visible. The epoch count is not changed after the freeze.

Checkpoint rule: the final-epoch EMA weights (`last.pt`) are evaluated; `best.pt` selection on the validation split is never used, so `images_thermal_val` serves only for monitoring, for the pre-freeze feasibility estimate (§4.2) and for the S1 operating point.

**Why batch normalisation absorbs a dataset-wide affine, and why it does not absorb A1 or A2.** The YOLOv8 stem is a convolution followed by batch normalisation and SiLU. For an input affine u to alpha u + beta, conv(alpha u + beta) = alpha conv(u) + beta sum(w) per output channel in the interior of the image; at padded borders the fill value 0 is not transformed by the affine, so the border response differs by a term proportional to beta. Batch normalisation then removes the per-channel scale and offset using batch statistics, up to its variance floor eps (1e-3 in Ultralytics `[UNVERIFIED for the pinned version]`), so the invariance is exact only when the pre-BN channel variance is large relative to eps; the pre-BN variance of the stem on a training batch will be logged in the pilot for each arm. The post-BN activation is therefore invariant to a *dataset-wide* affine up to border and eps effects, at train time through batch statistics and at test time through running statistics. This is an invariance of the function class, not of the optimisation trajectory: the effective learning rate on the stem scales with alpha, so A0 is defined to land the dataset band on [0, 1], the scale the library's hyperparameters were tuned for. A per-video (A1) or per-frame (A2) affine is not removed by BN, because BN subtracts a batch-level statistic; such a map changes the information the network sees, which is exactly why A1 and A2 are informative arms rather than redundant ones.

### 3.5 Train-time versus test-time information

| Quantity | Computed from | Available at deployment? | Uses labels? |
|---|---|---|---|
| q_1,D, s_D (A0) | `images_thermal_train` pixels | yes, constants | no |
| q_1,v, s_v (A1) | all frames of the video in its split | as a causal running estimate (offline here) | no |
| q_1(x), q_99(x) (A2) | the frame itself | yes, O(HW) per frame | no |
| g_x, g~_x, r_x, f_x, e_x (A3) | the frame's own TIFF and vendor JPEG | yes, the camera's AGC output | no |
| psi (A5) | training frames and labels (joint training) | yes, fixed weights | yes, at training only |
| DR tercile of a frame | the test frame itself (pixel stage) | yes | no |
| Isotonic map movement m_t (S1 dose) | consecutive test frames (pixel stage) | yes | no |
| Flicker rate, confidence SD (S1) | test labels (label stage) | evaluation only | yes |

No arm uses test labels for anything but evaluation. No arm uses `video_thermal_test` statistics at training time. The test split is opened in two registered stages (§4.8): a pixel stage (manifest, video count K, DR terciles, A3 curves, m_t and the S1 gate), whose outputs are written into the OSF record, and a label stage, entered once after every budgeted run has finished.

### 3.6 Algorithm box

```
Inputs : paired (TIFF x, JPEG j, labels y) for train / val / test; arms A in {A0,A1,A2,A3,A4,A5,R};
         precision Q in {float, 8bit}; init I in {scratch-1ch, scratch-3ch, coco-3ch}; seeds S.
Output : per-cell mAP50-95 per seed; two-way (video x seed) paired bootstrap CIs; covariates.

0  Day-1 checks (CPU)
   0.1 Manifest every frame: path, SHA-256, size, bit depth, max count, video id,
       boxes per class. Assert TIFF/JPEG stem pairing for every split.
   0.2 Assert no video id appears in more than one of train / val / test.
       Record K_test (videos in video_thermal_test) and frames per video.
   0.3 Record the licence text verbatim.
   0.4 Kill condition F1: TIFFs missing or unpaired for video_thermal_test.
   0.5 Per-video median series: flag within-video bimodality (gain-state switch).
1  Dataset statistics (train only): sample 10k pixels per frame with a fixed seed;
   q_1,D, s_D; per-video q_1,v, s_v for every split.
2  Per-frame A3 curves (all splits, CPU): isotonic fit of j on x (10k stratified pixels,
   PAVA); knots; strictly increasing piecewise-linear g~_x stored as a 2^14-entry float32
   table (64 kB/frame, the only cached map); r_x, f_x (JPEG re-encode of round(g_x(x))
   with the file's quantisation tables), e_x. Decide the A3 label (median e_x rule).
   Covariates per frame: DR(x) = q99(x) - q1(x); clipped fraction under Q_8 per arm;
   histogram-weighted map movement m_t (S1).
3  Calibration run (pre-freeze): C13 seed 0 (A4, scratch-1ch, full train, 100 epochs).
   Record wall time per epoch, loader throughput, pre-BN stem variance; evaluate on val;
   val video-cluster bootstrap half-width h_val; predicted h_test from K_test (S4.2).
   Feasibility rule S4.2 decides test-set enlargement or re-scoping before the freeze.
4  Freeze: OSF record with margin delta = 1.0, decision rules, cell list and drop order,
   epoch count, batch size, A3 label, tercile boundaries, K_test, analysis code hash.
5  For each cell (A, Q, I) and seed s in S:
   5.1 Loader yields float32 tensors computed on the fly (A0, A1, A2, R by formula;
       A3 by table look-up; A4 by JPEG decode); apply Q_8 if required; replicate to
       3 ch if I is 3-channel.
   5.2 Train YOLOv8-s 100 epochs, logged config, identical augmentation (geometric only).
       For A5 the mapper is trained jointly.
   5.3 Evaluate last.pt on images_thermal_val (monitoring; val curve released).
6  Label stage: open video_thermal_test labels once. For every cell and seed: inference
   at conf 0.001, NMS IoU 0.7, max 300 detections; store detections; run
   COCOeval.evaluate() once per model and store the per-image match records.
7  Two-way paired bootstrap: draw B = 1000 video index sets (K_test videos with
   replacement) ONCE, shared by every cell and seed; for each replicate and each cell,
   draw n_c seeds with replacement from that cell's seeds; the cell value is the mean
   over drawn seeds of the mAP50-95 obtained by re-accumulating the stored per-image
   match records of the resampled videos (duplicated videos re-indexed); differences
   between cells use the same replicate; percentile CIs. Report K and the frame count.
8  Apply the pre-registered decision rules (H1..H4), then the S1 gate and, if it passes,
   the descriptive flicker analysis.
```

### 3.7 Complexity and resource plan

- Per-frame maps: A0, A1, A2 and R are O(HW) and computed in the loader; A2 needs two percentiles, O(HW) with a 2^14-bin histogram; A3 needs a sort of n = 10,000 sampled pixels and a linear-time PAVA pass, O(n log n) per frame on CPU once, plus an O(HW) table look-up to apply; A4 is a JPEG decode; A5 adds the mapper's forward pass `[PENDING: FLOPs of the re-implemented mapper]`.
- Inference cost of A0 is zero beyond the first convolution once folded; A1 adds a running per-video statistic; A2 two percentiles per frame; A3 and A4 are what the camera already does; A5 adds a network. This column is descriptive (§8.3).
- Training: `[PENDING: measured by the calibration run]`. The only prior figure on this machine is the reproduction protocol's estimate of about 6 GPU-hours for a 50-epoch stock YOLOv8-s run (`audit/REPRODUCTION_PROTOCOL.md` §5), which extrapolates linearly to about 12 GPU-hours at 100 epochs; the council's 5-hour figure for pretrained runs has no recorded basis and is not used. With 49 budgeted runs (§6.1) the extrapolated total is about 590 GPU-hours, about 12 days on two RTX 2080 Ti or about 25 days on one; GPU 0 hosts the display and its availability is `[UNVERIFIED]`, so both figures are stated and the drop order in §6.1 is pre-registered. The loader (16-bit TIFF decode, float mosaic, per-frame statistics) with at most 4 workers on a shared CPU may be the bottleneck; the calibration run measures it.
- Evaluation: the bootstrap re-accumulates stored per-image match records (COCOeval.evaluate() is run once per model); the per-replicate cost is the accumulate step only. CPU time `[PENDING: measured on the calibration model]`. A unit test asserts that a replicate using the identity index set reproduces the full-set mAP exactly.
- Disk: about 17 GB TIFF plus about 16 GB JPEG `[council estimate, unverified until the archive is extracted]`; no cached float frames (revision 1's float cache would have needed about 1.3 MB per frame per arm, which does not fit in 76 GB); A3 tables 64 kB per frame, order 1 GB for the whole dataset `[arithmetic on an unverified frame count]`; checkpoints about 0.1 GB each, last-epoch only, about 5 GB for 49 runs.

---

## 4. Statistical design

### 4.1 Estimand

For a cell c = (A, Q, I), let mu_c be the expected COCO mAP50-95 of a detector trained with configuration c, where the expectation is over the training seed and the evaluation is on the population of videos from which `video_thermal_test` is drawn. The primary estimand is the difference

Delta_1 = mu(A0, float, scratch-1ch) minus mu(A2, float, scratch-1ch).

mAP is a set-level metric, so it is computed on the union of frames in a (resampled) set of videos, never per frame, and never averaged over frames.

### 4.2 Units, clustering, the two-way bootstrap, and feasibility

The exchangeable unit for test sampling is the video; the unit for training variability is the seed. Both sources enter the CI. Uncertainty is estimated by a two-way paired bootstrap: B = 1000 replicates; each replicate draws K videos with replacement from the K videos of `video_thermal_test` `[K UNVERIFIED until the manifest exists; the audit recorded two different frame counts for the video partition, 3,742 per [2] and 1,144 per the matrix row A10, and the manifest decides]`, the same video index sets being used for every cell and every seed so that differences are paired by video; and seed indices drawn **jointly** for the cells in a contrast: seeds are paired by index (seed s fixes the data order, augmentation stream and initialisation across arms), so each replicate draws n seed indices with replacement from the common index set of the two cells being contrasted (5 for C1 versus C5; the shared indices 0 to 2 when a 3-seed cell is involved, the 5-seed cell's extra seeds entering only its across-seed SD and a sensitivity CI). The cell value in a replicate is the mean over the drawn seeds of the mAP on the resampled frame set (duplicated frames given fresh image ids so pycocotools counts them); differences between cells are taken within the replicate. The point estimate is the difference of the seed means on the full test set. The 95 percent CI is the 2.5 and 97.5 percentile of the B replicate values. Unbalanced seed counts (5 versus 3) are handled by the joint draw over shared indices as stated; the independent per-cell seed draw, which discards the seed covariance, is reported alongside as a sensitivity check because seeds of different cells are independent models. Two components are reported beside the CI: the video-only CI of the seed-averaged difference (test-sampling component) and the per-seed values with their SD (training component). With 3 to 5 seeds the seed bootstrap is a coarse estimate of the training component and is labelled as such; it is included because omitting it, as revision 1 did, understates the CI by a term comparable to the margin. K and the frame count are stated beside every CI. No significance test is ever run on frames.

**Feasibility before the freeze (day-1 kill condition).** The primary test cannot be conclusive if the CI half-width h exceeds the margin delta. K_test is read from the manifest on day 1. The calibration run (C13 seed 0, which is reused as a budgeted model) is evaluated on `images_thermal_val` with a video-cluster bootstrap over the K_val videos of val, giving h_val; the per-video contribution to the variance is then rescaled to K_test to give a predicted h_test (a heuristic that ignores differences in frames per video and in seed variance; stated as such). Rule: if predicted h_test > delta, the frozen test set is enlarged before the freeze by holding out additional whole videos from the stills (by file-name video id, removed from training) until predicted h_test <= delta, recorded as a pre-registered deviation; if that would remove more than one fifth of the training videos, the primary contrast is declared "inconclusive by design" in the OSF record before any budgeted run, and the paper is re-scoped to the quantization and initialisation terms, which are larger effects. The minimum K is therefore whatever the rule produces, not a number chosen now.

### 4.3 Seeds

Seed counts are fixed in advance and not decided by the pilot: A0-float and A2-float at scratch-1ch (the two primary cells) at 5 seeds; every other budgeted cell at 3 seeds. All seeds are reported, including diverged runs; no best-of-seeds selection [10]. Across-seed SD is reported for every cell.

### 4.4 Non-inferiority margin

delta = 1.0 mAP50-95, fixed now and written into the OSF record. It is a design choice, not a demonstrated practically important difference, and is defended as such; a failure to reject a difference is never reported as equivalence (the TOST-style rules of §4.5 require the whole CI inside the margin), and the calibration run's predicted CI half-width h_test is written into the record beside delta so that the design's resolving power is stated before any result exists (§4.2). Reasons (design choice, Appendix A): a margin must be fixed on practical grounds rather than estimated, because an across-seed SD from 3 to 5 runs has a sampling uncertainty of roughly a factor two in either direction and would make the margin a random variable; 1.0 is below the smallest published AGC-versus-classical-map gap on this protocol (1.4, Fieldscale [3], single run, metric unrecorded), so a loss smaller than delta is one that the existing literature does not itself treat as a meaningful difference; and it is the resolution at which single-run tables in this literature report differences. The across-seed SD of each cell is reported next to delta so that the reader can see whether the margin is above or below the training noise; it plays no role in the decision rule.

### 4.5 Pre-registered hypotheses and decision rules

Let hat Delta be the point estimate and [L, U] its two-way paired bootstrap 95 percent CI. All rules are CI-based; no rule uses a point estimate alone.

- **H1 (primary, non-inferiority).** Delta_1 = mu(A0-float, scratch-1ch) minus mu(A2-float, scratch-1ch).
  Non-inferior if L > minus delta. Inferior (H1 fails) if U < minus delta. Inconclusive otherwise (and, by construction, whenever h > delta). Secondary comparators, same rule, reported in the same table: A0-float versus A3-float (the vendor curve at count resolution, 3 seeds) and A0-float versus A5-float (the learned mapper, 3 seeds). *Discordance rule:* the headline follows H1 only; a secondary that is inferior or inconclusive narrows the scope of the claim ("non-inferior to a per-frame affine; `[not / inconclusively]` to the vendor curve / the learned mapper") and never reverses it. If A3 was relabelled under F2, the A3 secondary is reported as "global component of the vendor map".
- **H2 (requantization terms; planted positive control).** Delta_2a = mu(A0-float, scratch-1ch) minus mu(A0-8bit, scratch-1ch) (rounding within the dataset band; the two arms differ by the rounding operation alone); Delta_2b = mu(A0-8bit, scratch-1ch) minus mu(R-8bit, scratch-1ch) (range allocation at fixed 8-bit output, bundling any saturation difference); Delta_2 = Delta_2a + Delta_2b = mu(A0-float) minus mu(R-8bit). Delta_C = mu(A0u-float) minus mu(A0-float) (clipping; cell C21, tier 2) is reported beside them and is not part of the chain.
  *Positive control:* the pipeline is declared able to resolve differences larger than the margin if L_2 > delta. No equivalence or non-inferiority statement from H1 or H4 is interpreted as evidence of equality unless the positive control has passed. Each term is reported with its own CI and no threshold is imposed on any; revision 1's 5-point threshold had no source and is withdrawn (§10, item 16).
  *Attribution rule:* the conditional title's claim that requantization, not adaptivity, explains the RAW penalty is supported if L_2 exceeds max(|L_1|, |U_1|), that is, if the smallest plausible rounding-plus-range-allocation effect exceeds the largest plausible adaptivity effect; it is weakened if the two CIs overlap and withdrawn if U_2 < delta. The word "quantization" is used for Delta_2a only if its own CI excludes zero; if Delta_2a is small and Delta_2b large, the penalty is reported as a range-allocation (mapping) effect at 8 bits, not as a rounding effect, and the title is changed accordingly before submission.
- **H3 (initialisation; the Lynred effect isolated and the published setting reproduced).**
  (a) Delta_3 = mu(A0-float, pretrained-3ch) minus mu(A2-float, pretrained-3ch): the **map sensitivity under pretrained initialisation**. On its own it compares two maps within one initialisation and says nothing about pretraining; the Lynred effect is confirmed only if U_3 < minus delta **and** the interaction in (b) is supported. Delta_3 is also reported relative to the comparator's mean, because the pretrained and scratch cells may sit at different absolute mAP levels.
  (b) Interaction, difference-in-differences within the 3-channel architecture: Delta_DiD = [mu(A0-float, scratch-3ch) minus mu(A2-float, scratch-3ch)] minus [mu(A0-float, pretrained-3ch) minus mu(A2-float, pretrained-3ch)] = (C15 minus C22) minus (C10 minus C12), computed within replicates with shared video and seed draws; the pretraining x map interaction is supported if its CI excludes 0. This replaces revision 2's (Delta_1 minus Delta_3), which compared a 1-channel scratch contrast with a 3-channel pretrained one and so confounded channel count with initialisation.
  (c) Published setting: Delta_P = mu(A4-native, pretrained-3ch) minus mu(R-float, pretrained-3ch), with its two brackets (§3.2) each reported with a CI; the input-scale bracket mu(A0-float, pre) minus mu(R-float, pre) is the pure Lynred term. A4-native-pretrained is also the one cell that reproduces the configuration of the published AGC rows (8-bit AGC into a pretrained detector).
  (d) Channel control: mu(A0-float, scratch-3ch) minus mu(A0-float, scratch-1ch) with CI inside [minus delta, delta] establishes that the channel count alone does not move A0; otherwise factor I is reported as "initialisation plus channel count" and H3(a) is interpreted against the scratch-3ch cell instead.
  H3 fails if the pretrained network shows no float penalty (L_3 > minus delta); this is reported as a non-replication of [4] on FLIR, and the scratch cells still identify quantization versus adaptivity.
- **H4 (adaptivity buys nothing at float; dynamic-range interaction only at 8 bits).**
  (a) Equivalence family: for each of the ten pairs among A0, A1, A2, A3, A5 at float, scratch-1ch, the pair is *equivalent* if [L, U] lies inside [minus delta, delta], *different* if [L, U] lies entirely outside it, and *inconclusive* otherwise (TOST-style). The ten CIs are reported without multiplicity adjustment and labelled a descriptive family.
  (b) Dynamic-range difference-in-differences: with frames binned into per-frame dynamic-range terciles (DR(x) = q_99(x) minus q_1(x), boundaries fixed at the pixel stage and registered), define for a cell c the tercile gap G_c = mAP_c(low) minus mAP_c(high). Low- and high-DR terciles differ in scene content, so G_c alone is confounded; the identified quantity is D_8 = G(A0-8bit) minus G(A2-8bit), predicted negative (a dataset-wide container gives the fewest grey levels to low-DR scenes, a per-frame stretch gives them the full container) with CI excluding 0, and D_f = G(A0-float) minus G(A2-float), predicted to have its CI inside [minus delta, delta]. Tercile subsets are evaluated with the same two-way bootstrap, tercile membership fixed per frame.
- **S1 (descriptive secondary, gated).**
  Gate: on `video_thermal_test` at the pixel stage, the vendor curve moves between consecutive frames by more than its fit residual, m_t > max(r_t, r_{t minus 1}), on at least 10 percent of consecutive frame pairs, where m_t is the count-histogram-weighted RMS difference of the two curves, m_t^2 = sum over counts c of w_c (g_t(c) minus g_{t minus 1}(c))^2 with w_c proportional to h_t(c) + h_{t minus 1}(c), the pooled count histogram of the two frames (so that sparsely populated tails, where isotonic fits are unstable, do not inflate m_t). If the gate fails, the finding is "FLIR's shipped AGC is temporally damped on this split" and no flicker claim is made. If the gate passes: for A0-float, A3-float and A4 (scratch-1ch), with the operating point per arm and seed chosen on val by maximal F1, (i) flicker rate = number of ground-truth objects detected at t minus 1 and t + 1 but not at t, divided by the number of interior track frames, (ii) per-object confidence SD along ground-truth tracks of length at least 5, and (iii) the dose-response of (i) and (ii) against m_t. Ground-truth tracks come from annotation track ids if present `[UNVERIFIED]`, otherwise from greedy IoU >= 0.5 linking of consecutive ground-truth boxes, which is label-derived and stated as such. CIs are the two-way bootstrap and are labelled descriptive; the video count is stated.

### 4.6 Falsifier of the conditional title

H1 fails if U_1 < minus delta, that is, if A0-float trails A2-float by more than the margin with the two-way CI excluding the margin, and the failure is concentrated in the low-DR tercile. Then per-frame adaptivity is a genuine detector requirement at float precision, TCNet and N-AGC are justified, and the paper is rewritten as a multi-seed, CI-bearing, precision-controlled confirmation of that requirement; the title changes, the tables do not. H2 fails if U_2 < delta: quantization is not the mechanism. H3 fails if L_3 > minus delta: Lynred's prose result does not replicate on FLIR.

### 4.7 Minimum detectable effect and multiplicity

Next to every null or equivalence statement (H1 non-inferiority, H4a equivalence, H4b float DiD, S1 if null) we report delta, the across-seed SD of each cell, and the two-way CI half-width h of the relevant difference; h is the smallest loss the design could have detected. If h > delta the corresponding statement is "inconclusive", a pre-registered third outcome, not a null.

There is one primary contrast (H1, A0 versus A2 at float, scratch-1ch). H2 (positive control and two terms) and H3 (a to d) are pre-registered supporting contrasts, each interpreted by its own rule. The H1 secondaries, the H4a family and the H4b DiDs are reported uncorrected and labelled descriptive; none supports a headline claim on its own. S1 is descriptive.

### 4.8 Blinding and freeze

The protocol, the arms, the margin, the epoch count, the batch size, the A3 label, the cell list with its drop order, the decision rules and the analysis code (by commit hash) are frozen and registered on OSF before any budgeted run starts, with the exception of the calibration run, which precedes the freeze and whose val-based outputs (throughput, h_val, predicted h_test) are themselves written into the record. The test split is opened in two registered stages: the pixel stage (manifest, K_test, DR tercile boundaries, A3 curves, m_t, S1 gate outcome), whose outputs are added to the OSF record before the label stage; and the label stage, entered exactly once, after every budgeted training run has finished. The pilot (§5.5) uses a video-grouped subset of `images_thermal_val` only and is a functional test, not a source of any number in the record.

---

## 5. Datasets, protocol, implementation, licences

### 5.1 Data

FLIR ADAS v2 (Teledyne FLIR) [7]: 640 x 512, FLIR Tau 2, TIFF counts (14-bit per the dataset card) and 8-bit AGC JPEG per frame, COCO-format JSON. The frame counts recorded during the audit do not agree: the matrix row for the dataset gives "26,442 annotated frames (9,711 thermal stills and 7,498 video frames)", whose two parts sum to 17,209, not 26,442 (the larger figure may include RGB frames `[UNVERIFIED]`); the matrix row A10 gives thermal train/val/video-test = 10,742 / 3,749 / 1,144 for an 8-class subset; N-AGC [2] gives 3,742 frames for the video-test partition. None of these was verified on this machine; the manifest produced on day 1 (`audit_tools/verify_manifest.py`: per-file SHA-256, TIFF mode and max count, JPEG pairing and dimensions, video id, exact and near-duplicate frames across splits, box geometry, category ids) is the only count this paper will report, and the discrepancy is noted so that no published count is quoted as ours. Classes are restricted to person, car and bicycle through an explicit category map, following [1] and [3]. Splits: `images_thermal_train` for training, `images_thermal_val` for monitoring, the feasibility estimate and the S1 operating point, `video_thermal_test` for the single final evaluation. Video ids are parsed from file names (`video-<id>-frame-<n>`) and the three splits are asserted disjoint by video id before anything is trained; if the official still splits overlap by video, a video-grouped validation split is carved from train and the official val is kept as a second monitoring set (`audit/REPRODUCTION_PROTOCOL.md` §3).

Licence: the FLIR licence text could not be retrieved during the audit; a third-party CC BY-NC-SA 4.0 statement exists but is `[UNVERIFIED]`. The data are obtained by registration, stored outside any repository and never redistributed. We release manifests with SHA-256 per file, the code, run configurations, logs, the bootstrap index sets and the analysis notebook unconditionally; the per-frame isotonic tone curves, residuals, covariates and detection files are released only if the licence permits derived data, which is `[UNVERIFIED]`. If it forbids per-frame derived data, curves and covariates are released as per-video summaries; if it forbids any derived data, the paper's reproducibility asset reduces to code, hashed manifests, configurations and logs, and the paper says so (F11).

No external dataset is used. The 16-bit automotive set ZUT-FIR-ADAS [12] would allow an external check of the precision effect on a second camera, but its archive (about 95 GB) does not fit on the available disk and it is single-class; UNIRI-TID [13] is 8-bit and therefore cannot host a float arm. Both are listed as future work, not as arms.

### 5.2 Fixed protocol (applies to every cell)

- Detector: stock Ultralytics `yolov8s.yaml`, version 8.4.178 pinned; `ch = 1` from scratch; `ch = 3` from scratch on replicated input (channel control); or `yolov8s.pt` COCO weights with 3 input channels on replicated input in [0, 1].
- Input: letterbox 512, fill value 0 in the mapped domain for padding and mosaic; float32 tensors through one patched loader for every arm `[implementation item: the library's uint8 pipeline (mosaic, letterbox, geometric augmentation) is patched for float input]`. Loader control: a deterministic unit test feeds the A4 image through the patched float loader and through the library's native uint8 path with the same augmentation seed and asserts elementwise agreement within 1/255 on a fixed batch of frames; this test, not a trained-mAP comparison, is the control, because a trained one-seed comparison cannot resolve a bug of the margin's size. A trained native-path replicate of C13 at one seed is an optional descriptive cell (L0).
- Augmentation: library defaults for geometry (mosaic, scale, translate, horizontal flip; values logged); all photometric augmentation off (`hsv_h = hsv_s = hsv_v = 0`); no added jitter. Jitter-on sensitivity replicates of C1 and C5 at 3 seeds are optional cells (J1, J2; failure case F7).
- Training: 100 epochs, batch `[PENDING from pilot, then fixed]`, optimiser and schedule at library defaults, EMA on, final-epoch weights; val curve released; last-20-epoch slope reported.
- Evaluation: pycocotools mAP50-95 (primary), AP50, AP_small (COCO area below 32^2), per-class AP; conf 0.001, NMS IoU 0.7, max 300 detections (`audit/MATH_STAT_REVIEW.md` §4.3); evaluation in `eval()` mode (the audit found an offline evaluator that ran BatchNorm in train mode, `audit/CODE_AUDIT_2026.md` C5; our evaluator is the library's plus pycocotools, and a unit test asserts that running statistics are not mutated by evaluation). AP50 is reported for every cell so that the published AP50 rows of [2] can be compared at their own metric; no threshold in this paper is derived from a published number.
- Class map: one constant `CLASSES = ("person", "car", "bicycle")` with the explicit FLIR category ids, used by every script (audit finding C6).
- Latency: batch-1 fp32 on RTX 2080 Ti for A0 (folded), A1, A2, A3, A5 including the map; FP16 or TensorRT only if available and then stated. Descriptive only (§8.3). No real-time claim is made from fp32 figures.
- Hardware: 2 x RTX 2080 Ti (11 GB), GPU 1 preferred (GPU 0 hosts the display; its availability for training is `[UNVERIFIED]`), 16 CPU cores, 62 GB RAM, 76 GB free disk at audit time; other users' CPU jobs share the machine, so `num_workers <= 4` (`audit/REPRODUCTION_PROTOCOL.md` §7).

### 5.3 Implementation notes carried over from the audit

The audited student repository is not used for any arm: its custom head is roughly twice the size of the stock YOLOv8-s (21.5 M versus 11.2 M parameters, T01/T07), its offline evaluator ran in train mode (C5), its notebook loaded no weights (C1), and it has three class-index conventions (C6). The lessons are encoded as tests in this project: checkpoint round-trip equality, eval-mode invariance of running statistics, a single class map, a pycocotools agreement test for any in-house metric code (T03 showed the audited `metrics.py` agrees with pycocotools after patches; we use pycocotools directly), the loader equivalence test (§5.2), the bootstrap identity test (§3.7) and the A3 rounding test (§3.3).

### 5.4 Software and versions

Python 3.10.12, PyTorch 2.6.0+cu124, torchvision 0.21.0, Ultralytics 8.4.178, pycocotools, scikit-learn (isotonic regression) `[version PENDING]`, Pillow or libjpeg for quantisation-table readout and re-encoding `[PENDING]`, NVIDIA driver 535.309.01. A `requirements-lock.txt` is generated with `pip freeze` and released. Every run directory stores `config.json`, `git_commit.txt`, `pip_freeze.txt`, `stdout.log`, `metrics.json` and the val curve.

### 5.5 Pilot

Before the freeze: a video-grouped subset of `images_thermal_val` (on the order of 200 train frames from at most 10 videos and 50 held-out frames from 3 other videos, 2 epochs; sizes from `audit/REPRODUCTION_PROTOCOL.md` §4) is used to confirm that the float loader, every map, the 8-bit path, both 3-channel paths and the mapper fit in memory and run end to end; that pycocotools evaluation works; that the loader equivalence, bootstrap identity and A3 rounding tests pass; and to fix the batch size. The pilot sets no threshold, no metric, no seed count and no margin, and its numbers are never reported as results. The 3-cluster bootstrap that revision 1 planned on the pilot is dropped as uninformative.

---

## 6. Experiment plan

### 6.1 Cells

Budgeted cells (run in tier order; tier 1 before any tier 2 run starts):

| Cell | A | Q | I | Seeds | Role | Tier |
|---|---|---|---|---|---|---|
| C1 | A0 dataset affine | float | scratch 1-ch | 5 | H1 arm; H2; H3(b),(d); H4 | 1 |
| C5 | A2 per-frame affine | float | scratch 1-ch | 5 | H1 primary comparator; H3(b); H4 | 1 |
| C2 | A0 | 8-bit | scratch 1-ch | 3 | H2 rounding term; H4b | 1 |
| C14 | R full-range | 8-bit | scratch 1-ch | 3 | H2 container term; positive control (TCNet RAW container) | 1 |
| C10 | A0 | float | COCO 3-ch | 3 | H3(a),(b),(c) | 1 |
| C12 | A2 | float | COCO 3-ch | 3 | H3(a) comparator | 1 |
| C9 | A5 learned mapper | float | scratch 1-ch | 3 | H1 secondary comparator; H4a | 2 |
| C7 | A3 vendor curve | float | scratch 1-ch | 3 | H1 secondary comparator; H4a; S1 | 2 |
| C13 | A4 vendor JPEG | 8-bit native | scratch 1-ch | 3 | descriptive tie-in to [1,2,3]; S1; seed 0 = calibration run | 2 |
| C16 | A4 vendor JPEG | 8-bit native | COCO 3-ch | 3 | H3(c): the published AGC setting | 2 |
| C17 | R full-range | float | COCO 3-ch | 3 | H3(c): the published RAW setting (float reading) | 2 |
| C15 | A0 | float | scratch 3-ch | 3 | H3(d) channel-count control; H3(b) DiD | 2 |
| C22 | A2 | float | scratch 3-ch | 3 | H3(b) DiD partner of C15 (added 16 Oct after review) | 2 |
| C21 | A0u (unclipped dataset affine) | float | scratch 1-ch | 3 | clipping intervention Delta_C (added 16 Oct after review) | 2 |
| C3 | A1 per-video affine | float | scratch 1-ch | 3 | slow camera-state control; H4a | 3 |
| C6 | A2 | 8-bit | scratch 1-ch | 3 | H4b DiD; N-AGC RAW-stretch without JPEG | 3 |
| C8 | A3 vendor curve | 8-bit | scratch 1-ch | 3 | C13 minus C8 = JPEG plus local component (descriptive) | 3 |

Optional cells, run only if time allows after tier 3, in this order, and stated as such: C11 (A0, 8-bit, COCO 3-ch; completes the 2 x 2 at A0), C18 (A3, float, COCO 3-ch), C19 (z-scored float counts, COCO 3-ch; Lynred's exact input [4]), C23 (A0u, float, COCO 3-ch; the pure input-scale bracket of §3.2), C4 (A1, 8-bit, scratch 1-ch), J1 and J2 (C1 and C5 with photometric jitter on, 3 seeds each; F7 sensitivity), C20 (COCO weights loaded into a 1-channel stem `[UNVERIFIED whether the library supports it without re-initialising the stem]`), L0 (C13 through the native uint8 loader, 1 seed, descriptive).

Budget: 43 scratch runs plus 12 pretrained runs = 55 budgeted runs (C21 and C22 added after the 16 Oct review). Per-run time `[PENDING: calibration run]`; the linear extrapolation of the protocol's 50-epoch estimate gives about 12 GPU-hours per run and about 660 GPU-hours in all, about 14 days on two GPUs or 28 on one (§3.7). Tier 1 alone is 22 runs, about 265 GPU-hours. Drop rule, pre-registered: if the measured per-run time implies that the full budget exceeds the available window, tiers are dropped from the bottom (tier 3, then tier 2 from C15 upward); tier 1 is never cut, and the epoch count and seed counts are never reduced to fit. Any dropped cell is listed in the paper as "not run (budget)".

### 6.2 Controls

- Positive control: C1 versus C14 (H2), the full-range requantization that the published RAW rows contain, with its two parts C1 versus C2 and C2 versus C14.
- Slow camera-state control: C3 (A1) versus C1 (A0); if A1 is superior to A0 by the H4a rule, the wording "no per-frame parameters" is kept but the mechanism statement adds "after a per-video affine", and F6 applies.
- Channel control: C15 versus C1.
- Published-setting cells: C16 and C17 against C10 (H3c).
- Loader control: the deterministic equivalence test (§5.2); L0 optional.
- Augmentation sensitivity: J1, J2 (optional).
- JPEG-confound control: C13 versus C8 (descriptive).
- Map-fidelity control: the distributions of r_x, f_x and e_x; the A3 label rule (§3.3, F2).

### 6.3 Results tables (to be filled)

Table 1, main table: rows = every run cell, columns = mAP50-95 per seed, mean, across-seed SD, AP50, AP_small, per-class AP, last-20-epoch val slope, all `[PENDING EXPERIMENT]`.
Table 2, pre-registered contrasts: H1 (A0 vs A2; secondaries A0 vs A3, A0 vs A5), H2 (Delta_2, Delta_2a, Delta_2b), H3 (a to d), H4a family, H4b DiDs, each with hat Delta, two-way [L, U], video-only [L, U], delta, h, K, frame count, outcome, all `[PENDING EXPERIMENT]`.
Table 3, dynamic-range terciles: per-tercile mAP for every scratch cell with CIs and the two DiDs, `[PENDING EXPERIMENT]`.
Table 4, A3 fidelity: distributions of r_x, f_x and e_x per split, per-video summary, fraction of frames with e_x above one grey level, the A3 label decision, `[PENDING EXPERIMENT]`.
Table 5, S1 gate and, if passed, flicker rate, confidence SD and dose-response, `[PENDING EXPERIMENT]`.
Table 6, descriptive latency per map on one named device, `[PENDING EXPERIMENT]`.
Table 7, feasibility record: K_test, frames per video, h_val, predicted h_test, the rule outcome, measured per-run time and loader throughput from the calibration run, `[PENDING EXPERIMENT]`.
Figure 1, the mechanism: per-frame occupied count range and the number of distinct grey levels after Q_8 under R, A0 and A2, from the manifests, `[PENDING EXPERIMENT]`.
Figure 2, the isotonic AGC curves of a video over time, with r_x, f_x, e_x, `[PENDING EXPERIMENT]`.

---

## 7. Planned negative results and failure cases

Each is pre-registered with its handling; a failure is reported as a measurement, never silently dropped.

- **F1, hard kill for the primary split.** TIFFs are missing for `video_thermal_test`, or TIFF and JPEG stems do not pair. Checked on day 1. Fallback: hold out whole videos from the stills by file-name video id as the frozen test set, reported as a pre-registered deviation; N-AGC [2] used the video partition with 16-bit TIFFs, so coverage is probable but `[UNVERIFIED]`.
- **F2, the vendor map is not a global monotone curve.** Median e_x above one grey level on train. A3 is relabelled "global monotone component of the vendor map" before the freeze; A4 stays descriptive; the A0-versus-A2 primary is unaffected; the distributions of r_x, f_x, e_x are reported as a finding about the camera.
- **F3, too few video clusters.** Predicted h_test > delta from the calibration run. Handled before the freeze by the enlargement rule in §4.2; if enlargement is not possible within the stated limit, the primary is declared inconclusive by design before any budgeted run and the paper is re-scoped to H2 and H3.
- **F4, learned mapper unavailable or too large.** TCNet code is withheld and N-AGC code availability is `[UNVERIFIED]`. The mapper is re-implemented from the published description and labelled "re-implementation, unreproduced numbers". If it does not fit in 11 GB or does not train, C9 is dropped, the claim is narrowed to classical maps, and A2 stands in only for the shape of the DR interaction, never as the learned reference.
- **F5, H3 fails.** The pretrained network shows no float penalty. Reported as a non-replication of [4] on FLIR; the scratch cells still identify quantization versus adaptivity, and the conditional title loses the words "and pretraining mismatch".
- **F6, A1 is superior to A0.** Slow camera-state changes matter. The mechanism statement gains "after a per-video affine"; A1 becomes the recommended minimal map; the headline is unchanged if H1 still holds.
- **F7, augmentation sensitivity.** Only testable if the optional J1, J2 cells run; if they run and the C1 minus C5 difference changes by more than delta between jitter-off and jitter-on, the equality claim is restricted to "without photometric augmentation", and the direction is reported.
- **F8, H2 fails.** U_2 < delta: quantization is not the mechanism; the conditional title's attribution is withdrawn and the remaining candidate mechanism (pretraining) is reported from the H3 cells. If the positive control fails for this reason, no equivalence statement in the paper is interpreted.
- **F9, S1 gate fails.** FLIR's shipped AGC is temporally damped on this split; reported as the measurement; no flicker claim.
- **F10, H1 fails.** The falsifier in §4.6; the paper is rewritten around the confirmation of a genuine adaptivity requirement.
- **F11, licence.** Per §5.1: per-video summaries, or code-and-manifests-only release, with the paper stating which.
- **F12, budget.** Measured per-run time forces tier drops (§6.1); dropped cells are listed, and any hypothesis that loses a cell is reported as "not tested (budget)".
- **F13, channel control fails.** C15 differs from C1 by more than delta: factor I is reported as initialisation plus channel count, and H3(a) is interpreted against C15.

---

## 8. Reproducibility, limitations, deployment caveats

### 8.1 Reproducibility

Released: pre-registration (OSF), code at a tagged commit, `requirements-lock.txt`, manifests with SHA-256 and video ids, the dataset statistics (q_1,D, s_D, per-video statistics) and sampling seeds, every run configuration, log and val curve, the bootstrap index sets, the feasibility record, and the analysis notebook that regenerates every table from the detection files. Released subject to the licence (F11): per-frame covariates, isotonic curves and detection files. Not released: images; weights trained on the dataset if the licence forbids it `[UNVERIFIED]`.

### 8.2 Limitations

- One dataset, one camera model (Tau 2), one detector family (YOLOv8-s), one input size, one training recipe. The decomposition is general, the numbers are not, and "non-inferior to the per-frame maps tested" is not "no tone map beats a fixed affine".
- The 3-class protocol follows [1] and [3]; no peer-reviewed YOLOv8-s number exists for this exact protocol (`audit/PRIOR_ART_NOVELTY_MATRIX.md` §A), so our baselines are published with their own split and class map and are not comparable to 7- to 9-class YOLOv8n figures [8, 9]. Published "mAP" figures in [1] and [3] are of unrecorded definition and are not compared numerically to ours.
- No adverse-weather claim: FLIR ADAS v2 has no per-frame weather labels; the real-weather thermal sets are single-class and either 8-bit or too large for the available disk.
- The learned mapper is a re-implementation with unreproduced numbers unless code becomes available.
- Non-inferiority within 1.0 mAP50-95 is a statement at that margin; a reader who considers a smaller loss meaningful can read the CI directly.
- The two-way bootstrap's seed component rests on 3 to 5 seeds and is a coarse estimate; the per-seed values are published.
- The DR-tercile DiDs and S1 are low-powered by construction (few videos) and are labelled descriptive.
- BatchNorm's absorption of a dataset-wide affine is an invariance of the function class up to border and eps effects, not of the optimisation trajectory; the chosen scale of A0 keeps the library's hyperparameters in their tuned regime but we do not claim trajectory invariance.
- A3's count-resolution curve assumes the vendor curve is smooth between the plateaus the 8-bit image reveals; the vendor's true curve between plateaus is unobservable from the shipped files.
- The pretrained cells use one pre-registered input scale (the library's own grey-image scale); the Lynred claim concerns z-scored input, which is an optional cell. If C19 does not run, the non-replication statement in F5 is restricted to the [0, 1] scale.
- All 8-bit cells use Q_8 applied to the mapped float image; the vendor's own 8-bit pipeline (A4) may differ in dithering or rounding, which the C13 minus C8 comparison measures only together with JPEG and local processing.

### 8.3 Deployment caveats

- A0 saves inference time only when float counts are already available to the host, in which case the camera's 8-bit AGC output is usually also available for free; the latency table therefore measures the host-side cost of each map for integrators who stream counts, and nothing else. On a camera whose AGC runs in firmware and exposes only 8-bit video, nothing here argues for replacing it.
- A1 requires a running per-video statistic; A2 requires per-frame statistics before the first convolution can run; A3 and A4 require the camera's AGC; A5 requires an additional network. Latency is reported for each on one named device in fp32 and no real-time claim is made.
- Any detector trained on float counts is tied to the radiometric calibration regime of its camera; a change of lens, FFC behaviour or gain state changes q_1,D and s_D. A1 is the minimal mitigation for slow changes; residual pixel-wise non-uniformity is mitigated by no global map.

### 8.4 Future work (one paragraph, not an arm)

Because the TIFF carries the full count range, tone-map randomisation as an augmentation is free: a training set can be rendered through a random per-frame monotone map at each epoch, which tests invariance to the mapper rather than the mechanism studied here. It was proposed and refuted as a distillation study in the audit and is recorded only as a possible follow-up once the present decomposition exists. A second camera (ZUT-FIR-ADAS [12]) would test whether the precision term transfers.

---

## 9. Timeline and venue

Week 1: data verification, manifests, day-1 checks, K_test, A3 curves and label decision, pilot on the quarantined validation subset, batch size fixed, calibration run (C13 seed 0) started. Week 2: calibration run complete; per-run time, loader throughput, h_val and predicted h_test recorded; feasibility rule applied (test-set enlargement or re-scoping if required); systematic literature search (§2.3) logged; freeze and OSF pre-registration with the margin, the rules, the cell list and drop order. Weeks 3 to 5: tier 1, then tier 2, then tier 3 runs (about 12 days on two GPUs at the extrapolated per-run time, longer on one). Week 6: pixel-stage opening of `video_thermal_test` (tercile boundaries, m_t, gate) registered. Weeks 6 to 8: label-stage opening once, bootstrap, analysis, writing. Venue: IEEE Sensors Journal or Infrared Physics and Technology (primary); CVPR PBVS 2027 or IEEE RA-L as alternatives.

---

## References

Every entry below was taken from `paper/COUNCIL_RECORD.md` §3 or `audit/PRIOR_ART_NOVELTY_MATRIX.md`; identifiers are as recorded there. Titles marked "verified 2026-10-15" were copied from the primary page on that date by the lead author; the rest remain as the audit recorded them.

[1] D.-G. Lee, J. Kim, Y. Cho, A. Kim. Thermal Chameleon: Task-Adaptive Tone-mapping for Radiometric Thermal-Infrared Images. IEEE Robotics and Automation Letters, 2024. arXiv:2410.18340 (submitted 24 Oct 2024). https://arxiv.org/abs/2410.18340 . Opened during the audit; title and authors verified 2026-10-15 from the arXiv abstract page.

[2] Javidnia. Task-Driven Neural Adaptive Gain Control: 16-bit-to-8-bit Thermal Tone Mapping for [...] (N-AGC). ICCV 2025 Workshops (MIRA), CVF open access. https://openaccess.thecvf.com/content/ICCV2025W/MIRA/html/Javidnia_Task-Driven_Neural_Adaptive_Gain_Control_16-bit-to-8-bit_Thermal_Tone_Mapping_for_ICCVW_2025_paper.html . Opened during the audit. `[Title taken from the URL slug recorded in the audit, truncated where the slug is truncated; the CVF page returned HTTP 403 on 2026-10-15; complete the title, the detector-training setting and the v1/v2 assignment of the quoted numbers from the paper before submission.]`

[3] H. Gil, M.-H. Jeon, A. Kim. Fieldscale: Locality-Aware Field-based Adaptive Rescaling for Thermal Infrared Image. IEEE Robotics and Automation Letters, 2024. arXiv:2405.15395 (submitted 24 May 2024). https://arxiv.org/abs/2405.15395 . Opened during the audit; title and authors verified 2026-10-15 from the arXiv abstract page.

[4] C. Karam, J. Matias, X. Breniere, J. Chanussot. Optimizing the Image Correction Pipeline for Pedestrian Detection in the Thermal-Infrared Domain. arXiv:2407.04484 (submitted 5 Jul 2024). https://arxiv.org/abs/2407.04484 . Opened during the audit; title and authors verified 2026-10-15 from the arXiv abstract page (no venue listed there).

[5] Grimming, McIntosh, Mahalanobis, Driggers. OSA Continuum 4(2):529, 2021. DOI 10.1364/OSAC.404600. https://doi.org/10.1364/OSAC.404600 . `[Seen only via snippet during the audit; title not recorded; the publisher page did not load on 2026-10-15. Verify title and initials before submission.]`

[6] N. Bhowmik, J. W. Barker, Y. F. A. Gaus, T. P. Breckon. Lost in Compression: the Impact of Lossy Image Compression on Variable Size Object Detection within Infrared Imagery. arXiv:2205.08002 (submitted 16 May 2022). https://arxiv.org/abs/2205.08002 . Title, authors and abstract read 2026-10-15 from the arXiv abstract page; body not opened; cited for the abstract-level claim only.

[7] Teledyne FLIR. FLIR ADAS Dataset v2. https://www.flir.com/oem/adas/adas-dataset-form/ . Registration required; licence text not retrievable during the audit; third-party CC BY-NC-SA 4.0 claim `[UNVERIFIED]`.

[8] Ling et al. YOLO-APDM. Sensors 24(22):7197, Nov 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11598205/ . Opened. Cited only for the FLIR ADAS v2 YOLOv8 baseline landscape (7- to 9-class subsets, not comparable to the 3-class protocol) and for its recorded split sizes.

[9] Zhang et al. MS-YOLO. arXiv:2509.21696, Sep 2025. https://arxiv.org/abs/2509.21696 . Opened. Same role as [8].

[10] A. Cameron, H. H. Hoos, K. Leyton-Brown. Bias in Algorithm Portfolio Performance Evaluation. IJCAI 2016. https://www.ijcai.org/Abstract/16/107 . Cited for the rule that per-instance or per-run selection over randomised runs is optimistically biased and that independent runs must be averaged, which motivates reporting all seeds and no best-of selection.

[11] Ultralytics YOLOv8, software version 8.4.178 (as pinned in `audit/CODE_AUDIT_2026.md`); pycocotools; scikit-learn `IsotonicRegression`. `[Software; no DOI recorded in the audit; version strings to be taken from requirements-lock.txt.]`

[12] Tumas et al. ZUT-FIR-ADAS. IEEE Access 2020. IEEE DataPort DOI 10.21227/7f37-hx89. Cited in limitations and future work only (16-bit automotive LWIR, pedestrians only, about 95 GB).

[13] Krišto et al. UNIRI-TID. IEEE Access 2020. IEEE DataPort DOI 10.21227/yec9-yy29. Cited in limitations only (8-bit, static camera, persons only).

---

## Appendix A. Design choices made in this draft that the council record does not settle

Each item is a choice of this draft (or a change to the council's design after the internal review), with its reason. All are written into the OSF record.

1. **A0 formula.** S_0(x) = (x minus q_1,D) / s_D, the dataset 1-99 percent band on [0, 1], one formula for both precision cells; Q_8 clips and rounds only in the 8-bit cell. The council record names the median and the span without fixing the formula; revision 1 used three incompatible definitions.
2. **A1 is a per-video affine**, not a per-frame median. Reason: a per-frame median neither distinguishes camera-state drift from scene adaptivity nor addresses gain changes; a per-video affine isolates slow camera state (§3.3, §10 item 7).
3. **A2 is the primary comparator**, A3 and A5 secondary. Reason: A2-float is the only per-frame map whose float version is assumption-free (§3.3, §10 item 2). The council record made A3 primary.
4. **A3 is applied at count resolution** by strictly increasing interpolation of the isotonic knots, with a JPEG floor f_x and the label rule "median e_x on train at most one 8-bit grey level". The grey level is the quantisation unit; the rule is a choice.
5. **R (x / (2^14 minus 1)) is a budgeted arm** at 8-bit scratch (container term, positive control) and float pretrained (published RAW setting). The council listed it as optional at one seed.
6. **Factor I has three levels** (scratch-1ch, scratch-3ch, pretrained-3ch); the pretrained input affine is the identity on the mapped [0, 1] image, the library's own grey-image scale.
7. **Cells C16 (A4 pretrained), C17 (R float pretrained), C15 (A0 scratch-3ch) are budgeted**; C11 (A0-8bit pretrained) is optional. Reason: the published contrast must be run in its own configuration (§10 item 5).
8. **Epochs = 100** (library default) rather than the council's 50; budget extrapolated accordingly and a tiered drop order defined (§6.1).
9. **No photometric augmentation in any arm**; jitter-on replicates optional at 3 seeds. The council specified a logged brightness/contrast jitter; it was removed because it is itself a per-frame affine randomisation (§10 item 28).
10. **Fill value 0** for letterbox and mosaic in every arm, in the mapped domain (the bottom of the band for A0, A1, A2; vendor black for A3, A4; 0 counts for R).
11. **Margin delta = 1.0 mAP50-95 fixed**, not the across-seed SD. Reasons in §4.4.
12. **Two-way (video by seed) paired bootstrap** with per-cell independent seed draws; video-only CI and per-seed SD reported alongside.
13. **Seed counts fixed in advance** (5, 5, 3 elsewhere); the pilot decides nothing.
14. **Feasibility rule** (§4.2): K_test on day 1; calibration run C13 seed 0 before the freeze; predicted h_test; enlargement of the frozen test set from training videos up to one fifth of training videos; else re-scoping. The one-fifth limit is a choice.
15. **Decision rules are all CI-based**: non-inferiority L > minus delta; inferiority U < minus delta; equivalence [L, U] inside [minus delta, delta]; positive control L_2 > delta; attribution L_2 > max(|L_1|, |U_1|). The council's point-estimate clauses and the 5-mAP threshold are withdrawn.
16. **H4b is a difference-in-differences** across arms, not a within-arm sign.
17. **Two-stage test opening** (pixels, then labels), with tercile boundaries, K_test, the A3 curves and the gate outcome registered between the stages.
18. **S1 map movement is count-histogram-weighted.**
19. **Checkpoint rule**: last-epoch EMA weights, never best-on-val. The record does not state a rule.
20. **Evaluation settings** conf 0.001, NMS IoU 0.7, max 300 detections are from `audit/MATH_STAT_REVIEW.md` §4.3; `num_workers <= 4` and the pilot sizes are from `audit/REPRODUCTION_PROTOCOL.md` §4, §7. Draft-only: S1 track length at least 5, greedy IoU >= 0.5 linking, maximal-F1 operating point on val, the 10 percent gate fraction, B = 1000, the 10,000-pixel samples, the last-20-epoch slope statistic.
21. **Loader control is a deterministic tensor-equivalence test**, not a trained comparison.
22. **No float frame cache**; only A3 tables are cached.
23. **Bootstrap re-accumulates stored per-image match records**; COCOeval.evaluate() runs once per model.
24. **A "superior" outcome of H1 (L > 0) is reported but not claimed** as a pre-registered goal.
25. **H4a's family is uncorrected and descriptive**; the record lists H4 without a multiplicity rule.
26. **Ground-truth tracks for S1** from annotation track ids if present, else greedy IoU linking.
27. **References**: [1], [3], [4], [6] titles verified from the arXiv abstract pages on 2026-10-15; [2] and [5] could not be reached that day and remain as recorded; [2]'s detector-training setting and v1/v2 assignment are open.
28. **A5's architecture, parameter count, FLOPs, memory and objective are `[PENDING]`** and must be in the OSF record before any run; revision 1's statement that TCNet trains "on the detection loss alone" was not traceable to the audit and is withdrawn.
29. **The systematic literature search for tone-mapping and bit-depth work on this dataset is `[PENDING]`**; every "to our knowledge" is conditional on it.

---

## 10. Response to internal review

The internal hostile review of revision 1 raised 45 items. Forty are accepted and fixed above (the correspondence is given in Appendix A and in the section references below). Five are accepted in substance but implemented differently from the fix the reviewer proposed, or are contested in part; they are set out here so that they are not relitigated.

**Item 2 (A3-float is not a float arm).** Accepted: a step function fitted to 8-bit targets carries at most 8-bit information, so revision 1's C7 versus C8 contrast was empty by construction and H1 as written did not identify the adaptivity term. The fix differs from both options the reviewer offered: rather than re-implementing a plateau-equalisation algorithm whose parameters for this camera are unknown, A3 is kept as the vendor's own curve but applied at count resolution through strictly increasing interpolation of the isotonic knots (§3.3), which preserves count differences inside each plateau, and the primary comparator is moved to A2, whose float version requires no assumption at all. A3 becomes a secondary comparator under the same rule and carries the interpolation assumption explicitly. The reviewer's observation that only A2 and A5 were genuine float comparators was correct for revision 1 and drove the change.

**Item 7 (FFC-drift physics; demand for a two-parameter per-frame affine).** Accepted that a global per-frame median removes neither gain changes nor pixel-wise non-uniformity, and the "one parameter, not a histogram remap" language is gone. The proposed fix, a per-frame offset-plus-gain affine, is not adopted as the drift control because it is A2 without the clip, and a per-frame affine cannot distinguish camera-state drift from scene adaptivity, which is the distinction the control exists to make. A1 is instead a per-video affine (§3.3): it removes slow camera-state changes between recordings and nothing per frame. The gain-state concern is additionally checked on day 1 through within-video bimodality of the frame median. A2-float, now the primary comparator, is the two-parameter per-frame affine the reviewer asked for, in its own cell.

**Item 16 (the 5-mAP threshold is contradicted by prior art).** Accepted that the threshold had no source, and it is withdrawn in favour of a positive control defined relative to the margin and an attribution rule that compares CIs (§4.5). The reviewer's claim that Fieldscale [3] and TCNet [1] contradict a 5-point quantization effect is not accepted: both rows compare an 8-bit stretch to the 8-bit vendor AGC, so they bound the adaptivity term at 8 bits (1.4 to 4.4 points, single run), and say nothing about the float-versus-8-bit precision term, which no published row measures. That is precisely why those rows are now cited as the strongest existing evidence on H4 (§1.2, §2.1) rather than on H2. The threshold was dropped because it was unsourced, not because the prior art bounds it.

**Item 35 (Bhowmik [6] cited for a specific finding without being opened).** The "especially small objects" clause traces to the council record's row for [6], which was based on the abstract. On 2026-10-15 the arXiv abstract page was read for this revision and it states that tiny and small objects are more sensitive to compression than medium and large ones, so the clause is now supported at abstract level and labelled so (§2.1, [6]). The body remains unopened, and [6] is used for one purpose only: that a JPEG-versus-no-JPEG contrast is confounded with the map, which is the reason A4 is not the primary comparator. That reason does not depend on the small-object clause.

**Item 44 (the latency column measures nothing a practitioner would act on).** Partly accepted: the column is relabelled descriptive and the "zero inference cost" wording is confined to the host-side case (§3.7, §8.3). It is retained because the dataset's own TIFFs demonstrate that integrators with host access to the count stream exist, and for them the per-frame cost of A1, A2 and A5 relative to a folded affine is a real host-side cost. No argument in the paper rests on it.

**Item 43 (no "so what" beyond error bars).** Accepted that revision 1's consequence paragraph was symmetric and therefore uninformative. The replacement (§1.4) states the asymmetric consequence: the two large published RAW rows are the baseline against which learned per-frame mappers justify their existence; if H1 and H2 hold, that baseline is mis-specified and the mapper papers' margins over AGC are not evidence of an adaptivity requirement, whereas if H1 fails the requirement is confirmed with error bars for the first time. The reviewer's description of the expected result as a "textbook default" is accepted as a prior; the paper's claim is that the default has never been tested in a cell where it is the only thing that changes, and that the literature has been drawing the opposite conclusion from confounded rows.

**Items 6, 12 and 13 (epochs, K, pilot).** Accepted in full. One qualification on item 6: a compressed low-mAP regime makes equivalence easier to pass at a fixed margin, which is the reason the epoch count was raised and the val slope is reported, but it does not necessarily inflate the seed SD; since the margin is no longer the seed SD, the second concern is moot. On item 12, the audit records two incompatible frame counts for the video partition (3,742 and 1,144) and no video count, so K is unknown in both directions; the feasibility rule (§4.2) makes it a day-1 kill condition without choosing a minimum K now.

**Item 38 (BatchNorm eps).** Accepted and the caveat added (§3.4). The pre-BN stem variance will be logged in the pilot for each arm; if it is not large relative to eps at the chosen input scale, the argument in §3.4 is weakened and the paper will say so. The scale argument for A0 does not depend on exact invariance.

**Item 20 (test set "opened once").** Accepted; the two-stage opening is now stated as such (§3.5, §4.8), with the pixel-stage outputs registered before the label stage.

All remaining items (1, 3, 4, 5, 8, 9, 10, 11, 14, 15, 17, 18, 19, 21 to 34, 36, 37, 39 to 42, 45) are accepted and implemented as described in the sections cited in Appendix A; items 36 and 45 are only partly fixable now (two primary pages unreachable on 2026-10-15; authorship and licence remain open items that no draft can settle).
