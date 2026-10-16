# PUBLICATION OPTIONS (16 Oct 2026): candid ranking of Tracks A to E and of the RAW-penalty paper

Basis: `PRIOR_ART_COUNTERCHECK.md` (this date), the 15 Oct novelty matrix, `CURRENT_STATE_AND_BLOCKERS.md`. Everything is a **recommendation**; no detector experiment exists. The ranking weighs (i) whether the question is open after the counter-check, (ii) expected outcome given published evidence, (iii) cost on this machine, (iv) dependence on artefacts not in hand, (v) reuse of the collaborator's code.

## 1. Ranking

| Rank | Study | Open after counter-check? | Expected outcome | Cost (GPU 1) | Blockers beyond data | Venue if it works | Reasons **not** to proceed |
|---|---|---|---|---|---|---|---|
| 1 | **RAW-penalty decomposition** (`audit/EXPERIMENT_MATRIX.md`, `paper/DRAFT.md`) | yes: no multi-seed, CI-bearing, precision-controlled table exists; TCNet/Fieldscale rows confirmed single-run with dtype unstated | either way informative; pre-registered falsifier | tier 1 ≈ 265 GPU-h (≈ 11 days on GPU 1 alone), all tiers ≈ 590 | none (stock Ultralytics, FLIR TIFF/JPEG only); TIFF presence confirmed on the vendor page | RA-L / T-ITS / PBVS | long compute; a null on H2 (positive control) would leave only a descriptive table; does not use the collaborator's code |
| 2 | **Track D: GT-informed enhancement-oracle upper bound** on a frozen thermal YOLOv8-s, plus a cross-fitted router from image features only | yes for thermal: no per-image oracle on thermal data published; realised router gains elsewhere ≈ 0.1 mAP | most likely: small cross-fit headroom → kills the weather-gate idea with numbers | ≈ 12 GPU-h for the detector (rank 1's calibration run C13 seed 0, shared), then inference over 4 to 6 enhancers on development and sealed videos, router fitting on development videos, CPU bootstrap: about 1.5 GPU-days end to end, **not** inference-only | a trained detector | short paper / workshop; or a section of rank 1 | a finding of large headroom would require a router study (GDIP/IA-YOLO/AMIEOD baselines) that is a separate, crowded paper |
| 3 | **Track A: four-arm teacher-modality controls** (`PILOT_PROTOCOL.md`) | yes, but Yoon et al. (IEEE Access, unread) may already hold part of it | predicted null or negative for the pseudo-RGB teacher (FreqKD, HalluciDet, ModTr) | ≈ 120 GPU-h + teacher fine-tune + cache (≈ 6 days); + 1 to 2 GPU-days if the CycleGAN must be retrained | `cyclegan_epoch_45.pth`; Yoon et al. full text; the collaborator's custom student (patched) | PBVS workshop / short paper | a negative result with a known-bad translator (PearlGAN, HalluciDet already show CycleGAN is poor) is "expected"; publishable only with all four arms, seeds and code; relies on the audited code path with the most defects |
| 4 | **Track B: translation artefact attribution** | partly occupied (PearlGAN, HalluciDet, ModTr quantify CycleGAN's object loss on FLIR); open only for size/boundary attribution into a teacher | descriptive | small once Track A exists | Track A | none alone; a section of Track A | not a paper on its own; "do not claim causal evidence from pictures" (roadmap) is correct |
| 5 | **Track E: temporal flicker vs tone-map movement** | open for thermal; metric exists (Stability Index) | gated; may fail the S1 gate (damped AGC) | inference only, after rank 1 | count-level video frames (present in v2) | section of rank 1 (S1) | gate may fail; GT tracks may not exist; never headline |
| 6 | **Track C: P2 head / transformer neck** | largely occupied (YOLO-APDM, LFIR-YOLO, CRT-YOLO, MDCFVit-YOLO) | incremental | ≈ 3 runs × seeds | none | none | an ablation, not a contribution; run only as a control if Track A's student is kept |
| — | Weather gate / enhancer routing as a method (the original GATE-IR inference pathway) | no (Jayaraman 2026, GDIP, IA-YOLO, Ye et al. 2024, AMIEOD 2026) | — | — | no weather labels in FLIR v2; gate untrained | — | **do not proceed** unless Track D shows large cross-fit headroom |
| — | Forensics proposal (DDIM reconstruction + contrastive) | no (DRCT, ICML 2024) | — | — | — | — | **STOP** (unchanged from 15 Oct) |

## 2. Recommendation

One plan, in this order, all on the same data download, manifests, video-grouped splits and pycocotools evaluator:

1. **Gate 0** (collaborator's artefacts, FLIR archive under its terms, authorship with the zer0 author in writing). Nothing below starts without the archive.
2. **Day-1 checks and the shared pilot** (matrix §5, `PILOT_PROTOCOL.md` §3): manifests, split integrity, TIFF/JPEG pairing, container bit depth. Under 2 h, no approval needed.
3. **Track D first after the calibration run** (rank 2): it needs one trained detector (rank 1's calibration run C13 seed 0, ≈ 12 GPU-h, which precedes the freeze anyway), then inference and CPU work. It answers the collaborator's original question (is a weather gate worth anything on this data?) with (i) fixed-enhancer baselines, (ii) the GT-informed per-image oracle as an infeasible upper bound computed as pooled COCO mAP on oracle-selected predictions with cross-fit correction, and (iii) a router fitted on development videos from image features only and scored on sealed videos, with latency and video-cluster CIs. A large oracle number is not evidence for a router; only (iii) is. Protocol in `audit/MATH_STAT_REVIEW.md` §1 (amended).
4. **RAW-penalty tier 1** (rank 1) as the main paper, frozen and pre-registered on OSF after the calibration run; approval requested for the ≈ 265 GPU-h block before launch.
5. **Track A** (rank 3) only if `cyclegan_epoch_45.pth` arrives and Yoon et al. is read and found to lack the controls; then as a pre-registered four-arm, three-seed study on the collaborator's patched code, written up as a workshop paper with him as first author. If the checkpoint never arrives, Track A is dropped; retraining a CycleGAN to test a translator the literature already rates as poor is not a good use of two GPU-days.
6. Tracks B and E fold into A and 1 respectively; Track C is not run on its own.

## 3. Reasons not to proceed with anything

- If the FLIR terms forbid derivative release (manifests, tone curves, detections), rank 1 loses its reproducibility contribution and rank 3 loses its code release; check the terms before any download.
- If the collaborator cannot supply any artefact (item list in `CURRENT_STATE_AND_BLOCKERS.md` §5), his reported numbers must be treated as unreproduced in every document and talk, and Track A becomes a study of our own runs of his code, which he should agree to in writing.
- If GPU 1 cannot be dedicated for about two weeks, rank 1's tier 1 cannot complete; the drop rule never cuts tier 1, so the paper would have to wait rather than shrink.

## 4. Changes after the 16 Oct independent method review

Accepted and implemented on branch `review-fixes/2026-10-16` (details in `REVIEW_RESPONSE_2026-10-16.md`): the P0 corrections to the mechanism attribution (clipping / rounding / range allocation as separate interventions; A0 clipped at both precisions; new cells C21, C22), the pretraining DiD, the Track D wording and cost above, the Yoon et al. counter-check, paired-seed bootstrap, causal A1, manifest verifier with near-duplicate and header checks. The budget rises from ≈ 590 to ≈ 660 GPU-h (tier 1 unchanged). The review's executive decision, keep the pivot but do not approve tier 1 or freeze the draft yet, is adopted: nothing is frozen, nothing is launched.

## 5. What this changes relative to 15 Oct

Nothing in the council's choice. The roadmap added two things worth keeping: it caught the matrix/council/draft inconsistency (now reconciled, matrix header), and it argued correctly that the oracle-headroom check should precede any gate work (now rank 2 and step 3). Its ranking of Track A as the first priority is not adopted for the reasons in the table.
