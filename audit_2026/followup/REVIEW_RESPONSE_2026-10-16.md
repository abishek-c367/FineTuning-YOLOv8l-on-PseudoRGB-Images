# RESPONSE TO THE INDEPENDENT METHOD REVIEW OF 16 OCT 2026

Scope of the review: `PUBLICATION_OPTIONS.md`, `PILOT_PROTOCOL.md`, `PRIOR_ART_COUNTERCHECK.md`, `CURRENT_STATE_AND_BLOCKERS.md`, `audit/EXPERIMENT_MATRIX.md`, `paper/DRAFT.md`. Each point was cross-checked against the current files, the transforms as written, and the sources named; nothing was taken on trust. Everything below is static: no GPU training, no data, no other process touched. Branch: `review-fixes/2026-10-16` (from `followup/2026-10`). Anything untested is marked UNVERIFIED.

| Review point | Cross-check result | Decision | Where |
|---|---|---|---|
| **Executive decision**: keep the pivot; do not approve tier 1 or freeze the draft yet | Agreed; nothing had been launched or frozen | Adopted | `PUBLICATION_OPTIONS.md` §4 |
| **P0.1** C1 − C2 called "rounding" but C1 was unclipped float and C2 clip + round | **Confirmed** from draft §3.3 ("No clipping at float precision; in the 8-bit cell, Q_8 clips … and rounds") | A0 is now clipped to [0,1] at both precisions, so C1 − C2 is the rounding operation alone; the clipping intervention gets its own cell **C21 (A0u, unclipped float, 3 seeds, tier 2)**; `requant8` refuses unclipped input so clipping can never hide inside the precision factor | draft §3.2, §3.3, §4.5, §6.1; matrix §0 to §3; `audit_tools/preproc_controls.py`; T09 |
| **P0.1** C2 − C14 called "container/quantization" though both are 8-bit | **Confirmed** | Relabelled **range allocation at fixed 8-bit output**, stated to bundle saturation differences; the word "quantization" is reserved for Δ2a and only if its own CI excludes 0; title changes if Δ2a is small and Δ2b large | draft §3.2, §4.5; matrix §3 |
| **P0.1** float32 does not create precision | Agreed (the draft's §3.3 argument already said so, the abstract did not) | Abstract reworded: the intervention is the absence of requantization after a specified map; T09 asserts U has no more distinct levels than the counts | draft abstract; matrix §1 |
| **P0.1** supplied script `gateir_preprocessing_controls.py` | Not received with the review (only named) | Written independently as `audit_tools/preproc_controls.py` with the U/V/W/Z chain, A2, causal and offline A1, covariates, conv1 fold; unit-tested (T09, 16 checks, all pass) | `audit_2026/audit_tools/`, `tests/test_09_*` |
| **P0.2** C10 vs C12 compares two maps inside a pretrained architecture, not pretraining vs scratch; H3(b) used Δ1 − Δ3 (1-ch scratch vs 3-ch pretrained) | **Confirmed** | H3(a) relabelled "map sensitivity under pretrained initialisation"; H3(b) is now a DiD within the 3-channel architecture, (C15 − C22) − (C10 − C12), with the new cell **C22 (A2 float, scratch 3-ch, 3 seeds, tier 2)**; the Lynred effect is claimed only if (a) and (b) both hold; optional C23 (A0u pretrained) for the pure input-scale bracket; no R scratch-3ch cell is added (not needed for the DiD) | draft §3.2, §4.5, §6.1; matrix §2, §3 |
| **P0.3** Track D oracle must be named a GT-informed infeasible upper bound; mAP non-additive; router must be cross-fitted from image features | Agreed; `MATH_STAT_REVIEW.md` §1 already had the cross-fit correction but not the pooled-mAP rule | Oracle protocol amended: per-image utility pre-specified, selected predictions pooled, COCO mAP on the pooled set, cross-fit; router on development videos only, scored on sealed videos, reported beside baselines, oracle bound, latency, CIs | `MATH_STAT_REVIEW.md` §1; `PUBLICATION_OPTIONS.md` |
| **P0.3** Track D is not "inference-only, < 1 GPU-day" | **Confirmed**: it needs the calibration detector (≈ 12 GPU-h) | Cost restated as ≈ 1.5 GPU-days end to end after the calibration run; order of execution corrected | `PUBLICATION_OPTIONS.md` §1, §2; ledger L31 |
| **P1** Yoon, Park & Paik, IEEE Access 14 (2026) | Existence, venue, DOI and abstract **verified** (Consensus record): LoRA-tuned T2V generator, MMD + response distillation, thermal-only student, FLIR Aligned + LLVIP, 40.4 % mAP. The review's ablation rows (38.0 / 36.8 / 39.4) and architecture details are **UNVERIFIED** here: IEEE Xplore and ResearchGate full text are blocked from this machine; the library (libkey) link is the route | Counter-check row rewritten; verdict A amended: the method is published, only the four-arm control question is open; head-to-head comparison required before any Track A write-up | `PRIOR_ART_COUNTERCHECK.md`; ledger L32 |
| **P1** T1 (thermal fine-tuned teacher) has extra in-domain supervision | Agreed | T1 kept as a practical reference; T2 vs T3 named as the clean translation-front-end contrast; teacher-only mAP and class map recorded | `PILOT_PROTOCOL.md` §2 |
| **P1** bootstrap resamples seeds independently per cell, discarding seed covariance | **Confirmed** (draft §4.2: "no seed pairing across cells is assumed") | Seeds paired by index; joint seed draw over the common index set per contrast (5 for C1 vs C5; indices 0 to 2 when a 3-seed cell is involved); independent draw kept as a sensitivity check | draft §4.2; matrix §0; ledger L33 |
| **P1** margin 1.0 is a design choice; release precision/CI-width calculations; inconclusive ≠ equivalence | Agreed; the draft's §4.4 reasons stand but did not say so | Stated as a design choice; predicted h_test written into the OSF record beside δ (calibration run, draft §4.2); TOST rules already require the whole CI inside the margin | draft §4.4; matrix §0 |
| **P1** verify partition annotations, enough video groups, overlap incl. near-duplicates and the CycleGAN training set | Agreed; the matrix had video overlap only | `audit_tools/verify_manifest.py`: SHA-256, TIFF mode and max count, JPEG pairing by stem and dimensions, video overlap, exact and dHash near-duplicates across splits, box geometry, category ids, duplicate image ids; exit 1 on hard failure; tested on a synthetic dataset with eight planted violations (T10, all flagged). K_test rule already in draft §4.2. CycleGAN training-list check added to day 1 | matrix §5; `tests/test_10_*` |
| **P1** check local TIFF headers and pairing, not file names | Agreed | Same verifier (mode, max count, dimensions); isotonic-fit residual option for map fidelity | `verify_manifest.py --fit-residual` |
| **P1** per-video mapper must be causal | **Confirmed** (draft said "non-causal … upper bound") | A1 defined causal (expanding window, `arm_A1_causal`); offline form is a pilot diagnostic only; T09 verifies no future-frame dependence | draft §3.3; matrix §1 |
| **P1** execution order and resource guardrails | Consistent with `PILOT_PROTOCOL.md` §5 and matrix §4; disk floor 20 GB already an abort criterion; 68 GB free | No change beyond the Track D re-ordering | — |
| Budget | Two cells added: 55 runs, ≈ 660 GPU-h (tier 1 unchanged at 22 runs, ≈ 265 GPU-h) | Recorded | draft §6.1; matrix §2 |

## Points not adopted

- Adding an R scratch-3ch cell: not needed for the H3(b) DiD as now defined; listed nowhere to keep the budget flat. Reconsider only if H3(c) is promoted.
- Replacing the review's four-cell chain U/V/W/Z with separate retraining for every step at tier 1: U (C21) is tier 2 because A2, the primary comparator, is itself clipped, so the primary contrast and the positive control are unaffected by the clipping term.

## Still blocked (unchanged)

FLIR ADAS v2 archive and terms; the collaborator's checkpoints, CycleGAN weights, teacher weights, commands, class map, raw metrics, split lists; authorship record with the zer0 scaffold author; GPU approval for any run over 2 hours; Yoon et al. full text (library access). No number in any document is a result.

## Static test record (16 Oct 2026)

```
tests/test_05b_cache_paths_patched.py   all_pass: true   results/T05b_cache_paths_patched.json
tests/test_09_preproc_factor_isolation.py  all_pass: true   results/T09_preproc_factor_isolation.json
tests/test_10_manifest_verifier.py     all_pass: true   results/T10_manifest_verifier.json
```
Environment: Python 3.10.12, numpy/PIL from `~/gateir_audit/venv`, CPU only. Commit ids are in the branch log.
