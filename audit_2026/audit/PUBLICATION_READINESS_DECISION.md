# PUBLICATION READINESS DECISION — GATE-IR (15 Oct 2026)

## Gate status

| Gate | Status | Reason |
|---|---|---|
| 0 — Inputs | **NOT PASSED** | No data, weights, logs or pinned environment available; authorship/roles with the zer0-data author unresolved; FLIR licence text not verified. |
| 1 — Reproducibility | **NOT PASSED (blocked)** | Nothing to reproduce against; smoke tests show the code path is sound on Ultralytics 8.4.178 (T01/T02) but the training run's version is unknown and the only "result" artefact (the notebook) is invalid (C1). |
| 2 — Scientific room | **FAILS for GATE-IR as proposed; PASSES narrowly for a re-scoped ablation study** | Every component and the routing idea have verified 2019–2026 precedents (matrix §E). Two questions remain open: the value of an *unpaired* pseudo-RGB teacher vs same-modality / no teacher (A-§E.4a), and whether cheap enhancement has any oracle headroom on thermal detection (B-matrix bullets). |
| 3 — Empirical evidence | NOT STARTED | — |
| 4 — Writing | NOT STARTED | `paper/OUTLINE.md` drafted with placeholders only. |

## Decision: **PIVOT** (not GO, not STOP)

**Do not pursue** "GATE-IR" as a method paper (weather gate + enhancers + ViT/P2 YOLOv8 + CycleGAN KD). A competent 2026 reviewer will cite Jayaraman 2026 / YOLO-DER / CADENet for the routing, GDIP/IA-YOLO for gated classical filters, CGDet for the distillation, YOLO-APDM for P2-on-FLIR-v2, and Ye et al. 2024 + learning-to-defer for "risk-aware" selection. The rain-removal stage additionally lacks physical grounding for LWIR.

**Pursue instead, in this order (ranked by publishability × feasibility):**

1. **Controlled ablation / reproducibility study of cross-modal distillation for thermal YOLO on a released 3-class FLIR ADAS v2 protocol** (arms B0, B0', B1, B3a/b/c, P1/P2 of the matrix; 3–5 seeds; video-cluster bootstrap; code + manifests released). Claim type: "an unpaired CycleGAN pseudo-RGB teacher [does / does not] add anything beyond a same-modality teacher and a no-teacher baseline at matched compute; here is the first literature-comparable 3-class YOLOv8 FLIR-v2 baseline with code." A negative result is acceptable here. Feasible on this machine in ~3 weeks of GPU time once data and weights exist. Fits an applied journal.
2. **Oracle-headroom study of input enhancement for thermal detection** (B4/B5/B6), extended with the real-weather sets (UNIRI-TID, ZUT-FIR-ADAS) for a person-only external check. If headroom is near zero, the paper's message is "stop building weather routers for LWIR detectors; train with augmentation instead", with physics discussion (matrix §D). Cheap (inference-only once B0 exists); modest venue; honest.
3. **STOP** the forensic proposal (`SECOND_PROJECT_FEASIBILITY.md`) unless the single DRCT ablation is wanted on cluster compute.

Only after (1) or (2) produces a positive, multi-seed, externally checked effect would a method paper (e.g. a cost-aware router that beats run-all + confidence) be worth considering, and it would still need to be positioned against B6 and the L2D literature.

## Venues (checked 15 Oct 2026; re-verify before submission)

| Venue | For | Against | Notes verified |
|---|---|---|---|
| **IEEE Sensors Journal** | Applied, sensor-centric; thermal cameras are in scope; accepts negative/ablation studies if well executed | 8-page double-column limit; $175/page overlength; OA optional (~US$2,800); not a venue that rewards methodological novelty | Length/fee rules verified on the journal's for-authors page |
| **Infrared Physics & Technology** (Elsevier) | Thermal imaging is core; published the 2026 enhancement/detection review (matrix header) | Scope note: papers that use thermography "in an essentially routine way to advance some other field" are not suited; a pure CV ablation may be judged routine unless the LWIR-physics angle (§D) is substantive | Scope verified on Elsevier listing; page limits/APC not retrieved |
| **Pattern Recognition** (Elsevier) | Strong CV venue | Scope guidance warns against routine applications of known methods; an ablation study on YOLOv8 is unlikely to clear it | Scope note from a third-party guide; official guide not retrieved |
| **Information Fusion** | Cross-modal (RGB→thermal) transfer is in scope | Expects a fusion contribution; a KD ablation is marginal | Scope verified |
| **IEEE TIP / CVPR / ICCV** | — | Not realistic for a reproducibility/ablation paper without a new mechanism | — |
| Alternative: *Sensors* (MDPI), *Electronics* (MDPI), *IEEE Access* | Fast, accept ablation/negative results, where YOLO-APDM / LKD-YOLOv8 / YOLO-DER appeared | APCs; lower prestige | Precedents in matrix A10, A12, B4 |
| Alternative: workshop (CVPR PBVS, WACV workshops) | Natural home for thermal-detection ablations (Borrow from Anywhere was PBVS) | Short; deadlines seasonal | — |

Recommendation: target **IEEE Sensors Journal** or **CVPR PBVS workshop** for (1); *Infrared Physics & Technology* only if (2) carries a real LWIR-physics analysis. Do not promise acceptance anywhere.

## Conditions to re-enter GO

- Collaborator delivers logs, checkpoints, CycleGAN weights, data access, authorship agreement (INPUTS §5).
- Video-grouped split verified; matrix frozen; pilot passes (REPRODUCTION §4).
- B1 reproduced within tolerance (REPRODUCTION §6).
