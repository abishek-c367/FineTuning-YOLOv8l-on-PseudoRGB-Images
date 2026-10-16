# PAPER OUTLINE (conditional on PIVOT option 1; all results `[PENDING EXPERIMENT]`)

**Working title:** Does an unpaired pseudo-RGB teacher help thermal object detection? A controlled, released ablation of cross-modal feature distillation on FLIR ADAS v2.

**One-sentence claim (to be filled only from results):** On a video-grouped 3-class FLIR ADAS v2 protocol with matched compute and ≥3 seeds, feature distillation from a COCO-pretrained YOLOv8-L fed CycleGAN pseudo-RGB `[does / does not]` improve a single-channel YOLOv8-s student beyond `[no-teacher / thermal-teacher / grayscale-input-teacher]` controls, by `[PENDING]` mAP50-95 (95% video-cluster bootstrap CI `[PENDING]`).

1. **Abstract** — `[PENDING EXPERIMENT]` for every number; structure: problem (thermal data scarcity; translate-then-distill is routine but never controlled), protocol, controls, headline finding, release.
2. **Introduction** — Open problem: since Borrow-from-Anywhere (2019) and up to CGDet (2025), pseudo-visible translation is used to inject RGB knowledge into thermal detectors, but no study isolates (i) the translator, (ii) the teacher modality, (iii) the mimic loss as regulariser, under matched compute. Why current approaches fail to answer it: different detectors, class subsets, splits, no seeds, no released 3-class YOLOv8 baseline.
3. **Related work and claim boundary** — A1–A15 from the matrix; explicitly: we add no architecture; P2 and ViT necks are prior work; we do not claim a new KD loss.
4. **Method (as studied)** — Student (1-ch YOLOv8-s, stock + repo variant), teacher inputs {pseudo-RGB via CycleGAN, grayscale-replicated thermal, thermal-trained teacher, none, shuffled pairing}, mimic loss (MSE on L2-normalised P3/P4 with 1×1 adapters, α), training vs inference information (teacher and translator train-only), complexity (train-time cost of translator and teacher from T07).
5. **Statistical design** — video-grouped splits, frozen test, cluster bootstrap, seeds, decision rule, pre-registered primary contrast (MATH_STAT_REVIEW §4). No theorems.
6. **Datasets / protocol / implementation** — FLIR ADAS v2 (licence), manifests with hashes, class map, letterbox 512, pinned versions, hardware; external M3FD check.
7. **Results** — `[PENDING]`: main table (B0, B0', B1, B3a/b/c), per-class and AP_small, seeds, CIs; mechanistic ablation (P2); negative results and failure cases (e.g., translator artefacts, pairing control).
8. **Reproducibility, limitations, deployment caveats** — single dataset family, 8-bit AGC only, no real-weather claims, latency measured on RTX 2080 Ti fp32 only.
9. **References** — from the verified matrix (DOIs/arXiv IDs as listed).

Optional §: oracle-headroom analysis of cheap enhancement (B4/B5) if PIVOT option 2 is folded in; otherwise a separate short paper.
