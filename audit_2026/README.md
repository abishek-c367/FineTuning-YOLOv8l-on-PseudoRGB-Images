# GATE-IR audit workspace (15 Oct 2026)

Independent audit of this repository (commit 373dbdd) and of https://github.com/zer0-data/GATE-IR (commit 7948bd6), following the project research brief, plus a council-reviewed proposal for the follow-up paper.

| Path | Content |
|---|---|
| `audit/INPUTS_AND_PROVENANCE.md` | documents, repos, commits, authorship questions, what is missing |
| `audit/PROJECT_COMPONENT_MAP.md` | done / partial / scaffold / proposed; train vs inference pathways |
| `audit/CODE_AUDIT_2026.md` | 9 defects with evidence, tests, impact, fixes; false alarms; latency |
| `audit/PRIOR_ART_NOVELTY_MATRIX.md` | 40+ verified works (2019–Oct 2026), datasets with weather labels, LWIR physics, novelty verdict |
| `audit/RECOMMENDATIONS_COUNTERCHECK.md` | every brief recommendation rated SUPPORTED / PARTIAL / REFUTED / UNTESTED |
| `audit/MATH_STAT_REVIEW.md` | DPI, 2ε bound, estimands, cluster bootstrap, pre-registration items |
| `audit/EXPERIMENT_MATRIX.md` | pre-registration draft (arms, hypotheses, falsifiers, budget) |
| `audit/REPRODUCTION_PROTOCOL.md` | environment, inputs to obtain, manifests, pilot, full runs needing approval |
| `audit/CLAIM_EVIDENCE_LEDGER.md` | 25 claims → evidence class → status |
| `audit/PUBLICATION_READINESS_DECISION.md` | gates, PIVOT decision, ranked alternatives, venues |
| `audit/SECOND_PROJECT_FEASIBILITY.md` | forensics proposal: STOP / one DRCT ablation |
| `paper/OUTLINE.md`, `paper/COUNCIL_RECORD.md`, `paper/DRAFT.md` | paper outline, council record and draft (all results pending) |
| `tests/test_0{1..8}_*.py`, `results/T0*.json` | smoke/regression tests and raw outputs (with env + commit hashes) |
| `patches/000{1..7}-*.patch` | the seven fixes, also committed on branch `audit/2026-10` |
| `docs/` | project slides, summary PDF and the separate forensics proposal |

Environment used: Python 3.10, torch 2.6.0+cu124, ultralytics 8.4.178, RTX 2080 Ti.
Run a test from `tests/`: `CUDA_VISIBLE_DEVICES=1 python -s test_01_loss_path.py`
