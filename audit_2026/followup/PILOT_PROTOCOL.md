# PILOT PROTOCOL: smallest fair teacher comparison (Track A, Gate 2)

**Status: proposal, not run. Conditional on Gate 0 (FLIR ADAS v2 on disk; `cyclegan_epoch_45.pth` received).** This protocol is written because the follow-up roadmap of 16 Oct 2026 asked for it. The recommendation in `PUBLICATION_OPTIONS.md` is that Track A is run only as a bounded diagnostic and is not the paper; the pilot below is sized accordingly. Nothing in the pilot sets a threshold, a metric or a seed count for any later pre-registration (`feedback: pilot never sets thresholds`).

## 1. Question the pilot answers

Only a functional one: do the four teacher configurations train end to end on the same student, with finite losses, a working evaluator and reloadable checkpoints, within the memory of one RTX 2080 Ti? It does **not** answer whether the pseudo-RGB teacher helps.

## 2. Arms (same student in every arm)

Student: the repository's `yolov8_thermal.py` student with the trained configuration (`include_p2=True`, `use_transformer_neck=False`, 1-channel input), head bias initialisation from patch 0007, trained with `train_distillation.py` on branch `followup/2026-10`.

| Arm | Teacher | Teacher input | Distillation | Needs |
|---|---|---|---|---|
| T0 | none | — | `--alpha 0` (detection loss only) | nothing extra |
| T1 | thermal-trained YOLOv8-L (fine-tuned from COCO on the same 3-class FLIR thermal train split, 1-channel replicated to 3) | native thermal 8-bit AGC JPEG, grey replicated | P3/P4 feature mimic as in the repository | one teacher fine-tune (≈ 3 h on the pilot subset is not meaningful; see §6) |
| T2 | COCO YOLOv8-L, frozen | thermal grey-replicated to 3 channels, no translation | same | stock `yolov8l.pt` (present) |
| T3 | COCO YOLOv8-L, frozen | CycleGAN pseudo-RGB from the cache (patch 0002) | same | `cyclegan_epoch_45.pth` |

Matched across arms: split, labels (one class convention, patch 0006), architecture, image size 512, batch, epochs, optimiser, schedule, augmentation (geometric only; photometric off because the teacher input must stay identical to the student's), feature levels (P3, P4), α = 0.2 where a teacher exists, seed. Documented mismatches: T1 costs one extra training run; T3 costs the CycleGAN cache (≈ 10 GB, ≈ 1 h on the full train set; minutes on the pilot subset).

## 3. Leakage prevention

- Pilot subset: ≈ 200 train frames from ≤ 10 videos of `images_thermal_train` and 50 held-out frames from 3 other videos of the same split (video id parsed from `video-<id>-frame-<n>`); `images_thermal_val` and `video_thermal_test` untouched.
- Manifest with SHA-256 per file; assertion that no video id is in both pilot subsets.
- The CycleGAN checkpoint was trained by the collaborator on FLIR train; if it was trained on frames that end up in any later held-out set, that is recorded as a known leak of the translator (not of labels) and the later protocol holds out whole videos the checkpoint never saw, if its training list is available (`request list` item 2).
- Teacher T1 is fine-tuned on the pilot train subset only during the pilot.

## 4. Resources and commands (GPU 1, `num_workers 4`, ≤ 2 h total, no approval needed once data exists)

```
cd ~/gateir_audit/repos/student_repo   # branch followup/2026-10
CUDA_VISIBLE_DEVICES=1 python Cycle_gan/cache_pseudo_rgb.py --subset pilot_train.txt         # T3 only, minutes
CUDA_VISIBLE_DEVICES=1 python train_distillation.py --subset pilot_train.txt --val pilot_val.txt \
    --epochs 2 --batch 8 --imgsz 512 --seed 0 --alpha {0|0.2} --teacher {none|thermal|coco_grey|coco_pseudorgb}
python audit_tools/eval_coco.py --ckpt <run>/last.pth --split pilot_val   # pycocotools, conf 0.001, NMS 0.7
```
The `--subset`, `--teacher` and `--seed` flags do not yet exist in `train_distillation.py`; adding them is the first implementation step and is a code change on the follow-up branch, not an experiment.

Checks recorded per arm: loss finite every step; `yolo_loss_ref` / `feat_loss_ref` magnitudes; BN running statistics unchanged by evaluation (audit C5 test); checkpoint reload equality (T04 logic); pycocotools mAP > 0 on the held-out 50 frames (any value; not a result); peak GPU memory; wall time per epoch (for the full-run budget).

## 5. Abort criteria

- Any arm fails to run end to end after two fix attempts → report the defect; do not proceed to Gate 3 for that arm.
- Peak memory > 11 GB at batch 8 → halve the batch once; if still over, the arm is "not feasible on this hardware".
- Pilot wall time > 2 h → stop; report throughput; the full comparison is re-budgeted before any approval request.
- Disk below 20 GB free at any point → stop.

## 6. What follows the pilot, and what it would cost (requires explicit approval)

A bounded full comparison: 4 arms × 3 seeds × 50 epochs on the full train split, video-grouped val carved from train, `images_thermal_val` as the single test opening. Estimated 12 runs × ≈ 8 to 10 h ≈ 100 to 120 GPU-h, plus one teacher fine-tune (≈ 6 h) and the cache (≈ 1 h): about 5 to 6 days on GPU 1. Protocol frozen (arms, seeds, metric, cluster bootstrap over videos, no margin needed because the hypothesis is a difference, not equivalence) before the test opening. Expected outcome, given CGDet (arXiv 2511.01435) and Borrow from Anywhere: a small or null advantage of T3 over T2, which is informative for the collaborator's project but is not, on its own, a paper (see `PUBLICATION_OPTIONS.md`).
