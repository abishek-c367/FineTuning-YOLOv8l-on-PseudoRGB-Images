# MATHEMATICAL / STATISTICAL REVIEW — GATE-IR

Scope: what can honestly be stated, what is elementary, what would require real assumptions, and how the statistics of a paper on this project must be done. Nothing here is proposed as a contribution unless explicitly marked.

## 1. Objects and estimands

- Frame x ∈ X (LWIR, 8-bit AGC JPEG in the data actually used; the 14-bit TIFFs are unused), label set Y (boxes+classes), detector f_θ, enhancement actions a ∈ A = {id, gamma, lsrb+lcn}, cost C(a) in ms.
- Per-frame conditional risk r_a(x) = E[ℓ(f_θ(a(x)), Y) | X = x]. For detection, ℓ must be fixed in advance: the Ultralytics training loss is differentiable but not the deployment target; per-frame AP is ill-defined for frames with no objects. A usable surrogate is the per-frame **sum of matched-GT recall shortfall + false-positive count at a fixed operating point**, or the per-frame training loss itself. Choose one before running anything; log both.
- **Oracle action** a*(x) = argmin_a r_a(x) estimated with labels. Oracle headroom H = E[r_id(X)] − E[r_{a*}(X)] ≥ 0 by construction. This is an *offline* quantity. **If H is small relative to seed variance, no router can matter.** This is the first thing to measure (EXPERIMENT_MATRIX B4/B5).
  *Amended 16 Oct 2026 after review:* the oracle is a **GT-informed, infeasible upper bound**, never a deployable router. Because COCO mAP is not additive over images, the headroom is computed by (i) choosing per image, with a pre-specified per-image utility (matched-TP minus FP at a fixed operating point, or the training loss), the enhancer; (ii) pooling the selected predictions over the whole evaluation set; (iii) computing COCO mAP on the pooled set versus the pooled identity predictions; the per-image utility is never averaged as if it were mAP. Selection with the labels of the same model is corrected by seed-split / cross-fitting (§5). Any *router* must be fitted on development videos only, from image features, and applied to sealed held-out videos; its gain is reported beside the oracle bound, the fixed-enhancer baselines, latency, and a video-cluster CI.

## 2. Data-processing inequality (brief §5.8)

For the Markov chain Y → X → G(X) with any measurable G, I(Y; G(X)) ≤ I(Y; X). True, textbook (Cover & Thomas, Thm 2.8.1). Three reasons it is irrelevant to GATE-IR as a claim:
1. The student never sees G(X) at inference; the teacher's features T(G(X)) enter only as a *training signal*. The relevant question is whether the regulariser E‖A(S(X)) − T(G(X))‖² improves the finite-sample risk of the student, which the inequality says nothing about.
2. Even for a model that consumed G(X), the inequality bounds information, not the excess risk of a learned predictor with finite data and a fixed hypothesis class; translation can trivially help by moving inputs into the support of a pretrained model.
3. It would apply equally against every enhancement action a(x) in the proposed router, which the brief treats as beneficial.
Verdict: one background sentence, no theorem.

## 3. Regret of risk-based action selection (brief §5.9)

If sup_{a,x} |r̂_a(x) − r_a(x)| ≤ ε then r_{â(x)}(x) ≤ r_{a*(x)}(x) + 2ε. Proof: r_{â} ≤ r̂_{â} + ε ≤ r̂_{a*} + ε ≤ r_{a*} + 2ε. This is the standard plug-in argument in selective prediction / learning-to-defer (matrix B12). Problems:
- Uniform ε is exactly what fails on rare conditions (fog/rain frames are rare in any thermal dataset without targeted collection); a conditional-coverage statement would need, e.g., ε(x) from conformalised regression of r̂ with group-conditional calibration (Vovk-style split conformal per weather group), which in turn needs labelled weather groups that do not exist in FLIR.
- With a cost term, the same argument gives r_{â} + λC(â) ≤ r_{a*} + λC(a*) + 2ε where a* now minimises the penalised objective; still elementary.
- A non-trivial result would be a *lower* bound showing when selection cannot beat the best fixed action by more than H (the oracle headroom) — but that is just the definition of H. Nothing publishable in theory here; the empirical headroom table is the content.

## 4. Statistics that a paper must get right

### 4.1 Units of analysis and clustering
FLIR ADAS v2 stills are sampled from videos; file names encode `video-<id>-frame-<n>`. Frames from the same video share scene, camera state (AGC), time and weather. **The exchangeable unit is the video, not the frame.** Consequences: (i) train/val/test splits must be by video id (the official `images_thermal_train`/`val` split should be checked for video overlap before use; `video_thermal_test` is a separate set of sequences); (ii) confidence intervals for mAP must come from a **cluster bootstrap over videos** (resample videos with replacement, recompute COCO mAP on the union, 1000+ replicates), paired across methods by using the same resampled index set; (iii) any per-frame "weather" subgroup is nested in videos, so subgroup comparisons need the same clustering and will be very low-powered if few videos contain fog/rain.

### 4.2 Seeds vs sampling uncertainty
Report both: (a) across-seed SD of each metric (≥3 seeds, preferably 5; include all seeds, including diverged runs); (b) test-set sampling uncertainty via the cluster bootstrap for a fixed seed. A difference between methods is claimed only if the paired bootstrap CI of the difference (averaged over seeds, or seed-paired) excludes 0 and the magnitude exceeds the across-seed SD. Do not run significance tests on frames.

### 4.3 Pre-registration items (freeze before touching `video_thermal_test`)
Primary metric: COCO mAP50-95 on the frozen test set, 3 classes, imgsz 512 letterbox, conf 0.001, NMS IoU 0.7 (Ultralytics default), max 300 det. Secondary: AP_small (COCO area < 32²), AP50, per-class AP, batch-1 latency incl. routing/enhancement on a named device. Acceptance criteria per hypothesis are in `EXPERIMENT_MATRIX.md`.

### 4.4 Multiple comparisons
The matrix has ~10 arms. Pre-declare the one primary contrast (e.g. B1 vs B2 for the KD question; oracle vs identity for the headroom question). Everything else is exploratory and labelled so.

### 4.5 Confounding of "weather"
With no weather labels, any gate trained on FLIR learns image-statistic clusters, which correlate with scene type, time of day and AGC behaviour. A gain from routing on FLIR would be a gain from *scene-type* routing. Report it as such. Real-weather external sets (UNIRI-TID, ZUT-FIR-ADAS, DENSE FIR) are the only way to attach the word "weather" to a number; they are single-class and different cameras, so results transfer qualitatively at best.

## 5. What would be a legitimate quantitative contribution (none is a theorem)

1. **Oracle-headroom analysis** with cluster-bootstrap CIs: a clean, reusable methodology for deciding whether input-adaptive preprocessing is worth building. Honest and useful for practitioners; modest novelty.
2. **Teacher-input ablation with matched compute** (no teacher / thermal teacher / COCO teacher on grayscale-replicated thermal / COCO teacher on CycleGAN pseudo-RGB), 3–5 seeds, paired CIs. Answers a question the matrix shows is open (A-matrix §E.4a).
3. Calibration of the student under enhancement actions (reliability diagrams of detection confidence per action) — descriptive, supports the "enhancement can delete weak evidence" claim if true.

Anything beyond this (regret bounds, information-theoretic arguments) should be omitted.
