# SECOND PROJECT — AI-generated-image forensics proposal (`Research_Proposal.pdf`)

**Verdict: STOP as framed. DEFER unless re-scoped to one ablation on top of DRCT.** This project is kept strictly separate from GATE-IR and consumes none of its compute budget.

Source: literature check run on 15 Oct 2026 by a delegated search agent; every row was verified against the publisher/arXiv/GitHub page, items it could not open are marked UNVERIFIED. I (the auditor) have re-read the proposal PDF and agree with the assessment below; where I disagree I say so.

## What the proposal is

Real and synthetic images → random JPEG compression / resize → DDIM forward-noise + reverse-denoise at a random timestep `t` (diffusion model trained on real images) → ResNet-50 or ViT-B/16 + MLP → contrastive loss where positives are any two images from the same set (both real, or both synthetic) → attach a classifier. Motivating numbers ("99%+ clean, 70–75% at 0.5× resize, 60% at JPEG-75") are given without citation.

## Prior-work table (verified)

| Paper | Date | Venue | Link | Contribution | Overlap with proposal |
|---|---|---|---|---|---|
| DIRE (Wang et al.) | Mar 2023 | ICCV 2023, DOI 10.1109/ICCV51070.2023.02051 | https://arxiv.org/abs/2303.09295 | DDIM inversion (20 steps) + reconstruction with an ADM model; feeds the residual to a ResNet-50 classifier. Tests blur σ=1–3 and JPEG Q=65/30 (figure only). | Same noise-then-DDIM-denoise idea; no contrastive loss. |
| DRCT (Chen, Zeng, Yang, Yang) | Jul 2024 | ICML 2024 (Spotlight), PMLR 235:7621–7639 | https://proceedings.mlr.press/v235/chen24ay.html, code https://github.com/beibuwandeluori/DRCT | Reconstructs real AND fake images through SD-inpainting (VAE-latent noise-then-DDIM-denoise, 50 steps); treats reconstructions as hard "fake" samples; trains ConvNeXt-B or CLIP-ViT-L/14 with λ·margin-contrastive + (1−λ)·CE (λ=0.3). Same-label pairs are positives. Sec. 4.4 tests resize 0.5–1.5 and JPEG Q=60–100. Ablation Table 4: contrastive term lifts GenImage cross-set average 76.99→83.53%. Released code offers `NTXentLoss`, `TripletMarginLoss`, `CircleLoss`, `ArcFace` as loss options. | **Near-complete overlap.** Diffusion reconstruction + label-driven contrastive loss + classifier with a resize/JPEG robustness section is DRCT; the proposal's NT-Xent variant is a `--loss_name` switch in the released code. |
| Revisiting DIRE / UR2EA (Lin et al.) | Jan 2026 (online Sep 2025) | Neural Networks 193:108084 | https://doi.org/10.1016/j.neunet.2025.108084 | Reconstruction residuals from pretrained diffusion models as a universal prior; attention module; UniversalForensics dataset (30 generators); +3.3% avg acc. | Reconstruction family; no contrastive loss in abstract. JPEG/resize numbers UNVERIFIED (paywalled). |
| LaRE² (Luo et al.) | Mar 2024 | CVPR 2024 | https://arxiv.org/abs/2403.17465 | Single-step latent reconstruction error + error-guided refinement; 8× faster than DIRE. | Reconstruction feature; no contrastive. |
| AEROBLADE (Ricker et al.) | Jan 2024 | CVPR 2024 | https://arxiv.org/abs/2401.17879 | Training-free LPIPS distance to LDM-autoencoder reconstruction. | Reconstruction only. |
| FatFormer (Liu et al.) | 2024 | CVPR 2024 | https://openaccess.thecvf.com/content/CVPR2024/html/Liu_Forgery-aware_Adaptive_Transformer_for_Generalizable_Synthetic_Image_Detection_CVPR_2024_paper.html | CLIP + forgery-aware adapter + language-guided contrastive alignment. | Contrastive (image–text), no reconstruction. |
| UnivFD (Ojha, Li, Lee) | Feb 2023 | CVPR 2023 | https://arxiv.org/abs/2302.10174 | Frozen CLIP ViT-L/14 + linear probe. Cozzolino et al. CVPRW 2024 (https://openaccess.thecvf.com/content/CVPR2024W/WMF/html/Cozzolino_Raising_the_Bar_of_AI-generated_Image_Detection_with_CLIP_CVPRW_2024_paper.html) report ≤10 AUC-point loss under JPEG, collapse at 25% rescale, clean AUC 84.1→71.8 under combined crop/resize/compress. | Baseline. |
| Co-SPY (Cheng et al.) | 2025 | CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/html/Cheng_CO-SPY_Combining_Semantic_and_Pixel_Features_to_Detect_Synthetic_Images_CVPR_2025_paper.html | Semantic + VAE-artifact branches designed to survive JPEG; Co-SpyBench (22 generators). | JPEG robustness as explicit goal; no reconstruction/contrastive. |
| D³ (Yang et al.) | Apr 2024 / Mar 2025 | arXiv | https://arxiv.org/abs/2404.04584 | Parallel branch fed a distorted copy as discrepancy signal. | Distorted-view input adjacent to proposal. |
| SAFE (Li et al.) | Aug 2024 | KDD 2025 | https://arxiv.org/abs/2408.06741 | Crop-not-resize, invariant augmentations, patch masking. | None; AIGIBench shows F.Acc 0.0% at JPEG Q=50. |
| AIDE (Yan et al.) | Jun 2024 | ICLR 2025 | https://proceedings.iclr.cc/paper_files/paper/2025/hash/b0303773962ea1b5394c3a83cc7dd066-Abstract-Conference.html | CLIP semantics + frequency-patch experts; Chameleon benchmark. | None; AIGIBench: 67.0%→1.5% F.Acc at JPEG Q=50. |
| Community Forensics (Park, Owens) | Nov 2024 | CVPR 2025 | https://openaccess.thecvf.com/content/CVPR2025/html/Park_Community_Forensics_Using_Thousands_of_Generators_to_Train_Fake_Image_CVPR_2025_paper.html | 2.7M images / 4,803 generators; JPEG + resize augmentation in training. | Data-scaling route. |
| FIRE (Chu et al.) | Dec 2024 | CVPR 2025 | https://arxiv.org/abs/2412.07140 | Frequency-decomposed reconstruction error. | Reconstruction family; no contrastive. |
| DEUA (Huang et al.) | 2025 | ICCV 2025 | https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Diffusion_Epistemic_Uncertainty_with_Asymmetric_Learning_for_Diffusion-Generated_Image_Detection_ICCV_2025_paper.html | Diffusion epistemic uncertainty + asymmetric margin loss. | Diffusion signal + margin loss. |
| AIGIBench (Li et al.) | May 2025 | NeurIPS 2025 D&B | https://arxiv.org/abs/2505.12335 | 11 detectors × 23 fake subsets with JPEG Q=50, noise, up-down sampling. JPEG drives most detectors' F.Acc to 0–2%. | The benchmark to use. |
| GAPL (Qin et al.) | Dec 2025 | CVPR 2026 | https://arxiv.org/abs/2512.12982 | Generator-aware prototypes + LoRA; 90.4% mean clean acc; 11.09% drop under extreme JPEG. | Current SOTA reference. |
| B-Free (Guillaro et al.) | Dec 2024 | arXiv (CVPR 2025 per GAPL; UNVERIFIED) | https://arxiv.org/abs/2412.17671 | Content-matched real/fake pairs via SD conditioning. | Same content-matched-pair logic as DRCT. |
| Fake or JPEG? (Grommelt et al.) | Mar 2024 | arXiv (ECCV-W 2024, UNVERIFIED) | https://arxiv.org/abs/2403.17608 | GenImage detectors learn JPEG/size biases; debiasing +11 pts cross-generator. | Explains the proposal's motivating drops. |
| SupCon (Khosla et al.) | 2020 | NeurIPS 2020 | https://proceedings.nips.cc/paper_files/paper/2020/file/d89a66c7c80a29b1bdbab0f2a1a94af8-Paper.pdf | Label-driven contrastive loss. | This is the loss the proposal actually describes. |

## How the proposal differs from DRCT and from SupCon

- **Loss naming.** "Positives = any two images from the same set" is Supervised Contrastive Learning (Khosla et al. 2020), not SimCLR. SimCLR has exactly one positive per anchor (the other view of the same image). The proposal cites SimCLR's NT-Xent but describes SupCon.
- **Reconstruction.** DRCT: SD-inpainting latent noise-then-DDIM-denoise, fixed schedule. Proposal: pixel-space DDIM with a random timestep on a real-trained model. DIRE already uses a real-trained ADM; LaRE² already uses single-step noising at one t. Random t is the only untested knob.
- **Input to the detector.** DRCT keeps the original image and adds the reconstruction as a hard *fake*; the proposal feeds only reconstructions and drops originals, discarding the real-vs-real-reconstruction hard-negative structure that DRCT's ablation identifies as the main gain (68.98→76.03→83.53%).
- **Degradation before reconstruction.** Applying random JPEG/resize *before* the DDIM step is not in DRCT (DRCT augments the detector input). This is the only component with a plausible mechanism and it is cheap to test.
- **Backbones.** ResNet-50 / ViT-B/16 are weaker than DRCT's ConvNeXt-B / CLIP ViT-L/14.

## The quoted numbers

Unsupported. No published benchmark reports the triplet (99 / 70–75 / 60). DRCT reports its own detectors staying "up to 99%" under resize 0.5–1.5 and JPEG 60–100 (figure only); Cozzolino et al. report ≤10 AUC-point JPEG loss for UnivFD; AIGIBench reports far *worse* than 60% for AIDE/SAFE/NPR at JPEG Q=50 and only mild drops for 0.5× up-down sampling; GAPL reports an 11.09% drop under extreme JPEG. The proposal's numbers must be replaced by a cited table or measured in-house.

## Verdict and the one pilot that would be informative

**STOP** as framed:
1. DRCT (ICML 2024, open code, 2M-image dataset) already publishes diffusion reconstruction + label-contrastive + classifier with a resize/JPEG section; the proposed loss is a config flag in that repo.
2. The proposed input removes the hard-sample structure that produced DRCT's gain; the expected result is at or below DRCT.
3. The framing mislabels SupCon as SimCLR; both components are 2+ years old.
4. Motivating numbers are uncited and contradicted by AIGIBench.
5. The field has moved to scale (Community Forensics, GAPL) and bias-free pair construction (B-Free); a ResNet-50 pipeline on GenImage will not be competitive on AIGIBench/Chameleon.

**If a pilot is run anyway (one experiment, ~1 GPU-week, separate budget):** take the released DRCT/ConvNeXt-B pipeline, keep everything fixed, add exactly one change: random JPEG (Q∈[50,95]) / resize (0.5–1.5×) applied to each image *before* SD reconstruction. Evaluate on AIGIBench JPEG-Q50 and up-down-sampling splits plus GenImage cross-set against unchanged DRCT. Pre-declared pass criterion: ≥10-point F.Acc improvement under JPEG-Q50 over DRCT; otherwise nothing beyond DRCT survives. Only if that passes, stack the random-t pixel-space DDIM knob.

Auditor's note: the "degrade-before-reconstruct" pilot is the only part of this proposal I would defend to a reviewer, and it is an ablation of someone else's method. On this machine (2× RTX 2080 Ti, 11 GB) SD-inpainting reconstruction of a GenImage-scale set is impractical; this pilot needs the cluster.
