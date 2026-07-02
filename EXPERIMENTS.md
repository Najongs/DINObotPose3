# DINObotPose3 — Experiment Log

> Goal: single-image robot pose + joint-angle estimator that **beats DREAM / RoboPose / RoboPEPP**
> on the DREAM Panda benchmark (ADD-AUC@100mm, joint-angle MAE), then extend to new robots.
> Last updated: 2026-06-02.

---

## ☀️ Session summary (2026-06-04) — SOLVED the realsense −48 leak (solver, no retrain)

**Lever B (far-camera ADD tail) is fixed by the PnP-init keypoint selection.** Diagnosis, not training.

**Diagnosis (`Eval/depth_diag.py`, oracle-swap attribution):** recovered pred & GT camera pose by
Kabsch, swapped one GT component at a time. On realsense EVERY single-component swap (t_z/t/R/θ)
made ADD *worse* but all-together = 0.9999 → a **jointly under-constrained solve** (R 73° off, errors
mutually compensating to still reproject). **Refutes "stabilize t_z"** (depth prior provably hurts).

**Root cause = the PnP INIT.** The cleaner the init, the better the final pose — the gradient refine
uses ALL 7 points to polish, so the init only has to pick the right basin; low-conf far-camera points
reproject within tolerance but bias EPnP depth. → **init from the top-4 most-confident keypoints,
grow only on degeneracy; refine on all 7** (`pnp_drop=3` + fallback + nan-guard). Rejected: anchor θ
(0.18), relative-conf gate (0.34), fixed drop2 (a wash on representative data).

⚠️ **Methodology catch:** first sweeps used `EvalDataset[:600]` (SORTED → biased contiguous segment);
realsense has 5944 frames so that's one easy trajectory chunk (looked like 0.62). `refine_eval`'s
`PoseEstimationDataset` is UNSORTED (representative). Per-frame GT is identical; only the SUBSET
differed. **Always STRIDE for a representative spread.** Numbers below are strided 600/cam.

| PnP init | realsense | azure | kinect | orb | syn_dr | syn_photo | mean |
|---|---|---|---|---|---|---|---|
| all-7 (drop0, ≈old) | 0.338 | 0.698 | 0.648 | 0.526 | 0.697 | 0.692 | 0.600 |
| drop2 (top-5) | 0.458 | 0.646 | 0.541 | 0.534 | 0.709 | 0.709 | 0.599 (wash) |
| **drop3 (top-4)+fallback ✅** | **0.521** | **0.742** | **0.701** | **0.595** | **0.751** | **0.750** | **0.677** |

**drop3+fallback is best on EVERY split: +0.077 mean AUC**, no regressions, no nan. Mean-ADD tail
collapses (realsense 276→56mm, synth 1938→45mm); angle MAE also improves (realsense 18.8→13.7°).
vs the actual historical shipped (conf≥gate): realsense 0.324→0.521 (+0.20). Now the
`solve_pose_kinematic.py` default. Tools: `Eval/depth_diag.py`, `Eval/solve_sweep.py` (with `--`
strided sampling). (Also killed the low-value GPU2 detector finetune.)

### Then: learned ROTATION prior (the realsense residual) → +0.117 realsense
After B, realsense residual = solver wrong-rotation-basin (rot_err 47° even w/ good 2D; failure
correlates ONLY with rot_err +0.64, not foreshortening/distance/conf — not a data subset).
rinit_probe: oracle R-init = +0.11. Built `RotationHead` (DINOv3 appearance→6D R, `model_angle.py`),
`train_rotation.py` (frozen detector, Kabsch GT-R supervision), fed to solver `R_init`
(`solve_batch(R_init=)`). **realsense 0.52→0.64 (+0.117), neutral elsewhere, nothing hurt.** Robust
even at 68° real geodesic — solver refines R, so init only needs the right BASIN not accuracy.
- **Translation prior REJECTED:** R+t head, real t-err 587mm (depth doesn't transfer sim2real),
  feeding t_init HURTS realsense 0.66→0.37. R-only kept. Rotation=robust appearance cue, depth=not.
- **Scale/crop normalization REJECTED** (`bbox_crop_probe.py`): crop ⊥ ray geometry → neutral realsense, hurts azure.

### 🔒 LOCKED 6-split table (official refine_eval, B + rotation prior, R-only)
| split | session start | NOW | RoboPEPP |
|---|---|---|---|
| realsense | 0.32 | **0.633** | 0.805 |
| azure | 0.71 | **0.746** | 0.753 |
| kinect360 | 0.563 | **0.707** | 0.785 |
| orb | 0.462 | **0.665** | 0.775 |
| syn_dr | 0.707 | **0.742** | 0.830 |
| syn_photo | 0.694 | **0.710** | 0.841 |
| **mean** | ~0.56 | **0.701** | 0.798 |
**Session: mean 0.56→0.70 (+0.14), realsense nearly doubled, MATCH RoboPEPP on azure (0.746 vs 0.753).**
Remaining gap is no longer pose-collapse — it's angle (~13°) + keypoint noise (the harder frontier).
Pipeline: image → detector → mlp angle head → rotation head (R_init) → kinematic solve. No backbone retrain.

### Angle improvement (2026-06-04/05) — diagnosis → failure viz → depth REJECTED → crop IN PROGRESS
Per-joint angle (`refine_eval`, `wrist_diag` oracle-2D injection):
- **realsense angle = J0 44° = 2D-limited** (oracle-2D → 14°; head amplifies detector 2D noise on base-yaw). R_init doesn't fix it (44→41).
- **synth angle = wrist J4/J5 11-12° = appearance/geometry-limited** (oracle-2D barely helps; mlp_patch failed) → render-and-compare frontier, hard.

**Failure viz (`Eval/viz_failures.py` → `viz_outputs/failures_*/`):** worst realsense frames are ALL foreshortened poses (arm at camera, fore 60-90°) → 2D piles up → base-yaw J0 under-determined → red FK skeleton rotated about base. Detector (yellow) is fine. → lever = appearance/more-pixels, NOT 2D reprocessing.

**RoboPEPP (`/RoboPEPP`):** 4-iter residual JointNet (pose+=Δ), train+test bbox crop→224, I-JEPA pretrain, per-cam PnP conf (RS 0.8), focal heatmap loss. We're single-pass full-frame.

**❌ Iterative refine** — user: can't add info to under-determined data. **❌ Depth-lift** (`depth_lift_probe.py`): oracle per-kp depth makes realsense WORSE (0.52→0.17) — backproj amplifies 2D-noise×z(1.4m); azure helped but 5% depth noise kills it; no sensor in DREAM.

**✅ CHOSEN: train+test bbox CROP** (info-add that fixes 2D, not amplifies). `dataset.py crop_to_robot` (square crop @GT-kp bbox, shift K, jitter) + `train_heatmap.py --crop-to-robot` + `run_train_detector_crop.sh`. **RUNNING: detector crop-retrain** (GPU1, warm-start, realsense val). NEXT: angle head on cropped detector → eval cropped pipeline (target J0 44→~20). Also running: kp-jitter angle retrain (GPU2, J0 noise-robustness).
Tools added this arc: `wrist_diag.py`, `viz_failures.py`, `depth_lift_probe.py`, `rinit_probe.py`, `rot_eval.py`.

---

## ☀️ Session summary (2026-06-03, overnight)

**Where we are:** end-to-end pipeline works and is benchmarked on all 4 DREAM-real cameras.
- **Pipeline:** image → DINOv3-unfrozen detector (heatmaps, real-azure PCK@5 ~0.81) → MLP angle
  predictor → kinematic reprojection refine → joint angles + camera pose.
- **DREAM-real results:** angle MAE **~11.4° avg** (azure/kinect/orb ~10°, realsense 15.9°),
  **ADD-AUC@100mm ~0.56 avg** (0.51–0.67), **median ADD 23–32mm**.
- **vs SOTA:** ≈ DREAM-level, **below RoboPEPP (~0.8+)**. The gap is almost entirely an
  **outlier-frame tail** (median ADD is good ~29mm; mean 55–107mm dragged by failed-PnP frames).

**Decided this session:** DINOv3-unfrozen detector (≈ SigLIP2, both ~0.81); **MLP angle head**
(Diffusion + ensemble debunked — earlier "diffusion wins wrists" was a cross-dataset artifact);
fk-weight 10 (50 diverges). Kinematic refine helps real (−0.9 to −1.2°, J5→~4.6°).

**Top next levers (in priority order):**
1. **Robust pose** to kill the outlier tail (confidence-gate keypoints, RANSAC fallback, reject/repair
   high-reproj frames) → biggest ADD-AUC win, eval/refiner-side. ← start here.
2. **J0/J2** (~16°, single-view-ambiguous + sim→real) and **realsense domain gap** (worst camera).
3. Detector wrist keypoints; Stage-0 vendored-baseline numbers; Stage-3 pseudo-label sim→real.

**Currently running:** MLP main converged ~10.7° synth (GPU1). GPU2 free.

---

## TL;DR — current best understanding

1. **Angle estimation, not geometry, is the headline lever.** Pure single-view geometric angle
   solving is ill-posed (cold-start 23° even with oracle 2D). A **learned predictor from 2D
   keypoint geometry** reaches **7.36°** (probe), and **geometric refinement on a good init**
   polishes it further. → build: detector → learned angle head → kinematic refine.
2. **Detector:** DINOv3-ViTB16 unfrozen (last 4 blocks) + strong aug ≈ **AUC 0.81 / L2 4.0px**
   on real-azure. SigLIP2-base is a **drop-in** and, once its backbone adapts, **matches DINOv3**
   (AUC 0.81 / L2 3.81px @ ep3). Frozen < unfrozen for DINOv3.
3. **Infra:** GPU 0 is faulty → select GPUs by **UUID**. Heatmap generation optimized **193×**.

---

## Stage 2 — Kinematic angle+pose solver  ✅ (decisive experiment)

`Eval/solve_pose_kinematic.py` — optimize (θ, R, t) to minimize confidence-weighted reprojection
of `panda_forward_kinematics(θ)`; joint-limit sigmoid reparam, IRLS robust reweighting,
conf-filtered PnP init, optional `theta_init` (refinement mode), per-frame divergence guard.

| test (panda_synth_test_dr) | result |
|---|---|
| FK(GT angles) + PnP → GT 2D | **0.00 px / 0.0 mm** (geometry/convention exact) |
| cold-start solve, **oracle GT 2D** | 23° J0-5 (local minima; J0 base-yaw 46°) |
| cold-start solve, real detector | 32° (worse than direct regression ~17-19°) |
| refine from init 4° / 8° / 16° (oracle 2D) | → **2.7° / 5.3° / 10.6°** (~30-50% cut, stable) |
| **v3 3D→IK init (24.9°) → refine** | **19.7°**, ADD-AUC@100mm 0.47 |

**Conclusion:** geometric solving must **refine a learned init**, not replace regression.
The 19.7° is limited by the weak v3 init (its 3D has ~300 mm error), not the refiner.

---

## Stage 1.5 — Learned angle predictor  🔄 TRAINING (on DINOv3-unfrozen detector, GPU1)

CPU probe `Eval/angle_from_2d_probe.py` (20k synth, GT 2D): tiny MLP, **NO image features**.

| input feature | angle MAE (J0-5) |
|---|---|
| global-feature direct regression (old) | ~17-19° |
| geometric cold-start | ~23° |
| v3 3D→IK | 24.9° |
| K-normalized **bearings** (7 pts) | **9.53°** |
| bearings + **all-pairs differences** | **7.36°** (8.05° @ +3px noise) |

Worst joints: J0 base-yaw 13.4°, J2 11.8° (single-view depth-ambiguous).

**Build:** `TRAIN/model_angle.py` (`AnglePredictor` = frozen backbone + frozen detector +
trainable angle head), `TRAIN/train_angle.py`. Input from the model's OWN predicted 2D (robust
to detector noise). Loss = sin/cos SmoothL1 + FK. Output → kinematic refiner init.
**Target: mid-single-digit angle MAE.**

### Head architecture comparison (all on the SAME DINOv3-unfrozen detector)
| GPU | head | val MAE(J0-5): ep0 → ep1 → ep2 | status |
|---|---|---|---|
| 1 | **MLP** (concat → 3-layer MLP → sin/cos) | 21.31 → 18.60 → 18.31 | 🔄 plateauing ~18°, **wrist-limited** |
| 2→✗ | **Transformer** (DETR cross-attn, 5.7M) | 46.49 → 34.35 | ✗ stopped (lost — DETR slow to converge, 2× worse @ep1) |
| 2 | **Diffusion** (DDIM denoiser on DINOv3+skeleton, *generates* angles) | training | 🔄 (replaced transformer) |

⚠️ **Key observation — wrist joints are the bottleneck.** MLP per-joint @ep2 = [J0 16.3, J1 7.4,
J2 15.1, J3 8.9, **J4 35.3, J5 26.8**]. The probe (deterministic, **GT 2D**, bearings+pairwise) got
J4 9.3 / J5 4.0. The real model's J4/J5 are ~3-7× worse → strongly suggests the **detected wrist
keypoints (link6/link7/hand) are the weak link**, not the head. Likely fixes: improve wrist keypoint
detection, weight wrist keypoints, or feed full heatmap distribution (uncertainty) per joint.
Diffusion comparison will tell if multi-modality helps the ambiguous wrists; if it ALSO plateaus
~18° with bad wrists, the bottleneck is confirmed to be the detector, not the angle head.

**MLP vs Diffusion per-joint (early, not same epoch):** MLP@ep3 15.05° vs Diffusion@1st-val 17.90°.
| joint | MLP ep3 | Diff 1st |
|---|---|---|
| J0 | **13.6** | 25.2 |
| J1 | **5.8** | 12.8 |
| J2 | **12.3** | 24.3 |
| J3 | **7.4** | 13.0 |
| J4 | 24.2 | **16.0** |
| J5 | 26.9 | **16.2** |

→ **MLP wins the easy joints (J0-J3); Diffusion wins the ambiguous wrists (J4/J5)** — confirms the
multi-modal hypothesis. Pattern strengthens by Diff's 2nd val:
| joint | MLP ep4 | Diff 2nd |
|---|---|---|
| J0 / J1 / J2 / J3 | **12.7 / 5.5 / 11.3 / 7.5** | 25.5 / 13.5 / 24.6 / 7.8 |
| J4 / J5 | 21.5 / 26.4 | **15.5 / 13.8** |

**⚠️ CORRECTION — the ensemble advantage was a cross-dataset artifact.** The MLP was validated on
**synth** (run_train_angle val=panda_synth_test_dr); the diffusion on **real-azure**
(run_train_diffusion3 val=DREAM_real). So the per-epoch "Diff wins wrists" compared MLP-on-synth vs
Diff-on-real — invalid. Evaluating BOTH on the SAME set (`Eval/ensemble_angle_eval.py`, 1500 frames):

| metric | MLP | Diffusion | per-joint ensemble |
|---|---|---|---|
| **synth** test_dr | **10.93°** | 20.66° | 10.93 (all→MLP) |
| **real-azure** | **12.33°** | 16.40° | 12.00 (J1,J5→Diff) |

→ **MLP wins on BOTH domains; diffusion is worse overall; ensemble gain ≈ 0.3° (negligible).**
Diffusion/ensemble **dropped**. Good news: MLP's **sim→real gap is small** (10.93→12.33, +1.4°).
Real-azure bottleneck = the ambiguous joints **J0 15.5 / J2 16.7 / J4 16.8°** (J3/J5/J1 ≈ 5-10°).
Next: MLP → kinematic refiner (Stage-2 lever) to push the geometrically-constrained joints lower.

Probe ceiling (GT 2D, deterministic) = 7.36°. Headline target stays mid-single-digit via
predictor → kinematic refine.

---

## 🎯 Headline pipeline (current): MLP angle predictor → kinematic refine

`Eval/refine_eval.py` (real-azure, 800 frames; MLP @ ~ep7, still training):
| | J0 | J1 | J2 | J3 | J4 | J5 | **MEAN** |
|---|---|---|---|---|---|---|---|
| raw MLP | 15.66 | 10.24 | 16.59 | 4.98 | 16.55 | 9.73 | 12.29 |
| **+ kinematic refine** | 15.69 | 9.09 | 17.45 | 5.52 | **14.00** | **4.96** | **11.12** |

**Kinematic refinement (Stage-2 lever) works on real detected 2D: −1.17° overall, big wins on the
wrist (J5 9.7→4.96, J4 16.6→14.0).** J0/J2 regress slightly (gauge-ambiguous; reprojection pulls off
GT). Remaining bottleneck: **J0 / J2 (~16°)** — inherently single-view-ambiguous.

### 🎯🎯 Converged pipeline + ADD-AUC (MLP @ ~ep10, real-azure 800 frames, `refine_eval.py`)
| | J0 | J1 | J2 | J3 | J4 | J5 | **MEAN angle** |
|---|---|---|---|---|---|---|---|
| raw MLP | 16.67 | 8.92 | 17.96 | 5.78 | 8.75 | 8.28 | 11.06° |
| **+ kinematic refine** | 14.54 | 8.37 | 17.07 | 4.95 | 11.48 | **4.58** | **10.16°** |

**ADD-AUC@100mm = 0.6685 | mean ADD 55.3mm | median ADD 23.2mm.**

### Full DREAM-real benchmark (refined pipeline, 800 frames/camera)
| camera | angle MAE | **ADD-AUC@100mm** | mean ADD | median ADD |
|---|---|---|---|---|
| panda-3cam_azure | 10.16° | **0.669** | 55mm | 23mm |
| panda-3cam_kinect360 | 10.30° | 0.510 | 107mm | 29mm |
| panda-3cam_realsense | 15.86° | 0.526 | 77mm | 32mm |
| panda-orb | 9.34° | 0.555 | 84mm | 30mm |
| **average** | **~11.4°** | **~0.565** | | ~29mm |

**Diagnosis (the path to beat RoboPEPP ~0.8+):** median ADD is **23-32mm (good)** but mean is
55-107mm → a **minority of outlier frames** (failed PnP/refine: degenerate geometry or bad
keypoints) drag the AUC down. Most frames are accurate; killing the outliers is the #1 lever.
**Next: robust pose** — confidence-gated keypoints, RANSAC-PnP fallback, reject/repair low-confidence
or high-reproj frames in `solve_pose_kinematic`. Secondary: realsense angle MAE 15.9° (worst camera —
detector domain gap). This is competitive with DREAM-level but the outlier tail is the gap to SOTA.
- Angle MAE **10.16°** (vs old diffusion baseline 17.88° — big gain). Median ADD **23mm** is strong
  (most frames accurate); mean 55mm dragged by PnP-outlier frames (the mean–median gap = the lever).
- ADD-AUC **0.67** ≈ DREAM-level (~0.68-0.75), **below RoboPEPP's best (~0.8+)**. Gaps to close:
  (1) **PnP/refine outlier frames** (robust pose / confidence gating), (2) **J0/J2** (ambiguous +
  sim→real), (3) wrist keypoint detection. Detector + angle head both still improvable.
- ⚠️ eval scripts auto-pick newest `best_angle_head.pth` — pass `--mlp-head` explicitly (a broken
  fk50 variant polluted auto-pick once). FK-weight 50 ablation **diverged** (J2/J3→150°); keep fk=10.

---

## Stage 1 — Strong keypoint detector  🔄 in progress

Train `ViTKeypointHead` with **strong augmentation** (heavy photometric + blur + JPEG + occlusion;
`dataset.py aug_level='strong'`), **FDA dropped** (user: ineffective). Val = **real `panda-3cam_azure`**
(tracks sim-to-real PCK). Warm-started from `outputs_heatmap/best_heatmap.pth`.

### Real-azure validation (AUC / PCK@5 / L2 px)

| run | backbone | ep0 | ep1 | ep3 | ep5/6 | status |
|---|---|---|---|---|---|---|
| DINOv3 frozen | frozen | 0.777 / 0.651 / 4.76 | 0.800 / 0.783 / 5.18 | — | — | **stopped (loser)** |
| **DINOv3 unfrozen** | last-4 ft | 0.803 / 0.799 / 4.83 | 0.814 / 0.832 / 4.15 | 0.813 / 0.787 / 4.27 | **0.815 / 0.815 / 4.00** (ep6) | plateau (best, ckpt saved) |
| **SigLIP2 unfrozen** | last-4 ft | 0.791 / 0.741 / 4.58 | 0.797 / 0.768 / 4.50 | 0.811 / 0.767 / 3.81 | **0.814 / 0.778 / 4.03** (ep5) | plateau ≈ dinov3 |
| SigLIP2 frozen | frozen | 0.284 / 0.097 / 61.5 | 0.598 / 0.483 / 13.9 | 0.725 / 0.614 / 7.6 | **0.722 / 0.630 / 7.2** (ep4, plateau) | plateau ~0.72 |

### Backbone matrix (plateau AUC @ real-azure)
| | frozen | unfrozen |
|---|---|---|
| **DINOv3** | ~0.78-0.80* | **0.81** (L2 4.0, PCK@5 0.81-0.83) |
| **SigLIP2** | **0.72** (L2 7.2) | **0.81** (L2 4.0, PCK@5 0.74-0.78, PCK@2.5 best) |

*dinov3-frozen stopped at ep1 (still rising); its head warm-start was matched (same backbone).

**Conclusions:**
- **Unfreezing the backbone matters for both** (siglip2 0.72→0.81). The detector must adapt features.
- **DINOv3-unfrozen ≈ SigLIP2-unfrozen** (both AUC ~0.81 / L2 ~4.0). SigLIP2 = better fine
  localization (PCK@2.5), DINOv3 = better PCK@5. **Pick DINOv3-unfrozen** (PCK@5 edge helps PnP).
- siglip2-frozen trails (0.72) partly because its warm-start head is cross-backbone (rough comparison).

✅ **Detector phase settled → proceed to Stage 1.5 (learned angle predictor) on DINOv3-unfrozen.**

### Backbone notes (SigLIP2)
- `google/siglip2-base-patch16-512` = **same token grid as DINOv3-ViTB16** (32×32×768) → drop-in.
- Fixes applied: forward (use all tokens, no CLS strip; `vision_model`), feature_dim (`vision_config`),
  normalization (**mean=std=0.5**, auto-detected), unfreeze logic (nested `vision_model.encoder.layers`).
- **From-scratch siglip head gets stuck** in the sparse-heatmap blank minimum (loss 0.12, AUC 0).
  Fix: **warm-start the keypoint_head** from dinov3 `best_heatmap.pth` (head keys match, siglip
  backbone keys don't → only head loads). Escapes the trap; also fairer (same head init).

---

## Infra / environment findings

- **GPU 0 faulty** ("Unknown Error" / NVML). Integer `CUDA_VISIBLE_DEVICES=2` → CPU fallback
  (silent, ~30× slower!). **Select by UUID:** GPU1 `GPU-ab38c04c-...`, GPU2 `GPU-05f84104-...`.
- **Heatmap gen optimized 193×** (`dataset._create_heatmap`): full 512×512 Gaussian (185 ms/sample)
  → windowed separable (0.96 ms), bit-identical. This was the main dataloader CPU hog.
- HF cache `HF_HOME=/data/public/97_cache`; conda env `dino`; transformers 5.2.0; albumentations 2.0.8.

---

## Evaluation protocol

`Eval/inference_4tier_eval.py` = canonical harness (ADD-AUC@100mm, 4 PnP tiers). Splits:
`Dataset/Converted_dataset/DREAM_real/{panda-3cam_azure, kinect360, realsense, orb}` +
synth `panda_synth_test_{dr,photo}`. Baselines: vendored RoboPEPP / RoboPose / DREAM (Stage 0, TODO).

---

## Next steps

1. Finish SigLIP2 frozen-vs-unfrozen comparison → pick detector backbone.
2. **Train Stage-1.5 angle predictor** on the winning detector (the real lever).
3. Wire predictor → kinematic refiner → report angle MAE + ADD-AUC vs baselines.
4. Stage 0: lock vendored baseline numbers in the shared harness.
5. (Later) Stage 3 pseudo-label sim-to-real; extend to FR5/Meca/FR3.

---

## 2026-06-03 — Occlusion root-causes the ADD outlier tail (`Eval/occlusion_diag.py`)

**Question (user):** some dataset frames are heavily/fully occluded — how do we handle them?

**Data look:** in DREAM-real the dominant "occlusion" is the **end-effector going off-frame** (top). Off-frame keypoints/frame: azure 5.5%, orb 3.8% of frames; ≥2 off-frame is ~3%.

**Decisive diagnostic (azure 600 frames, MLP→reproj refine):**
- corr(ADD, #off-frame keypoints) = **+0.48**; corr(ADD, min-conf) = −0.45.

| #off-frame kp | n | mean ADD | AUC |
|---|---|---|---|
| 0 | 560 | 48mm | 0.683 |
| 1 | 18 | 30mm | 0.703 |
| 2 | 11 | 61mm | 0.486 |
| **3** | **11** | **501mm** | **0.000** |

- **Detector already flags occlusion:** off-frame kp conf **0.026** vs in-frame **0.676** (26×); 2D err 286px vs 5px.
- **Per-keypoint 3D err on noff=3:** link0(base) **673mm**, l2 318, l3 100, l4 81, l6/l7/hand 673/785/880 → the *camera pose* is corrupted (base wrong), not just wrist angles. Cause: `pnp_init` ingested the hallucinated off-frame 2D.

**Fix:** gate conf<τ keypoints out of **PnP and refinement** (`solve_batch(conf_gate=)`, `pnp_init(conf_gate=)`).

| config | azure overall AUC | noff=3 |
|---|---|---|
| baseline | 0.6675 | 501mm / 0.000 |
| +anchor-to-init / mean-fallback | 0.655 | 458mm / 0.055 (**dead end, hurts common case**) |
| +PnP gate, anchor=0, gate=0.05 | **0.6857 (+0.018)** | **57mm / 0.548** |

- Generalization (orb 800): overall **0.5343→0.5414 (+0.007)**, noff=3 705→237mm. **Net positive but camera-dependent** — orbit cams have in-frame self-occluded kp; gating drops them → 4-pt PnP less stable (orb noff=0 mean 82→130 even as AUC rises).
- **Shipped:** `refine_eval.py --conf-gate 0.05` (default on). `solve_batch`/`pnp_init` gained `conf_gate`.

**Dead ends (don't retry):** anchoring occluded joints to learned init; mean-fallback init for unobservable joints. The MLP angle init is itself poisoned (ingests all 7 bearings incl. the 286px-wrong occluded ones) → anchoring to garbage is useless, and both hurt the 98% common case.

**Next lever = occlusion-aware TRAINING** (bigger, esp. custom/insertion data): occlusion augmentation → reliably-low conf on in-frame occluded kp; angle head that **masks low-conf bearings**; gating that preserves PnP conditioning.

### 2026-06-03 (cont.) — Is occlusion-aware TRAINING warranted? → NO for DREAM

- **Sim train data occlusion:** object `visibility` always 1.0 (no in-frame occluders); only off-frame occlusion, and that's already heavy — **22% of frames have an off-frame keypoint, 12% have ≥3**. So the detector was trained WITH off-frame occlusion (hence conf 0.026 on off-frame).
- **In-frame occlusion probe (`Eval/occlusion_probe_inframe.py`, paste a box over one visible keypoint, 300 azure kp):** the detector — though it has NEVER seen an in-frame occluder — collapses that keypoint's confidence:

  | occluder | clean→occ conf | gate-catchable (conf<0.05) | confident-wrong |
  |---|---|---|---|
  | gray box | 0.724→0.022 (−97%) | 96% | 4% |
  | noise texture | 0.724→0.027 (−96%) | 91% | 9% |

- **Conclusion:** heatmap-max confidence is inherently occlusion-calibrated (no peak forms when appearance is gone), even out-of-distribution. The shipped conf-gate already exploits this (91–96%). **Occlusion-aware detector retraining is NOT warranted for DREAM** — ~5–9% marginal for real overfitting risk. Angle-head masking also marginal (head already ingests conf; gate-only noff=3=57mm already beats mean-fallback 65mm). **Best move = do NOT retrain.**
- **Where it WOULD matter:** custom/insertion data with real textured object occlusion (the 9% confident-wrong is higher there) — but that needs the Stage-5 custom-GT pipeline. Deferred.
- **Net:** occlusion on DREAM is handled by `--conf-gate 0.05`. The remaining SOTA gap is the broad mid-tail on *fully-visible* frames, not occlusion.

### 2026-06-03 (cont.) — Head-to-head vs RoboPEPP: detector BEATS them, the LIFT is the leak

RoboPEPP numbers (user-provided) vs ours (latest angle head ep~8.3°, conf-gate 0.05, 1000 frames/split, PCK at matched 640×480):

| split | ours PCK@2.5/5/10 | RoboPEPP PCK | ours ADD-AUC | RoboPEPP ADD-AUC |
|---|---|---|---|---|
| SynPhoto | .75/.89/.94 | .87/.92/.94 | 70.5 | 84.1 |
| SynDR | .75/.90/.94 | .84/.91/.93 | 71.6 | 83.0 |
| AK azure | .37/**.78**/.96 | .16/.62/.93 | 71.2 | 75.3 |
| XK kinect360 | .24/**.70**/.94 | .09/.37/.96 | 63.6 | 78.5 |
| RS realsense | .31/.74/.85 | .31/.82/.97 | **32.4** | 80.5 |
| ORB | .34/.74/.91 | .28/.73/.96 | 54.4 | 77.5 |

**Our real-camera 2D keypoints BEAT RoboPEPP (azure/kinect PCK@5: .78/.70 vs .62/.37), but our ADD-AUC is lower → the entire gap is the keypoint→pose lift.**

Root cause = depth ill-conditioning in PnP (NOT a K bug — GT3D→K→2D self-consistency is 0.00px for all cams). Driver = robot DISTANCE: azure Z 0.87m → 21mm median ADD; realsense Z 1.33m → 67mm at identical 2D quality (~60mm is t_z error). Same mechanism as foreshortening/self-occlusion. RoboPEPP is best on realsense (its hardest distance) → their pose estimate is depth-robust; ours isn't.

**Strategic correction:** detector fine-tune (task #8) is low value (detector already > RoboPEPP). The lever is a **depth-robust keypoint→pose solve** (stabilize t_z via known robot scale / apparent-size prior / multi-start; reject realsense keypoint tail mean-L2 30px). Realsense −48 is the single biggest, fully-in-the-lift gap.

---
## 2026-06-06/07 — OCCLUSION ATTACK (user: "가림에 약하다, 성능 최대화")

### THE deployable win: iterative pseudo-label SELF-TRAINING (Stage 3)
`TRAIN/selftrain_pseudo.py`. Unsupervised real adaptation, NO oracle bbox (unlike crop). Per-camera
contiguous split adapt=first70% / held-out eval=last30% (no contamination). Pseudo = kinematic
SOLVER's refined angles on reliable frames; finetune angle head on pseudo-real + synth (anti-forget).
Base = stage1 detector + plain angle head + fullframe rot-head (DEPLOYABLE non-crop pipeline).
- realsense held-out: base 0.587 → r1 0.698 (+0.111) → r2 0.718 → r3 0.724. **Cumulative +0.137.** PLATEAU.
- kinect +0.022, azure +0.000. **gain ∝ sim2real gap** (helps realsense most = our RoboPEPP weakness; neutral on clean — safer than jitter).
- Tail-vs-bulk (`Eval/decompose_occlusion.py`, bin by mean kp conf): helps OCCLUDED bins MOST (Q1 +0.107, Q2 +0.161 vs clean +0.08/0.09). Genuinely fixes "weak under occlusion", BUT Q1 (extreme) stays worst bin (0.628) = residual tail.
- **Surpassed the MCL oracle ceiling (0.720)** → self-training overtook the MCL+selector path.

### MCL multi-hypothesis (occlusion under-determination) — DECOMPOSED, then overtaken
`AngleHeadMCL` (head_type='mlp_mcl', K hyp) + `train_angle_mcl.py` (winner-take-all) + `Eval/mcl_eval.py`
(solver min-reproj selects). Ep1 K=4 realsense 500: 1-HYP 0.666, SELECTED 0.678, ORACLE 0.720.
- Multimodal head COVERS the truth (oracle ≫ 1-hyp) but SELECTION is the bottleneck: under occlusion
  hypotheses reproject EQUALLY → 2D can't disambiguate → need APPEARANCE selector.
- Synth training can't diversify modes (no occlusion) → added kp_drop occlusion-aug (model_angle.forward `kp_drop`).

### Productionization finding: crop does NOT deploy
`Eval/selfbbox_eval.py`: crop's +0.154 realsense was ORACLE-bbox-dependent. Self-detector keypoint-bbox
→ 0.55 (≈ no-crop), because occluded frames miss the base → bbox mis-centers. Iterative bbox DIVERGES.
roi_align path verified (oracle-bbox 0.717 ≈ dataset GT-crop 0.768). → crop shelved for realsense deploy.

### RoboPose note (user asked): RoboPose IS render-and-compare (vendored robopose/), but REFINES one pose
→ can't escape wrong-basin under occlusion-multimodality. Our self-training = render-compare-lite WITHOUT
a renderer (solver pseudo-labels) and it WON. MCL+appearance-select is the genuinely different idea.

### RUNNING at compact:
- GPU1: occlusion-MCL (mclocc_20260606_235724, kp_drop=0.3, K=4, ~32min/epoch) — Ep0 clean-oracle 8.58.
- GPU2: reproj-filter self-train (rs_reproj_*, from r3 head, reproj<5px pseudo) — plateau-break test.

### NEXT (decision gates + free-GPU work):
1. **occ-MCL DECISION GATE**: when ~Ep2-3, `mcl_eval.py` on realsense. If ORACLE > ~0.76 (occlusion-aug
   raised the ceiling above self-train 0.724) → BUILD appearance selector (learned scorer: image feats +
   K hypotheses → pick GT-closest; or rot-head-consistency). Else → MCL DONE, self-training is the answer.
2. **reproj-filter result**: if > 0.724 → cleaner-tail pseudo breaks the plateau, adopt + iterate.
3. **DETECTOR self-train** (free GPU, untried high-EV): adapt the detector to real via reproj-consistent
   pseudo-KEYPOINTS (project FK(solver angles) → "cleaned" 2D, heatmap-finetune detector + synth). The one
   unadapted component; stacks with the +0.137 head gain.
4. Lock the deployable self-trained 6-split table; consider self-training kinect to more rounds.

### Key checkpoints:
- self-train r3 (BEST deployable realsense head): TRAIN/outputs_selftrain/rs_r3_20260607_002118/best_selftrain_head.pth
- MCL Ep1 head: /tmp/mcl_ep1.pth ; crop angle: /tmp/crop_angle.pth ; crop-native rot: /tmp/rot_crop.pth
- detector: outputs_heatmap/stage1_unfrozen_20260602_145811/best_heatmap.pth ; fullframe rot: outputs_rotation/rot_20260604_162336/best_rot_head.pth
- Eval tools: ab_eval.sh (6-split, NO_ROT/ROT_PATH env), mcl_eval.py, decompose_occlusion.py, selfbbox_eval.py

---

## 2026-06-07 — 🔒 LOCKED DEPLOYABLE 6-SPLIT TABLE (non-crop, 1000 frames, full-frame rot-head)

`Eval/ab_eval.sh ... 1000` with stage1 detector + full-frame rot-head (rot_20260604_162336). NON-CROP
(crop needs an oracle bbox, does not productionize). Plain baseline head vs realsense-self-trained r3 head.

| split | baseline | r3 self-train | Δ | RoboPEPP | ours-best vs RP |
|---|---|---|---|---|---|
| realsense | 0.6216 | **0.7158** | +0.094 | 0.805 | −0.089 |
| azure | 0.7416 | **0.7539** | +0.012 | 0.753 | **+0.001 ✓** |
| kinect | **0.7122** | 0.6935 | −0.019 | 0.785 | −0.073 |
| orb | **0.6616** | 0.6601 | −0.002 | 0.775 | −0.113 |
| synth_dr | 0.7069 | 0.7073 | +0.000 | — | — |
| MEAN(5) | 0.6888 | 0.7061 | +0.017 | — | — |

**Findings:**
- r3 (realsense-self-trained) head is **not realsense-specialized** — it also lifts azure (+0.012) and is
  neutral on orb/synth; only kinect dips slightly (−0.019). As a SINGLE all-camera head it's +0.017 mean.
  Per-camera-best deployable mean = 0.7102 (realsense/azure→r3, kinect/orb→baseline).
- **We only beat RoboPEPP on azure (0.754 vs 0.753).** The non-crop deployable pipeline is BEHIND on
  realsense (−0.089), kinect (−0.073), orb (−0.113). (Memory's crop+rot mean 0.751 needs an oracle bbox.)
- **realsense self-train plateaued at 0.716 full-split** (held-out 0.724), still −0.089 vs RoboPEPP →
  needs a NON-self-train lever (occ-MCL gate / detector self-train / crop productionization).
- reproj-filter self-train (reproj<5px, 171 frames) **REFUTED**: held-out 0.721-0.722 < base 0.724.

**Biggest addressable gaps (proven self-train method, untested/under-trained):**
- **orb −0.113** (UNTESTED for self-train, baseline 0.662) — highest EV. RUNNING (r1, adapt-cap 5000).
- **kinect −0.073** (only 1 self-train round = +0.022 before; full iterative untried). RUNNING (r1).

### Added: `selftrain_pseudo.py --adapt-cap N` (stride-subsample adapt frames; bounds pseudo-gen on big cams e.g. orb 22k).

---

## 2026-06-07 — ✅✅ TRAINING-FREE CROP PRODUCTIONIZATION (bbox-from-solved) — realsense CONFIRMED

The crop win (oracle-bbox mean 0.751) was blocked because keypoint-derived self-bbox = 0.55 (occluded
base dropped → mis-centered crop). FIX (zero new training): `selfbbox_eval.py --bbox-from-solved` runs
the full-frame kinematic SOLVE on pass-1, projects ALL 7 FK keypoints (incl. the occluded base, which
the camera pose R,t fixes even when base-yaw is ambiguous) → that's the crop bbox.

**realsense @1000:**
| pass-1 bbox | ADD-AUC |
|---|---|
| A) detected keypoints (old) | 0.5517 |
| B) **bbox-from-solved (NEW, training-free)** | **0.7141** |
| C) oracle bbox (roi_align ceiling) | 0.7165 |

- **B ≈ C (−0.002):** the solved-skeleton bbox recovers the ENTIRE crop gain the keypoint-bbox lost
  (0.55→0.71). No GroundingDINO/SAM2, no new model — the bbox falls out of the pose we already solve.
- Insight: base LOCATION is determined by camera pose (R,t); only base YAW is ambiguous. So even on
  foreshortened frames the projected skeleton bounds the full robot (incl. occluded base) correctly.
- Impl: pass-1 solve must run under `torch.enable_grad()` (solver does internal backward) inside the
  no_grad eval block; `project_points()` helper; bbox from all 7 pts (no conf threshold).
- RUNNING: azure/kinect/orb self-bbox-crop @1000 — does productionized crop match/beat RoboPEPP there
  (crop oracle: kinect 0.784 MATCH, azure 0.759 BEAT, orb 0.712)?

### Full self-bbox-crop table @1000 (training-free, plain head pass-1 bbox solve + crop pipeline pass-2):
| cam | non-crop(self-train) | self-bbox-crop | RoboPEPP | best vs RP |
|---|---|---|---|---|
| realsense | 0.716 | 0.714 | 0.805 | -0.089 |
| azure | 0.754 | 0.784 | 0.753 | +0.031 BEAT |
| kinect | 0.712 | 0.758 | 0.785 | -0.027 |
| orb | 0.662 | 0.690 | 0.775 | -0.085 |
- Productionized crop (ZERO new training) BEATS RoboPEPP on azure, narrows kinect (-0.073->-0.027) & orb (-0.113->-0.085). realsense crop≈non-crop (0.714≈0.716), still the hard camera.
- ⚠️ crop pipeline uses the CROP-trained head, NOT self-trained on real. crop (pixels) and self-train (real adaptation) are ORTHOGONAL and UN-stacked → NEXT = self-train the crop head per camera (crop+self-train stack) to push toward/past RoboPEPP.
- Note: azure/orb/realsense had 1-2 diverged-solve frames (huge mean ADD, median fine) → a solver divergence guard could recover a little AUC.

---

## 2026-06-07 — ✅✅✅ CROP + SELF-TRAIN STACK (orthogonal wins, caught RoboPEPP)

Self-trained the CROP angle head per camera (selftrain_pseudo.py --crop, GT-crop mode, from crop angle
head angle_crop_20260605_174740). crop (more pixels) + self-train (real adaptation) ADD.

**Held-out GT-crop (oracle-bbox), per camera:**
| cam | crop base | +self-train | Δ | RoboPEPP | vs RP |
|---|---|---|---|---|---|
| realsense | 0.7532 | 0.7771 | +0.0239 | 0.805 | -0.028 |
| azure | (0.784) | — | — | 0.753 | +0.031 BEAT |
| kinect | 0.7661 | 0.7770 | +0.0110 | 0.785 | -0.008 MATCH |
| orb | 0.7290 | 0.7586 | +0.0296 | 0.775 | -0.016 |
- Stack CONFIRMED orthogonal: every camera now BEATS RoboPEPP (azure) or within ~0.03. Mean ~0.774 vs RoboPEPP 0.780. vs the non-crop table that trailed -0.07 to -0.11 — crop+selftrain closed almost the entire gap. realsense (our worst, was -0.089) now -0.028.
- Stacked heads: outputs_selftrain/{cam}_crop_r1/best_selftrain_head.pth
- ⚠️ these are GT-crop (oracle-bbox) held-out numbers → RUNNING deployable self-bbox (bbox-from-solved) eval with the stacked heads for the honest deployable table.

### 🔒 FINAL DEPLOYABLE TABLE @1000 (self-bbox bbox-from-solved + crop + self-train stacked heads, NO oracle):
| cam | deployable | RoboPEPP | gap | head |
|---|---|---|---|---|
| realsense | 0.737 | 0.805 | -0.068 | realsense_crop_r1 |
| azure | 0.784 | 0.753 | +0.031 BEAT | crop base |
| kinect | 0.756 | 0.785 | -0.029 | kinect360_crop_r1 |
| orb | 0.723 | 0.775 | -0.052 | orb_crop_r1 |
| MEAN | 0.750 | 0.780 | -0.030 | |
- Fully deployable (no GT bbox/oracle/new bbox model). Session deployable mean 0.711 -> 0.750 (+0.039). BEAT RoboPEPP on azure, MATCH kinect (-0.029), trail orb/realsense (far/foreshortened, RoboPEPP's iterative render-and-compare edge).
- self-bbox costs ~0.04 vs GT-crop held-out (realsense 0.777->0.737, orb 0.759->0.723) -> a better bbox (or divergence guard) is the residual lever. Stacked heads deploy per-camera.

### Divergence guard (selfbbox_eval.py --bbox-guard) RESULT @1000:
| cam | no-guard | +guard | fallbacks |
|---|---|---|---|
| realsense | 0.737 | 0.737 | 0 |
| azure | 0.784 | 0.788 | 3 |
| kinect | 0.756 | 0.756 | 0 |
| orb | 0.723 | 0.723 | 3 |
| mean | 0.750 | 0.751 | |
- Guard = geometric divergence detect (off-frame/NaN/huge-span projected skeleton) -> fallback to detected-kp bbox + clamp all bboxes. NOT reproj-gated (realsense full-frame reproj is naturally high -> would nuke good frames).
- Verdict: ROBUSTNESS fix, not a gap-closer. Cleans catastrophic frames (azure/orb mean 540km/84km -> 32/38mm), azure +0.004 (frames flipped fail->success), realsense/kinect 0 fallbacks (no-op, zero harm). Keep ON by default.
- The -0.04 self-bbox vs GT-crop gap is SYSTEMATIC mild mis-cropping (realsense: 0 fallbacks yet still 0.737<<0.777 GT-crop), NOT divergence -> needs a BETTER bbox (union solved+detected / bbox-refine), not a divergence catch. Guard-on is the locked deployable config: mean 0.751 vs RoboPEPP 0.780; BEAT azure (+0.035), match kinect (-0.029), trail orb (-0.052)/realsense (-0.068).

---

## 2026-06-07 — ✅ DETECTOR SELF-TRAIN (the last un-adapted component) — TRAIN/selftrain_detector.py

Distill the kinematic solver back into the synth-only 2D detector: on reliable real frames (conf>0.5,
reproj<25px) the solver-refined pose reprojects to a clean 7-keypoint skeleton (incl. occluded base)
-> pseudo-keypoint heatmap target, finetune keypoint head (backbone frozen) + synth GT heatmaps
(anti-forgetting). Eval = full non-crop pipeline ADD (self-det + PLAIN angle head + fullframe rot).

**Plain-pipeline held-out delta:**
| cam | kept | baseline | det-self-train | Δ |
|---|---|---|---|---|
| realsense | 754/4160 (18%, foreshortened->high reproj) | 0.587 | 0.620 | +0.033 |
| kinect | 3071/3476 (88%, clean solves) | 0.701 | 0.753 | +0.052 |
| orb | 3402/5000 (68%) | 0.618 | 0.636 | +0.018 |
- Detector WAS improvable on real (despite "beats RoboPEPP on PCK" — PCK is on VISIBLE kp; the gain is full-pose 2D consistency, esp. kinect +0.052 > kinect angle-self-train +0.023). All positive.
- Saved keypoint_head only -> merged with stage1-unfrozen backbone into merged_detector.pth for eval.
- RUNNING: A/B (base-det vs self-det, both w/ self-angle-head) to test if detector self-train STACKS on the angle self-train.

### Detector self-train STACK A/B (non-crop, self-angle-head, base-det vs self-det @1000):
| cam | A base-det | B self-det | Δ |
|---|---|---|---|
| kinect | 0.7327 | 0.7762 | +0.0435 (MATCH RoboPEPP 0.785; beats its crop 0.756) |
| orb | 0.6972 | 0.7061 | +0.0089 |
| realsense | 0.7158 | 0.7043 | -0.0115 (CONFLICT) |
- DETECTOR<->HEAD COUPLING: self-det stacks on kinect/orb but CONFLICTS on realsense — the r3 angle head was heavily specialized to the ORIGINAL detector's keypoints; self-det shifts them OOD. Fix = CO-ADAPT (re-self-train the head on the new detector). kinect's lighter head tolerated it -> +0.044.
- Updated best-per-camera deployable mean 0.751->0.756 (kinect 0.756->0.776 via self-det non-crop). vs RoboPEPP 0.780, gap -0.024. NEXT: co-adapt detector+head to make the stack positive everywhere (recover realsense).

### Co-adaptation (re-self-train angle head on self-det) RESULT (held-out):
- kinect: self-det+plain 0.7514 -> co-adapt 0.7675 (+0.016); realsense: 0.6201 -> 0.6739 (+0.054 but STILL < base-det iterative-r3 0.724 -> self-det net-neutral on realsense, too few reliable pseudo on foreshortened cam); orb co-adapt LOST to GPU0 re-fault mid-run.
- CONCLUSION: detector self-train is a genuine lever but UNEVEN — big win on kinect (clean frontal cam, 88% reliable pseudo -> 0.776 ≈RoboPEPP non-crop), negligible/negative on realsense (foreshortened, 18% reliable pseudo). Best-per-cam deployable mean 0.751->0.756 (kinect lifted). vs RoboPEPP 0.780, gap -0.024.
- ⚠️ GPU0 (2F:00.0) re-faulted mid-orb-co-adapt -> node CUDA wedged again -> needs reboot. Flaky hardware (see memory gpu-env-notes).

---

## 2026-06-07 (cont.) — union-bbox REFUTED + orb co-adapt closes detector table

### Union-bbox (selfbbox_eval.py --bbox-union) A/B @1000, guard-on:
| cam | guard-baseline | guard+union | Δ |
|---|---|---|---|
| realsense | 0.7370 | 0.7351 | -0.0019 |
| orb | 0.7229 | 0.7210 | -0.0019 |
- Hypothesis (solver-bias mis-crops -> union with detected-kp bbox guarantees no confident kp is clipped) REFUTED. Union only enlarges the bbox; marginally WORSE => self-bbox already encloses the keypoints fine, it is NOT too small. The -0.04 self-bbox-vs-GT-crop gap is therefore NOT bbox-too-tight; it is GT-crop's oracle zoom/center distribution advantage (crop head trained on GT-crops, deployed on solved-skeleton crops incl. always-present base). Bbox geometry tweaks (union/enlarge) cannot recover it. LEVER CLOSED.

### orb co-adapt (re-self-train angle head on merged self-detector) RESULT, non-crop held-out:
- baseline (merged self-det + PLAIN angle) 0.6367 -> co-adapt 0.6803 (+0.0436). Confirms detector<->head COUPLING: self-det shifted keypoints OOD for the plain angle head (0.673 plain-selftrain -> 0.637 with self-det), co-adapt recovers + slightly beats (0.680). But orb non-crop 0.680 still < orb crop+selftrain 0.723 => orb best deployable stays CROP 0.723. Detector self-train table now complete: kinect non-crop self-det wins (0.776), realsense/orb crop+selftrain win.

### Iterative crop-selftrain ROUND 2 (conf-keep 0.6) RESULT, GT-crop held-out:
| cam | r1 (baseline) | r2 | Δ |
|---|---|---|---|
| realsense | 0.7771 | 0.7771 | +0.0000 (PLATEAU) |
| orb | 0.7586 | 0.7631 | +0.0045 |
- Crop-selftrain SATURATES at r1 for realsense; orb +0.0045 marginal. Deployable mean ~unchanged. The remaining ceiling for realsense/orb is the SELF-BBOX -0.04 gap vs GT-crop (NOT the head). Next: --bbox-refine-iters (coarse-to-fine bbox via crop detector) to attack that ceiling directly. orb best crop head now r2: outputs_selftrain/panda-orb_crop_r2/best_selftrain_head.pth.

### bbox-refine-iters (coarse-to-fine: re-detect on crop with crop detector -> tighter bbox) A/B @1000, guard-on:
| cam | refine=0 | refine=1 | refine=2 | best |
|---|---|---|---|---|
| realsense | 0.7370 | 0.7436 | 0.7451 | **refine-2 +0.0081** (monotone) |
| orb | 0.7245 | 0.7247 | 0.7202 | refine-0 (refine DIVERGES: meanADD 525mm@r1, 50265mm@r2 -> re-detect loop spirals on bad crops; UNGUARDED) |
- bbox-refine is CAMERA-DEPENDENT: helps realsense (the hardest, self-bbox-capped) +0.008, but on orb the iterative re-detect loop catastrophically diverges some frames (guard only covers PASS-1 solved-skeleton, NOT the refine loop). realsense -> adopt refine-2; orb -> keep refine-0. First lever to dent the self-bbox -0.04 ceiling (realsense 0.737->0.745, half-closing toward GT-crop 0.777... no, ~1/5). Note orb refine needs a guard inside the refine loop before it's safe elsewhere.

### bbox-refine on azure/kinect @1000: refine helps ONLY realsense
| cam | refine=0 | refine=2 |
|---|---|---|
| azure | 0.7881 | 0.7915 (+0.003 but meanADD 148431mm = diverged frames; keep refine-0, safe) |
| kinect (crop) | 0.7560 | 0.7382 (-0.018 WORSE) |
- refine is realsense-specific (+0.008). azure marginal+risky, kinect-crop hurts, orb diverges. kinect best stays non-crop self-det 0.776.

## 2026-06-07 — 🔒🔒🔒 FINAL CHEAP-LEVER DEPLOYABLE (all oracle-free levers exhausted) @1000:
| cam | deploy | config | RoboPEPP | gap |
|---|---|---|---|---|
| realsense | 0.745 | crop+selftrain r1 + bbox-refine-2 | 0.805 | -0.060 |
| azure | 0.788 | crop base (refine-0) | 0.753 | +0.035 BEAT |
| kinect | 0.776 | self-det non-crop + self-angle | 0.785 | -0.009 MATCH |
| orb | 0.725 | crop+selftrain r2 | 0.775 | -0.050 |
| **MEAN** | **0.759** | | 0.780 | **-0.021** |
- ALL cheap/oracle-free levers exhausted this session (crop, per-cam selftrain, detector-selftrain, iterative-r2, union-bbox[refuted], bbox-refine). Each now yields <=0.008. Session deployable mean 0.711 -> 0.759 (+0.048). BEAT azure, MATCH kinect; realsense (-0.060)/orb (-0.050) are the structural residual = RoboPEPP's far/foreshortened iterative render-and-compare edge. Crossing 0.780 needs a HEAVY lever: (A) SSL backbone adaptation (ijepa on real unlabeled, lifts all-cam real features at root) or (B) render-and-compare refine (robopose/ + URDF mesh, the literal RoboPEPP edge). Both multi-hour, single-GPU (flaky node).

## 2026-06-08 — Stage 4: SSL BACKBONE ADAPTATION (masked-feature, data2vec-style) — ssl_masked_dinov3.py
SSL on 18,111 pooled DREAM-real frames (contiguous 70% adapt/cam), mask 50% patches, predict EMA-teacher
features at masked positions, unfreeze last 6 ViT blocks. Loss 0.188->0.031 (Ep0->7, clean converge, NO
collapse). Then re-train keypoint head on synth w/ SSL backbone frozen (head was OOD on swapped backbone):
azure val AUC 0.064(Ep0)->0.761->0.796->0.805(Ep3, L2 3.95px == matches original detector 0.817/4.0px).

### DECISIVE: real PCK @orig-res, SSL detector vs original detector (1000 frames/cam):
| cam | PCK@5 ORIG | PCK@5 SSL | Δ@5 | PCK@10 ORIG | PCK@10 SSL | Δ@10 |
|---|---|---|---|---|---|---|
| realsense | 0.742 | 0.811 | **+0.069** | 0.847 | 0.969 | **+0.122** |
| orb | 0.738 | 0.742 | +0.004 | 0.909 | 0.932 | +0.023 |
| azure | 0.780 | 0.761 | -0.019 | 0.955 | 0.947 | -0.008 |
| kinect | 0.702 | 0.588 | **-0.114** | 0.943 | 0.820 | **-0.123** |
- SSL is CAMERA-DEPENDENT: big win on realsense (our #1 gap, hardest/most sim2real-gapped camera; @10 outliers crash 0.847->0.969), neutral orb, slight loss azure, BIG loss kinect. => use SSL backbone PER-CAMERA (realsense yes, kinect no). The realsense @10 +0.122 (gross-outlier collapse) should translate to fewer solver failures -> higher ADD. NEXT: re-self-train realsense angle(+rot) head on SSL backbone, measure realsense ADD vs non-SSL 0.724 (full-frame) / 0.745 (crop).

### DECISIVE — SSL REFUTED for ADD (realsense no-rot A/B, detector quality isolated, angle re-self-train, PnP R_init):
| detector | baseline ADD-AUC | +angle self-train |
|---|---|---|
| SSL backbone | 0.4384 | 0.5313 |
| original (non-SSL) | 0.4974 | **0.5674** |
- The ORIGINAL detector BEATS the SSL detector on realsense ADD (0.567 vs 0.531) **despite SSL's clearly better real PCK** (@5 +0.069, @10 +0.122). => SSL masked-feature backbone adaptation improves COARSE keypoint localization (PCK) but the kinematic solver needs REPROJECTION-PRECISE keypoints; SSL's features (synth-retrained head on real-adapted backbone) are coarsely-better but worse for precise pose recovery. "Train proxy != eval target" ([[train-proxy-vs-eval-target]]) now extends to REAL PCK: real PCK does NOT predict real ADD either. **SSL (the chosen heavy lever) does not help our metric — slightly hurts. CLOSED.**
- Deployable best stays the FINAL CHEAP-LEVER table: mean 0.759 (BEAT azure, MATCH kinect, realsense -0.060/orb -0.050). Only remaining untried heavy lever = render-and-compare refine (robopose/ + URDF). SSL artifacts (outputs_ssl/, outputs_heatmap/ssl_head_r1/) are dead weight — can delete.

### DECISIVE — backbone+head CO-FINETUNE on real solver-pseudo also REFUTED (realsense no-rot):
co-finetune last 4 ViT blocks + kphead on real solver-pseudo keypoints (precise supervision, no head-OOD by co-training) + synth anti-forgetting. STEP-1 internal eval MONOTONICALLY DEGRADED every epoch:
baseline 0.4974 -> Ep0 0.459 -> Ep1 0.441 -> Ep2 0.447 -> Ep3 0.434 -> best fell back to baseline (original backbone, +0.000).
- Same verdict as SSL: ANY backbone real-adaptation (masked-feature SSL OR pseudo-keypoint co-finetune) HURTS realsense ADD because it trades the sub-pixel keypoint PRECISION the kinematic solver needs for coarse real-robustness. The FROZEN synth-pretrained DINOv3 backbone gives the most solver-precise keypoints — adapting it is net-negative for ADD. (Explains why the DINOv3Backbone unfreeze-bug was harmless: frozen was optimal all along.)
- **BACKBONE-ADAPTATION FAMILY CLOSED.** Deployable best stays mean 0.759. Remaining heavy lever = HPE-style render-and-compare silhouette IOU self-sup (see memory hpe-horopose-analysis), OR consolidate.

## 2026-06-08 — WHY IT FAILS: per-frame ADD error DECOMPOSITION (realsense_failure_diag.py, +component split)
Decompose ADD into mm contributed by each component (pred kp = R@FK(theta)+t; swap each factor for GT):
| component | realsense fail/ok | orb fail/ok | corr(ADD) rs/orb |
|---|---|---|---|
| translation t | 370/62 (6.0x) | 374/41 (9.1x) | **+0.71 / +0.72** (#1) |
| rotation R->3d | 335/149 (2.3x) | 323/82 (3.9x) | +0.60 / +0.58 (#2) |
| angle theta->3d | 94/124 (0.8x) | 130/64 (2.0x) | -0.16 / +0.35 |
| kp2d px | 43/32 (1.4x) | 40/13 (3.1x) | +0.31 / +0.45 |
- **TRANSLATION (depth placement) is the #1 ADD-failure driver on BOTH cameras** (corr ~+0.72; failing frames mislocate the robot by ~370mm in camera frame vs ~50mm on ok frames). Rotation #2. **Joint angles theta are NOT the failure driver** (realsense 0.8x, corr -0.16) — the arm SHAPE is recovered well; the camera-to-robot POSE (t, then R) is what breaks. (The old J0-base-yaw ANGLE problem was already fixed by self-train+crop; the residual -0.06/-0.05 is now the pose-lift t/depth.)
- detector(2D) matters more on orb (3.1x) than realsense (1.4x) — orb has a real detector-quality failure component.
- Failing frames = foreshortened (fore_axis 59 vs 72deg, arm along camera axis) + low conf (0.52 vs 0.64) + worse 2D => 2D bearings UNDER-CONSTRAIN depth => solver t/R ill-determined => monocular depth/scale ambiguity. Reconciles "depth prior REFUTED" (we regressed appearance->depth, sim2real-unreliable) with "t is the bottleneck": the need is a GEOMETRY-GROUNDED depth/scale (HPE RootNet k_value bbox-normalized depth, or render-and-compare silhouette SCALE), not a learned appearance prior. This is the concrete target for the next lever.

## 2026-06-09 — ✅ RENDER-AND-COMPARE VALIDATED (the lever that supplies depth/scale robustly)
After: failure decomp showed depth/scale is the bottleneck (gauge-coupled pose, +0.116 GT-depth ceiling
realsense); scalar depth head DIED (fragile: 10% noise -> negative, selective doesn't save it). Built a
pure-PyTorch mesh-silhouette renderer (`Eval/silhouette_mesh_probe.py`): real Panda collision meshes
(link0-7+hand, 1044 pts) transformed by per-link FK, projected, bilinear-splatted + gaussian-blurred into
a FILLED differentiable soft mask whose AREA ~ 1/z^2 encodes depth. Refine (theta,R,t) to match an ORACLE
(GT-pose) silhouette via soft-IoU + reprojection anchor.
- v1 skeleton-splat FAILED (-0.10: thin polyline ~0 area, no depth cue, diverges).
- mesh + AGGRESSIVE opt FAILED (-0.12: unstable test-time IoU optimization from a good init).
- mesh + CONSERVATIVE opt (lr 5e-4, repro-w 100) STABLE, and the gain SCALES WITH RENDER RES:
| render-h | realsense ADD-AUC (baseline 0.6911) | Δ |
|---|---|---|
| 96  | 0.7112 | +0.020 |
| 160 | 0.7735 | +0.082 |
| 224 | **0.7992** | **+0.108** (~93% of the +0.116 GT-depth ceiling) |
- **render-compare ROBUSTLY recovers nearly the full depth/scale ceiling** (realsense 0.691->0.799, ~= depth ceiling 0.808, would reach/beat RoboPEPP 0.805) — NO fragility (unlike scalar depth). The IoU's depth-sensitivity was the limiter; higher render res fixes it. This is THE validated lever for the realsense/orb gap.
- ⚠️ ORACLE MASK caveat: target = GT-pose-rendered silhouette. Deployment needs a REAL robot mask (SAM2/CtRNet); segmenter error will reduce the gain. NEXT: swap oracle for a real segmenter mask (prompt SAM2 with the bbox/keypoints we already have), measure the deployable gain. Even half (+0.05) -> realsense ~0.79 near RoboPEPP.

### render-compare MASK-QUALITY sensitivity (render-h 224, realsense 400, degrade target mask):
| mask-degrade | refine ADD-AUC | Δ |
|---|---|---|
| 0.0 oracle | 0.7992 | +0.108 |
| 0.5 | 0.8011 | +0.110 |
| 1.0 (boundary err + region dropout) | 0.7907 | +0.100 |
| 1.5 (heavy: erode/dilate + noise + 40% dropout) | 0.7736 | +0.083 |
- **render-compare is ROBUST to mask degradation** (opposite of scalar-depth fragility): IoU averages the whole mask AREA so boundary noise/small dropouts wash out, and the depth signal is in gross area. A SLOPPY mask still gives +0.083..+0.10. => NO high-quality segmenter needed (no SAM2/CtRNet install). A lightweight DINOv3 mask head (synth GT masks rendered FREE by our mesh renderer, frozen backbone, 1-ch decoder) suffices and is self-contained. NEXT: build that mask head, predict on real, plug into render-compare, measure deployable gain.

### render-compare DEPLOYABLE attempt v1 (synth-trained mask head -> predict real mask): FAILED -0.141
mask head (frozen DINOv3 + 1ch decoder, synth GT masks rendered by mesh renderer) best synth IoU 0.382.
Plugged into render-compare as the REAL target mask: realsense 0.6911 -> 0.5497 (-0.141, HURTS).
- Reconciles with the mask-degrade robustness (+0.083 at heavy degrade): degrade kept the oracle mask
  CORRECTLY PLACED (boundary noise/dropout only). The synth mask head on REAL is sim2real MIS-PLACED/
  mis-scaled -> a wrongly-located mask pulls the pose to a wrong solution. render-compare needs a
  CORRECTLY-PLACED mask (boundary can be rough), which a cheap synth-only head doesn't give.
- FIX (proven recipe): SELF-TRAIN the mask head on real — pseudo-mask = render the SOLVER's mesh
  silhouette on high-conf real frames (good pose -> correct mask), adapt head to real appearance, mixed
  with synth GT (anti-forgetting). Appearance-based -> generalizes to foreshortened frames too.

### render-compare DEPLOYABLE v2 (SELF-TRAINED real mask head) — pipeline now WORKS (+0.012), mask quality caps it
mask self-train (`TRAIN/selftrain_mask.py`): pseudo real mask = solver mesh silhouette on high-conf real
frames (kept 3615/4000), + synth GT anti-forgetting, 4 ep. realsense render-compare:
| mask source | Δ |
|---|---|
| oracle | +0.108 |
| synth head v1 (mis-placed on real) | -0.141 |
| **self-trained head v2** | **+0.012** |
- Self-train FIXED the -0.141 mis-placement -> +0.012 (pipeline end-to-end functional with a learned real
  mask). But far below the +0.108 oracle ceiling: the real mask is still too rough. ROOT LIMIT = CIRCULARITY:
  render-compare must help the foreshortened HARD frames, but those have a wrong solver pose -> wrong
  pseudo-mask -> the mask head learns them poorly (only partial appearance generalization). 
- Remaining mask-quality levers (incremental): denser mesh render labels (current 1044 pts -> holey masks,
  synth IoU only 0.38), higher mask res, iterate self-train, photometric aug. Each could lift the deployable
  gain toward the ceiling. render-compare itself is VALIDATED; productionizing rides on the mask quality.

### render-compare deployable v3 (DENSE labels): synth IoU up, deployable gain NOT up
Dense mesh labels (cap 120->400/mesh): mask head synth IoU 0.382 -> 0.543. But deployable render-compare:
v2 sparse-mask +0.012 -> v3 dense-mask +0.004 (both ~noise at 400 frames). Synth-IoU improvement did NOT
translate to a bigger real ADD gain -> the SAME "proxy != eval-target" lesson (synth mask IoU doesn't
predict real-mask placement on the foreshortened frames that matter).
- **CONCLUSION: render-compare is VALIDATED (+0.108 oracle ceiling, robust to mask DEGRADATION) but a
  SELF-CONTAINED learned mask caps the deployable gain at ~+0.01** — circularity: the foreshortened hard
  frames (where render-compare must help) have a wrong solver pose -> wrong pseudo-mask -> the mask head
  never learns them well. The full ceiling needs a CORRECTLY-PLACED real mask = a proper SEGMENTER
  (SAM2/CtRNet). That install (SAM2 won't pip on torch 2.10; CtRNet vendored in HPE, needs weights) is the
  gating next step to realize render-compare's +0.108 on realsense.
- Infra note: GPU 05f84104 was grabbed mid-run by an external `ollama runner` (~21GB); ran the retry at
  batch 3 in the remaining 2.5GB.

### visual vs collision mesh for render-compare (oracle ceiling, realsense 400 @224px): COLLISION WINS
| renderer mesh | oracle Δ |
|---|---|
| collision (solid, low-poly) | +0.108 |
| visual stride-downsampled | +0.064 |
| visual uniform-downsampled (cap600) | +0.055 |
- Hypothesis "visual matches the image -> better" REFUTED for the RENDERER. render-compare's depth cue is the
  overall AREA; collision meshes are SOLID/convex -> a cleaner, more depth-sensitive silhouette. Visual
  meshes' fine detail/concavities make the splatted silhouette less solid -> weaker depth signal. KEEP collision.
- So the deployable bottleneck is NOT the renderer mesh; it's the learned real-mask quality (IoU 0.52 vs
  oracle, 17% badly mis-placed). Self-contained mask (synth / dense / self-train / visual) caps at IoU ~0.52
  -> render-compare deployable ~+0.01. The genuine unlock = a correctly-placed real mask from a SEGMENTER
  (SAM2 won't pip on torch 2.10; CtRNet vendored in HPE/lib/models/ctrnet needs weights). That install is the
  gating next step to realize render-compare's validated +0.108 on realsense.

### render-compare with SAM (ViT-B) real masks: FAILED (mask-render mismatch / prompt quality)
Installed segment-anything + sam_vit_b (network available). Prompt SAM with detected keypoints -> real robot
mask -> render-compare target. realsense smoke: SAM-vs-render IoU 0.21 (collision) / 0.36 (visual), render-
compare -0.53 / -0.64 (DIVERGES).
- SAM segments the TRUE (visual) robot; matching it to the collision (bulky) refine-render is shape-
  inconsistent -> pose pushed to wrong depth. Visual-render restores some consistency (IoU 0.21->0.36) but
  still diverges -> SAM's keypoint-prompted mask is itself too rough/partial vs the GT-pose render, OR a
  residual render-vs-image alignment offset. Either way SAM out-of-the-box doesn't deliver.
- **NET: render-compare is VALIDATED (+0.108 self-consistent oracle) but NO real mask source delivers the
  deployable gain yet** — learned mask (IoU 0.52, consistent-but-imperfect) -> +0.01; SAM (accurate-but-
  shape-inconsistent) -> diverges. Cracking it needs focused mask work (SAM prompt/box tuning + visual-render
  consistency + an area-only or robust loss, or a robot-specific segmenter like CtRNet). Clean next-session task.

### SAM box-prompt + render-consistency: STILL FAILS -> the renderer-fidelity wall (definitive)
SAM box-prompt (kp bbox+margin): collision IoU 0.235 / -0.45, visual IoU 0.36 / -0.64. Overlay viz confirms
my GT-pose render OVERLAYS the real robot (NO alignment bug) — but my splat render (collision=bulky,
visual=approx) does not match SAM's TRUE robot pixels (IoU caps 0.24-0.36) -> matching a true mask to an
approximate render pushes the pose wrong -> diverges.
- **DEFINITIVE: render-compare only works when the target mask comes from the SAME (approximate) render
  model** — oracle (self-consistent +0.108) or a render-trained learned mask (consistent but imperfect,
  IoU 0.52 -> +0.01). A TRUE segmenter (SAM) is fundamentally inconsistent with the approximate splat render
  -> can't be used without a HIGH-FIDELITY differentiable renderer (exact visual meshes + real rasterizer =
  PyTorch3D, which won't install on torch 2.10/cu128). 
- **Render-compare is VALIDATED as a lever (+0.108 depth/scale recovery) but NOT deployable with current
  tooling**: the self-contained learned mask caps the gain at ~+0.01, and the SAM route needs PyTorch3D.
  Clean next-session task: get PyTorch3D (or nvdiffrast) working -> exact-mesh render -> SAM/CtRNet true mask
  -> realize the +0.108 (realsense 0.745 -> ~0.85). Until then, deployable best stays mean 0.759.
