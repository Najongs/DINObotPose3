# Hypothesis test — where does the error actually live?

Final deployable pipeline (stage1 detector → mlp angle head → rot R_init → kinematic solve),
**1200 real frames** (realsense + azure + orb), `Eval/viz_hypothesis.py`.

Overall: fail(>100mm) = 8%, median ADD 26 mm.

## H2 — "2D keypoint detection is good" ✅ CONFIRMED (except the base)
| kp | median px @512 | PCK@5 |
|---|---|---|
| **link0 (base)** | **6.8** | **0.45** ⚠️ |
| link2 | 1.7 | 0.95 |
| link3 | 3.5 | 0.79 |
| link4 | 3.3 | 0.85 |
| link6 | 2.7 | 0.82 |
| link7 | 3.1 | 0.79 |
| hand | 2.7 | 0.77 |

Overall median **3.1 px @512 (≈1.3 px @224)** — detection is excellent **for every keypoint
except link0**, the robot base. The base is the *only* weak 2D detection (6.8px, PCK 0.45).

## H1 — "angle accuracy is low" ✅ CONFIRMED but concentrated in J0/J2
| joint | MAE |
|---|---|
| **J0 (base yaw)** | **24.2°** ⚠️ |
| J1 | 7.1° |
| **J2** | **12.5°** |
| J3 | 3.7° |
| J4 | 8.8° |
| J5 | 4.2° |

Overall 10.1°, but it is **dominated by J0 (base yaw)**. Wrist joints (J3, J5) are already fine (≈4°).
The angle problem is **not everywhere — it is the base**.

## H3 — "occlusion breaks FK → outliers" ✅ drives the TAIL (not the bulk)
- Spearman ADD ~ #off-frame kp = **+0.16** (weak linear) — but the binary effect is big:
  - fully-visible frames: median ADD **25 mm**, fail-rate **7%**
  - has ≥1 occluded kp:   median ADD **39 mm**, fail-rate **25%** (3.5× the failures)
- Stronger correlates are ADD~2D-err **+0.50** and ADD~angle **+0.53** — and both trace back to the base.
- So occlusion is a **tail trigger** (the conf<0.05 gate already removes off-frame kp from PnP; the
  residual damage is on the frames where the *base* is occluded/foreshortened, which is exactly the
  under-determined-J0 case).

## The unified read (the linchpin)
**link0 (base keypoint) → J0 (base yaw) is the single linchpin.** It is simultaneously the worst-detected
2D point (6.8px / PCK 0.45) and the worst angle (24°), and occluding it is what produces the ADD outliers.
Everything downstream of a good base is already solved (other kp 1.7–3.5px, wrist angles 4°). This matches
`realsense-failures-are-foreshortened-j0`: the base is large, low-texture, often foreshortened/occluded, so
its yaw is geometrically under-determined from a single view.

**Implication for "how to approach occlusion / outliers":** don't chase generic angle accuracy — the lever
is recovering the **base pose** when the base keypoint is weak. Two validated directions already on file:
1. **render-and-compare** (`render-compare-validated`): the silhouette area constraint re-determines the
   base depth/yaw even when the base keypoint is unreliable (+0.108 oracle). Blocked only on a clean mask.
2. **rotation head R_init** (`rotation-head-fixes-realsense`): learn the camera-R/base orientation from
   appearance instead of from the fragile base keypoint (+0.117 realsense). Already in the pipeline.

## DECISIVE follow-up — base error decomposition (`Eval/base_oracle_probe.py`, 800 real frames)
Substituting GT 2D into the solver to separate detection-error from geometric under-determination:

| variant | J0 angle | ADD median | fail% |
|---|---|---|---|
| baseline (detected 2D) | 27.8° | 26 mm | 7% |
| base-oracle (link0/link2 = GT 2D) | 27.5° | 20 mm | 9% |
| **all-oracle (ALL kp = GT 2D)** | **28.5°** | **4 mm** | 7% |

base detector 2D err: median **5.5px** (good) but mean **51.5px** (heavy off-frame/occluded outlier tail).

**Two separate reads — corrects the naive "fix base detection → fix angle":**
- **J0 ANGLE is NOT a detection problem.** Even all-oracle 2D leaves J0 at 28° → base-yaw is **gauge-coupled
  to camera-R, geometrically under-determined** from one view. And the **DREAM/RoboPEPP benchmark does not
  score joint angles — only ADD** (keypoint 3D positions); a 28° J0 with a compensating R still gives correct
  ADD. So chasing base/J0 angle accuracy is a dead end for the score.
- **ADD IS bottlenecked by 2D PRECISION/outliers.** all-oracle collapses ADD 26→4mm. But base-only oracle
  only reaches 20mm — the big gain needs **every** kp sharp on the hard (foreshortened/occluded) tail frames,
  not the base alone. The base's contribution is its **outlier tail** (mean 51px), = the H3 occlusion tail.

**Reframed lever:** not "base appearance sim2real" and not "angle accuracy" — but **robust base/keypoint
POSITION on the occluded-foreshortened tail.** Same two validated tools below (render-compare, rot-head).

## Objection answered — "bad angles make the render-compare mesh useless?" (`Eval/silhouette_gauge_probe.py`)
Render the silhouette at the deployed BAD-angle solved pose vs the GT pose (200 realsense frames, 224px):
- J0 angle error median **40.9°**, yet IoU(render@solved , render@GT) median **0.62**.
- **IoU ~ J0-error correlation = −0.07 (≈0)**: frames with J0 off 10–25° and 25–200° have the SAME silhouette
  IoU (0.616 vs 0.622). The J0 error is **gauge** — it cancels with camera-R, so the camera-frame silhouette
  is unchanged. A 40°-wrong-J0 mesh renders to the same place as the GT mesh.
- The real residual is **depth/scale**: |z_solved/z_GT| median 0.957, 12% of frames >10% off — exactly what
  render-compare (area ∝ 1/z²) fixes, and it is orthogonal to the angle decomposition.
- Confirms the validated +0.108 was measured FROM this bad-angle init (baseline 0.691). Bad angles do NOT
  block render-compare; the blocker remains a clean render-consistent real MASK.

## Files
- `panel_2d_vs_angle.png` — H2 (2D good, base bad) vs H1 (angle bad, J0 dominant)
- `panel_outliers.png`    — H3 (ADD vs 2D-err scatter; ADD-by-#occluded boxplot)
- `montage_examples.png`  — 3 columns: good2D+good pose | good2D+BAD angle | occluded→FK outlier
  (GREEN=GT 2D+skeleton, YELLOW=detected 2D, RED=solved-FK reprojection)
