# Stage 1 (CSI → 3D skeleton) — cross-environment results

Goal: WiFi CSI → person-centric 3D skeleton, generalizing to **unseen rooms** (for
downstream action captioning). Person-centric GT (pelvis-centered x,y; z=height
above floor), geometry conditioning (rel_rx) ON.

## Setup
- Train: Scene1 + Scene2 + Scene3 (28 subjects, multiple layouts), 60 inst/user subset → 1,680 (val 168 held out)
- Test: Scene4, Scene5 = **unseen rooms** (never in training)
- Model: PerceptAlign posenet (4 layers), 60 epochs, AdamW + ReduceLROnPlateau, bs=4×accum4, AMP
- Eval: raw L2 MPJPE (no Procrustes / root-align)

## Results
| split | MPJPE | PCK@20 | PCK@50 |
|---|---|---|---|
| val (Scene1+2+3, in-distribution) | **72.2 mm** | 9.6% | 42% |
| test Scene5 (unseen room) | **107.8 mm** | 3.2% | 17.7% |
| test Scene4 (unseen room) | **179.7 mm** | 1.8% | 7.7% |

## Reading
- Clear **cross-environment degradation**: 72mm in-distribution → 108mm (Scene5) → 180mm (Scene4).
  This is exactly the "coordinate/environment overfitting" the paper targets.
- Scene4 degrades far more than Scene5 — likely because the **test room's tx/rx geometry are
  AUTO floor-plane placeholders** (not measured), so the rel_rx condition can mislead on a very
  different layout. The geometry on/off ablation will test this directly.
- First pass on a **60/user subset** with **self-labeled GT** — a baseline, not a ceiling.

## Next levers
1. **Geometry ablation** (use_geometry true vs false): does conditioning help or hurt cross-room?
   (configs ready; ~2.5h to retrain the off variant.)
2. **More training data** (raise inst/user; full sets are labeled) → lower the whole curve.
3. **Refine test-scene tx/rx** (replace placeholders with measured antenna positions).
4. For **captioning (Stage 2)**: gross-motion fidelity matters more than mm-MPJPE; evaluate
   whether predicted skeletons are good enough to describe actions even at these errors.
