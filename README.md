# Compositional Dynamic Depth

**A symmetry-theoretic foundation for dynamic depth in neural networks, with experimental tools.**

This repository accompanies the technical note *Compositional Dynamic Depth in Neural Networks: A Symmetry-Theoretic Foundation and Experimental Programme* (Ruffini, 2026). It provides:

1. A **2D articulated cat renderer** (v6) with a fully controlled 5-level Lie-group compositional hierarchy
2. An **interactive GUI** with 32 sliders to explore the compositional parameter space
3. A **symmetry-gated residual autoencoder** (v6) where each residual block has a learnable gate
4. A **training and evaluation pipeline** that tests whether gate patterns reflect compositional complexity

## The Idea

Standard deep networks apply every layer to every input. We argue that depth should be *dynamic and input-dependent*, controlled by how compositionally complex the input is. A static scene needs fewer layers than an articulated body under changing viewpoint, pose, and appearance.

The Lie-pseudogroup framework of compositional symmetry (Ruffini 2025, [arXiv:2510.10586](https://arxiv.org/abs/2510.10586)) predicts *which* layers should activate for a given input: layers aligned with inactive symmetry scales should carry near-zero residuals.

This repo tests that prediction using a controlled generative model (the jointed cat) where we know the ground-truth hierarchy.

## The Compositional Hierarchy

The cat's configuration space decomposes into 5 levels, forming a Lie-group flag `G = H₀ ⊃ H₁ ⊃ ... ⊃ H₅`:

| Level | Name | Group | Params | Description |
|-------|------|-------|--------|-------------|
| 0 | Static | Identity | 0 | One fixed cat (reference image) |
| 1 | Pose | SO(1)¹⁶ | 16 | Spine (3), limbs (8), head pan/tilt/roll (3), tail (2) |
| 2 | Appearance | R⁶ | 6 | Body colour (HSV), limb thickness, eye size, stripes |
| 3 | Placement | R² × SO(3) | 5 | Position (x, y) + full 3D rotation (yaw, elevation, roll) |
| 4 | Camera | SE(2) × R⁺ | 4 | Observer rotation, translation (x, y), zoom |
| 5 | Background | R¹ | 1 | Uniform greyscale intensity |

**Total: 32 parameters** across 5 levels.

The generative story: *"pose the cat → dress it up → place it in the scene → point the camera → choose the backdrop."*

### Experimental Conditions

Six conditions activate progressively more levels:

| Condition | Active Levels | Compositional Depth |
|-----------|--------------|---------------------|
| `Static` | — | 0 (fixed image) |
| `PoseOnly` | 1 | 1 |
| `PoseAppearance` | 1, 2 | 2 |
| `PosAppPlace` | 1, 2, 3 | 3 |
| `PosAppPlaceCam` | 1, 2, 3, 4 | 4 |
| `Everything` | 1, 2, 3, 4, 5 | 5 |

**Prediction:** a gated autoencoder trained on condition with depth *d* should learn to use ≈ *d* active gated blocks.

## Files

```
compositional_cat_v2.py   — Renderer v6: articulated cat with 5-level hierarchy
cat_gui.py                — Interactive matplotlib GUI with 32 sliders
gated_resnet.py           — Gated ResNet v6 autoencoder with learned bypass constants
train_and_evaluate.py     — Per-condition training pipeline + gate analysis
```

## Quick Start

### Prerequisites

```bash
pip install numpy pillow matplotlib scipy  # renderer + GUI
pip install torch torchvision tqdm         # neural network training
```

### Generate a Sample Grid

```bash
python compositional_cat_v2.py --mode grid --img_size 192
```

Produces `dataset/sample_grid_v2.png` — 6 rows (one per condition) × 8 random samples.

### Interactive GUI

```bash
python cat_gui.py                          # default 256×256
python cat_gui.py --img_size 512           # higher resolution
python cat_gui.py --condition Everything   # start with random params
```

The GUI groups 32 sliders by hierarchy level with colour coding:

- **Red** — Level 1 Pose: spine joints, leg joints (upper/lower × 4), head pan/tilt/roll, tail joints
- **Amber** — Level 2 Appearance: body hue/saturation/brightness, limb thickness, eye size, stripe intensity
- **Green** — Level 3 Placement: position (x, y), yaw (full 360°), elevation, roll
- **Blue** — Level 4 Camera: rotation, translation (x, y), scale
- **Grey** — Level 5 Background: grey level

Buttons: **Reset** | **Random All** | **Rand L1** through **Rand L5** (randomise individual levels).

### Generate Training Datasets

```bash
# Single condition (5000 images at 128×128)
python compositional_cat_v2.py --mode generate --condition Everything --n_samples 5000 --img_size 128

# All conditions at once
python compositional_cat_v2.py --mode all --n_samples 5000 --img_size 128
```

Output structure:
```
dataset/
  Static/
    images/000000.png ... 004999.png
    metadata.jsonl
  PoseOnly/
    ...
  Everything/
    ...
```

Each `metadata.jsonl` line contains: `index`, `condition`, `active_levels`, `params` (all 32 values), `image` filename.

### Train the Gated Autoencoder

```bash
python train_and_evaluate.py \
  --n_train_per_cond 2000 \
  --n_eval 500 \
  --n_epochs 60 \
  --gate_warmup 15 \
  --img_size 128 \
  --batch_size 64 \
  --base_channels 32 \
  --n_stages 4 \
  --n_blocks 2 \
  --latent_dim 64 \
  --gate_penalty 0.01 \
  --device cuda \
  --output_dir results
```

On macOS or without GPU, use `--device mps` or `--device cpu`.

## Renderer Features (v6)

The v6 renderer includes several physically-motivated features:

**Kinematics:**
- Forward kinematics (Product of Exponentials) for the articulated skeleton: spine → shoulders → head/legs, with separate chains for limbs and tail.
- Ball-on-rod head model: `head_pan`/`head_tilt` move the neck (FK chain); `head_roll` rotates the face around the neck axis, foreshortening lateral offsets by cos(roll).

**3D Projection:**
- Pseudo-3D body rotation via `Rx(elevation) · Ry(roll)` with orthographic projection. The cross-term `x' = x·cos(r) + y·sin(e)·sin(r)` produces correct foreshortening.
- 3D sphere eye projection: eyes placed at angular positions on the head sphere with intrinsic face tilt toward the viewer. Per-eye visibility from the z-component of the surface normal. Size and shape (oriented ellipses) scale with visibility. Head roll creates natural left/right asymmetry.

**Rendering quality:**
- 2× supersampling anti-aliasing (render at double resolution, LANCZOS downsample).
- Drop shadow under paw positions (grounds the cat in the scene).
- Thin body and head outlines for definition against similar-hued backgrounds.
- Cubic-spline interpolated tail for smooth curvature.
- Depth-ordered limb rendering: near-side vs far-side legs determined by root_angle, drawn at different depths with distinct shading.
- Paw toe details (small dark lines).
- Off-screen rejection sampling: `sample_params()` rejects configs where the cat falls outside the frame.

**Performance:** ~280 images/sec at 128×128 with 2× AA on CPU.

## Neural Network Architecture (Gated ResNet v6)

The autoencoder (`gated_resnet.py`) uses:

- Encoder–decoder with skip connections across matching resolution stages
- Each residual block has a **learned multiplicative gate** (sigmoid) that can dynamically bypass the block
- Gate ≈ 0 → block bypassed (effectively reducing network depth)
- Gate ≈ 1 → block active (full depth utilised)
- A **learned bypass constant** prevents gate collapse during early training
- Gate conditioning via `[AvgPool, StdPool]` — spatial variance serves as a complexity signal

## Key Training Parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--gate_penalty` | 0.01 | Max L1 penalty on gate values. Try {0.005, 0.01, 0.02, 0.05}. |
| `--gate_warmup` | 15 | Epochs to ramp λ from 0 → max. Lets features form before pruning. |
| `--n_train_per_cond` | 2000 | Images per condition. Total = 6 × this. Use ≥2000 for real runs. |
| `--img_size` | 128 | Training resolution. Below 64, gated blocks may not be needed. |
| `--n_stages` | 4 | Encoder/decoder stages (each doubles channels). |
| `--n_blocks` | 2 | Residual blocks per stage. Total gates = n_stages × n_blocks. |
| `--latent_dim` | 64 | Bottleneck dimension. Use 64–128 for img_size=128. |

## What to Look For

The key output is `dynamic_depth_results.png` with diagnostic plots:

1. **Gate heatmap** — should show a "staircase": more active levels → more engaged layers
2. **D_eff vs complexity** — effective depth should increase monotonically (especially for geometric levels)
3. **Per-layer gate profiles** — simple conditions should have lower gate values
4. **Reconstruction error** — more complex conditions are harder to reconstruct

## Cloud Execution

**Kaggle:** Create a Notebook with **GPU** accelerator and **Internet** on. Do *not* `pip install torch` — use the pre-installed GPU build:
```python
!git clone https://github.com/giulioruffini/Compositional-NNs.git
%cd Compositional-NNs
!pip install -q tqdm scipy
!python train_and_evaluate.py \
    --n_train_per_cond 2000 --n_eval 500 --n_epochs 60 \
    --gate_warmup 15 --img_size 128 --batch_size 64 \
    --base_channels 32 --latent_dim 64 --gate_penalty 0.01 \
    --device cuda --output_dir results
```

**Google Colab:** Set **Runtime → Change runtime type → GPU (T4)**, then clone and run as above.

## References

- Ruffini, G. (2025). *Compositional Symmetry as Compression: Lie-Pseudogroup Structure in Algorithmic Agents.* [arXiv:2510.10586](https://arxiv.org/abs/2510.10586)
- Ruffini, G., Castaldo, F., Vohryzek, J. (2025). *Structured Dynamics in the Algorithmic Agent.* Entropy, 27(1):90. [doi:10.3390/e27010090](https://doi.org/10.3390/e27010090)
