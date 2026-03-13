# Compositional Dynamic Depth

**A symmetry-theoretic foundation for dynamic depth in neural networks, with experimental tools.**

This repository accompanies the technical note *Compositional Dynamic Depth in Neural Networks: A Symmetry-Theoretic Foundation and Experimental Programme* (Ruffini, 2026). It provides:

1. A **2D articulated cat** with a fully controlled 7-level Lie-group compositional hierarchy
2. A **symmetry-gated residual autoencoder** where each layer has a learnable gate α ∈ [0,1]
3. A **training and evaluation pipeline** that tests whether gate patterns reflect the compositional complexity of the data-generating process

## The idea

Standard deep networks apply every layer to every input. We argue that depth should be *dynamic and input-dependent*, controlled by how compositionally complex the input is. A static scene under fixed lighting needs fewer layers than an articulated body under changing viewpoint, pose, and appearance.

The Lie-pseudogroup framework of compositional symmetry (Ruffini 2025, [arXiv:2510.10586](https://arxiv.org/abs/2510.10586)) predicts *which* layers should activate for a given input: layers aligned with inactive symmetry scales should carry near-zero residuals.

This repo tests that prediction using a controlled generative model (the jointed cat) where we know the ground-truth hierarchy.

## The hierarchy

The cat's configuration space decomposes into 7 levels:

| Level | Generators | Parameters |
|-------|-----------|------------|
| 1 | Camera SE(2) × R⁺ | rotation, translation, scale |
| 2 | Root body SE(2) | body position and orientation |
| 3 | Spine SO(2)³ | 3 spine joints |
| 4 | Limbs SO(2)⁸ | 2 joints × 4 legs |
| 5 | Head & tail SO(2)⁴ | head pan/tilt, 2 tail joints |
| 6 | Appearance R⁶ | colour, thickness, stripes |
| 7 | Background R³ | gradient, colour, intensity |

Each "condition" activates a subset of levels. The prediction is: more active levels → higher effective depth D_eff = Σ α_ℓ.

## Files

```
compositional_cat.py      — Jointed-cat model v1 (legacy)
compositional_cat_v2.py   — Jointed-cat model v2: bigger cat, wider parameter ranges
gated_resnet.py           — Symmetry-gated autoencoder v3:
                              • 1×1 stem and downsampling (forces spatial processing
                                through gated blocks)
                              • Gate uses [AvgPool, StdPool] for complexity-aware gating
                              • Complexity-scaled penalty (simpler inputs → stronger λ)
train_and_evaluate.py     — Training, evaluation, and plotting v2:
                              • Mixed-condition training (all 8 conditions, balanced)
                              • Progressive gate penalty warmup
                              • Passes per-sample complexity to loss
dynamic_depth_TN.tex      — LaTeX source for the technical note
dynamic_depth_TN.pdf      — Compiled technical note
```

## Architecture (v3)

Three key design choices make the gates meaningful:

1. **Minimal non-gated capacity.** The stem is a 1×1 conv (channel projection only) and downsampling is AvgPool + 1×1 conv. All 3×3 spatial feature extraction lives inside gated residual blocks. This ensures that closing a gate removes actual processing power.

2. **Richer gate conditioning.** Each gate sees `[AvgPool(h), StdPool(h)]` — the spatial standard deviation is a direct proxy for geometric complexity. A static cat has low spatial variance; an articulated cat under camera rotation has high variance.

3. **Complexity-scaled penalty.** During mixed-condition training, each sample's gate penalty is scaled by `(max_levels - n_levels + 1) / max_levels`. Simpler inputs get a stronger push to close gates; complex inputs get more freedom. Combined with a progressive warmup (λ ramps from 0 to λ_max over the first N epochs), this lets the model learn good features before selectively pruning.

## Quickstart

### Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision pillow matplotlib numpy tqdm
```

### Generate a sample grid

```bash
python compositional_cat_v2.py --mode grid --img_size 128 --output_dir dataset
```

This produces `dataset/sample_grid_v2.png` — rows are conditions (Static through Everything), columns are random samples.

### Run the full pipeline

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

On macOS or without a GPU, use `--device cpu` (training will be slower).

This trains an 8-layer gated autoencoder on **all conditions simultaneously** (8 × 2000 = 16,000 images), then evaluates gate patterns on each condition. Outputs:

- `results/gated_autoencoder.pt` — trained model
- `results/gate_analysis.json` — per-condition gate statistics
- `results/training_history.json` — epoch-level training metrics
- `results/dynamic_depth_results.png` — the four prediction plots
- `results/training_curves.png` — loss, D_eff, and λ schedule

## Run online (Colab, Kaggle)

If your machine is limited, run the pipeline in the cloud with a free GPU:

- **Kaggle:** Create a new Notebook, then in the **right sidebar → Settings** turn **Accelerator** to **GPU** and **Internet** to **On**. Do **not** run `pip install torch` on Kaggle — it can replace the pre-installed GPU PyTorch with a CPU-only build. Example:
  ```python
  !git clone https://github.com/giulioruffini/Compositional-NNs.git
  %cd Compositional-NNs
  !pip install -q tqdm
  !python train_and_evaluate.py \
      --n_train_per_cond 2000 --n_eval 500 --n_epochs 60 \
      --gate_warmup 15 --img_size 128 --batch_size 64 \
      --base_channels 32 --latent_dim 64 --gate_penalty 0.01 \
      --device cuda --output_dir results
  ```

- **Google Colab:** **Runtime → Change runtime type → GPU** (T4), then clone and run as above.

### Generate a dataset to disk

```bash
python compositional_cat_v2.py --mode generate --condition FullPose --n_samples 10000 --img_size 128
python compositional_cat_v2.py --mode all --n_samples 5000   # all conditions
```

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--gate_penalty` | 0.05 | Max gate penalty (ramped during warmup). Try {0.005, 0.01, 0.02, 0.05}. |
| `--gate_warmup` | 10 | Epochs to ramp λ from 0 to max. Lets the model learn features first. |
| `--gate_init_bias` | 2.0 | Gates start at σ(2.0) ≈ 0.88 (mostly open). Keep at 2.0. |
| `--n_train_per_cond` | 1000 | Samples per condition. Total training = 8 × this. Use ≥2000 for real runs. |
| `--img_size` | 64 | Use 128 for real runs. Below 64, gated blocks may not be needed. |
| `--n_stages` | 4 | Number of encoder stages (each doubles channels). |
| `--n_blocks` | 2 | Residual blocks per stage. Total gated layers = n_stages × n_blocks. |
| `--latent_dim` | 64 | Bottleneck dimension. Use 64–128 for img_size=128. |

## What to look for

The key output is `dynamic_depth_results.png` with four panels:

1. **Gate heatmap** — should show a "staircase": more active levels → more layers engaged
2. **D_eff vs complexity** — should be monotonically increasing (at least for geometric levels 0–5)
3. **Per-layer profiles** — simple conditions should have lower gate values, especially in early layers
4. **Reconstruction error** — more complex conditions are harder to reconstruct

Note: appearance (level 6) and background (level 7) are global colour changes that don't require spatial convolution depth. They may not follow the monotonic trend of the geometric levels — this is actually consistent with the theory.

## Changelog

### v3 (current)
- **Mixed-condition training** — model sees all complexity levels, not just "Everything"
- **Minimal non-gated encoder** — 1×1 stem and downsamples; all 3×3 processing is gated
- **AvgPool + StdPool gating** — spatial variance gives gates a complexity signal
- **Complexity-scaled penalty** — simpler inputs get stronger gate pressure
- **Progressive warmup** — λ ramps from 0 to max over first N epochs

### v2
- Bigger cat (WORLD_SCALE 0.55 vs 0.35), wider parameter ranges
- Thicker limbs and body for more pixel coverage
- Vectorised background rendering (10× faster)

### v1
- Initial implementation with single-condition training on "Everything"

## References

- Ruffini, G. (2025). *Compositional Symmetry as Compression: Lie-Pseudogroup Structure in Algorithmic Agents.* [arXiv:2510.10586](https://arxiv.org/abs/2510.10586)
- Ruffini, G., Castaldo, F., Vohryzek, J. (2025). *Structured Dynamics in the Algorithmic Agent.* Entropy, 27(1):90. [doi:10.3390/e27010090](https://doi.org/10.3390/e27010090)
