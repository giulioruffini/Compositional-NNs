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

All Python entry points live in `scripts/` (run commands from that directory). Job metadata is stored under `jobs/`:

```
scripts/
  compositional_cat.py    — Jointed-cat model + dataset generator
  gated_resnet.py         — Symmetry-gated autoencoder architecture
  train_and_evaluate.py   — Training, evaluation, and plotting
  list_jobs.py            — List runs from jobs registry (--last N, --status)
jobs/
  jobs_registry.json      — Central log of all runs (created on first run)
ARCHITECTURE.md           — Pipeline, data model, and four predictions
dynamic_depth_TN.tex      — LaTeX source for the technical note (if present)
dynamic_depth_TN.pdf      — Compiled technical note (if present)
```

## Run online (Colab, Kaggle)

If your machine is limited, run the pipeline in the cloud with a free GPU:

- **Google Colab (recommended):**
  - **If the repo is public:** Open [this link](https://colab.research.google.com/github/giulioruffini/Compositional-NNs/blob/main/scripts/run_on_colab.ipynb) to open the notebook in Colab.
  - **If the repo is private (or the link returns 404):** Colab cannot open private repos from GitHub. Instead: **File → Upload notebook** in Colab and upload `scripts/run_on_colab.ipynb` from your local clone. In the first code cell, comment the default clone line and uncomment the one that uses `https://YOUR_GITHUB_TOKEN@github.com/...` (create a token at [github.com/settings/tokens](https://github.com/settings/tokens)).
  - In Colab: **Runtime → Change runtime type → GPU** (T4), then run the cells. The notebook clones the repo, installs deps, and runs a quick test; you can run a full training and download the results.

- **Kaggle:** Create a new Notebook, then in the **right sidebar → Settings** turn **Accelerator** to **GPU** and **Internet** to **On**. Use **`--device cuda`**. Do **not** run `pip install torch` on Kaggle—it can replace the pre-installed GPU PyTorch with a CPU-only build. Only install packages Kaggle may be missing (e.g. `tqdm`). Example:
  ```python
  !git clone https://github.com/giulioruffini/Compositional-NNs.git
  %cd Compositional-NNs
  !pip install -q tqdm
  !cd scripts && python train_and_evaluate.py --n_train 5000 --n_eval 500 --n_epochs 20 --img_size 128 --batch_size 64 --device cuda --num_workers 4 --output_dir ../results_kaggle
  ```
  Use **`--num_workers 4`** so the DataLoader prefetches batches in parallel; otherwise the GPU sits idle waiting for on-the-fly image generation.
  To confirm the GPU is used, run `import torch; print(torch.cuda.is_available())` — it should print `True`. Free GPU (P100) about 30 h/week. If your repo is private, use a token in the URL: `!git clone https://YOUR_TOKEN@github.com/giulioruffini/Compositional-NNs.git`

- **Paid GPU (long runs):** [Lambda Labs](https://lambdalabs.com), [RunPod](https://runpod.io), or [Vast.ai](https://vast.ai) offer cheap per-hour GPU instances. Clone the repo, install dependencies, and run the same commands as locally with `--device cuda`.

## Quickstart

### Setup

From the repo root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision pillow matplotlib numpy tqdm
```

### Generate a sample grid

Run from `scripts/`; create the output directory first if it doesn’t exist:

```bash
mkdir -p dataset
cd scripts
python compositional_cat.py --mode grid --img_size 128 --output_dir ../dataset
```

This produces `dataset/sample_grid.png` — rows are conditions (Static through Everything), columns are random samples.

### Run the full pipeline

From `scripts/` (use `--device cpu` on machines without CUDA):

```bash
cd scripts
python train_and_evaluate.py \
  --n_train 20000 \
  --n_eval 1000 \
  --n_epochs 50 \
  --img_size 128 \
  --batch_size 64 \
  --base_channels 64 \
  --n_stages 4 \
  --n_blocks 3 \
  --latent_dim 128 \
  --gate_penalty 0.01 \
  --lr 1e-3 \
  --device cuda \
  --output_dir ../results
```

On macOS or without a GPU, use `--device cpu` (training will be slower).

This trains a 12-layer gated autoencoder on the "Everything" condition, then evaluates gate patterns on all conditions separately. Outputs:

- `results/gated_autoencoder.pt` — trained model
- `results/gate_analysis.json` — per-condition gate statistics
- `results/dynamic_depth_results.png` — the four prediction plots

For many runs, job metadata is logged in `jobs/jobs_registry.json` and in each run’s `output_dir` as `job_manifest.json` (parameters, timings, status, summary). See [ARCHITECTURE.md](ARCHITECTURE.md) for the pipeline and design.

### Quick verification run (fewer samples, fewer epochs)

To check the pipeline end-to-end without a long training run:

```bash
cd scripts
python train_and_evaluate.py --n_train 2000 --n_eval 200 --n_epochs 3 --img_size 64 --batch_size 32 --base_channels 32 --n_stages 3 --n_blocks 2 --latent_dim 64 --gate_penalty 0.01 --device cpu --output_dir ../results_quick
```

### Overnight run (~8 h)

To size a run for about 8 hours:

1. **Calibrate** with the recommended preset (one epoch only) to get a suggested `n_epochs`:

```bash
cd scripts
python train_and_evaluate.py --calibrate \
  --n_train 12000 --n_eval 1000 --n_epochs 40 \
  --img_size 128 --batch_size 64 \
  --base_channels 64 --n_stages 4 --n_blocks 3 --latent_dim 128 \
  --gate_penalty 0.01 --lr 1e-3 --device cpu \
  --output_dir ../results_overnight_calibrate
```

The script prints e.g. `Suggested n_epochs for ~8 h run: 42`. Use that value in step 2.

2. **Run the full training** with the suggested `n_epochs` (replace `42` with the printed value):

```bash
python train_and_evaluate.py \
  --n_train 12000 --n_eval 1000 --n_epochs 42 \
  --img_size 128 --batch_size 64 \
  --base_channels 64 --n_stages 4 --n_blocks 3 --latent_dim 128 \
  --gate_penalty 0.01 --lr 1e-3 --device cpu \
  --output_dir ../results_overnight_8h
```

### List jobs

To list recent runs and their status (from `scripts/`):

```bash
python list_jobs.py --registry ../jobs/jobs_registry.json --last 10
python list_jobs.py --status success
```

### Generate a dataset to disk

From `scripts/`:

```bash
python compositional_cat.py --mode generate --condition FullPose --n_samples 10000 --img_size 128 --output_dir ../dataset
python compositional_cat.py --mode all --n_samples 5000 --output_dir ../dataset   # all conditions
```

## Key parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `--gate_penalty` | 0.01 | **Most important.** Too low → all gates stay open. Too high → reconstruction collapses. Sweep {0.005, 0.01, 0.02, 0.03}. |
| `--gate_init_bias` | 2.0 | Gates start at σ(2.0) ≈ 0.88 (mostly open). Keep at 2.0. |
| `--img_size` | 64 | Use 128 for real runs. At 32, pose changes are barely visible. |
| `--n_stages` | 4 | Number of encoder stages (each doubles channels). |
| `--n_blocks` | 2 | Residual blocks per stage. Total gated layers = n_stages × n_blocks. |
| `--latent_dim` | 64 | Bottleneck dimension. Use 128 for img_size=128. |

## What to look for

The key output is `dynamic_depth_results.png` with four panels:

1. **Gate heatmap** — should show a "staircase": more active levels → more layers engaged
2. **D_eff vs complexity** — should be monotonically increasing
3. **Per-layer profiles** — simple conditions should have gates near zero at deeper layers
4. **Reconstruction error** — more complex conditions are harder to reconstruct

## References

- Ruffini, G. (2025). *Compositional Symmetry as Compression: Lie-Pseudogroup Structure in Algorithmic Agents.* [arXiv:2510.10586](https://arxiv.org/abs/2510.10586)
- Ruffini, G., Castaldo, F., Vohryzek, J. (2025). *Structured Dynamics in the Algorithmic Agent.* Entropy, 27(1):90. [doi:10.3390/e27010090](https://doi.org/10.3390/e27010090)
