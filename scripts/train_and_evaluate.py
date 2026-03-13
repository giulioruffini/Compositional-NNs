"""
Training and Evaluation for Compositional Dynamic Depth
========================================================

Pipeline steps:
  1. Create on-the-fly dataset for the "Everything" condition (all 7 levels).
  2. Build the symmetry-gated autoencoder (encoder + decoder).
  3. Train on reconstruction with gate penalty λ Σ α_ℓ.
  4. Evaluate gate patterns on each condition (Static, CameraOnly, ..., Everything).
  5. Plot the four predictions (gate heatmap, D_eff vs complexity, per-layer profiles, recon error).
  6. Write job manifest and update central jobs registry.

See ARCHITECTURE.md for the overall design. Job metadata: jobs/jobs_registry.json and
output_dir/job_manifest.json.

Author: G. Ruffini / Technical Note companion code
"""

import os
import sys
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from compositional_cat import (
    JointedCat, generate_dataset, CONDITIONS, LEVEL_PARAMS, sample_params
)
from gated_resnet import GatedAutoencoder


# ═══════════════════════════════════════════════════════════
# Job logging: registry and per-run manifest
# ═══════════════════════════════════════════════════════════

def _job_id() -> str:
    """Generate a unique job id (YYYYMMDD_HHMMSS)."""
    return datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')


def _args_to_dict(args) -> dict:
    """Convert argparse.Namespace to a JSON-serialisable dict (reproducible run)."""
    return vars(args) if hasattr(args, '__dict__') else dict(args)


def _load_registry(registry_path: str) -> list:
    """Load jobs registry from JSON file; return list of job records."""
    path = Path(registry_path)
    if not path.exists():
        return []
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, IOError):
        return []


def _save_registry(registry_path: str, jobs: list) -> None:
    """Write jobs registry to JSON file."""
    path = Path(registry_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(jobs, f, indent=2)


def _write_manifest(output_dir: str, record: dict) -> None:
    """Write job_manifest.json into output_dir."""
    path = Path(output_dir) / 'job_manifest.json'
    with open(path, 'w') as f:
        json.dump(record, f, indent=2)


def _update_registry_with_job(registry_path: str, job_id: str, updated: dict) -> None:
    """Load registry, update or append the record for job_id, save."""
    jobs = _load_registry(registry_path)
    found = False
    for i, j in enumerate(jobs):
        if j.get('job_id') == job_id:
            jobs[i] = {**j, **updated}
            found = True
            break
    if not found:
        jobs.append(updated)
    _save_registry(registry_path, jobs)


# ═══════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════

class CatDataset(Dataset):
    """
    On-the-fly generation of cat images.
    More efficient than pre-rendering to disk for experimentation.
    """
    def __init__(
        self,
        condition: str,
        n_samples: int = 5000,
        img_size: int = 64,
        seed: int = 42,
    ):
        self.condition = condition
        self.active_levels = CONDITIONS[condition]
        self.n_samples = n_samples
        self.img_size = img_size
        self.rng = np.random.RandomState(seed)

        # Pre-generate all parameter sets
        self.all_params = []
        for _ in range(n_samples):
            self.all_params.append(
                sample_params(self.active_levels, self.rng)
            )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        cat = JointedCat()
        cat.params = self.all_params[idx]
        img = cat.render(img_size=self.img_size)
        # Convert to tensor [C, H, W] in [0, 1]
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, len(self.active_levels)


# ═══════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════

def train(
    model: GatedAutoencoder,
    train_loader: DataLoader,
    n_epochs: int = 30,
    lr: float = 1e-3,
    gate_penalty: float = 0.01,
    device: str = 'cpu',
    log_every: int = 50,
    job_pct_start: float = 15.0,
    job_pct_end: float = 90.0,
) -> list:
    """Train the gated autoencoder on reconstruction with gate penalty.

    Args:
        model: GatedAutoencoder to train.
        train_loader: DataLoader for "Everything" condition.
        n_epochs: Number of training epochs.
        lr: Adam learning rate.
        gate_penalty: Weight λ for the penalty Σ α_ℓ.
        device: 'cpu' or 'cuda'.
        log_every: Unused; kept for API compatibility.
        job_pct_start: Job completion % at start of training (for progress bar).
        job_pct_end: Job completion % at end of training.

    Returns:
        List of dicts, one per epoch, with keys: recon_loss, gate_loss, total_loss,
        mean_gate, effective_depth, n_batches.
    """

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = []

    epoch_pbar = tqdm(
        range(n_epochs),
        desc="Training",
        unit="epoch",
        leave=True,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
    )
    for epoch in epoch_pbar:
        epoch_pbar.set_postfix_str(f"Job {job_pct_start + (job_pct_end - job_pct_start) * ((epoch + 1) / max(1, n_epochs)):.0f}%")

        model.train()
        epoch_stats = {
            'recon_loss': 0, 'gate_loss': 0, 'total_loss': 0,
            'mean_gate': 0, 'effective_depth': 0, 'n_batches': 0
        }

        batch_pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{n_epochs}",
            leave=False,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        for batch_idx, (x, _) in enumerate(batch_pbar):
            x = x.to(device)
            optimizer.zero_grad()
            loss, diag = model.compute_loss(x, gate_penalty=gate_penalty)
            loss.backward()
            optimizer.step()

            for k in ['recon_loss', 'gate_loss', 'total_loss',
                       'mean_gate', 'effective_depth']:
                epoch_stats[k] += diag[k]
            epoch_stats['n_batches'] += 1
            batch_pbar.set_postfix_str(
                f"loss={diag['total_loss']:.4f} D_eff={diag['effective_depth']:.2f}",
                refresh=False,
            )

        scheduler.step()

        # Average
        nb = epoch_stats['n_batches']
        for k in ['recon_loss', 'gate_loss', 'total_loss',
                   'mean_gate', 'effective_depth']:
            epoch_stats[k] /= nb

        history.append(epoch_stats)
        epoch_pbar.set_postfix_str(
            f"recon={epoch_stats['recon_loss']:.4f} D_eff={epoch_stats['effective_depth']:.2f}",
            refresh=True,
        )

    return history


# ═══════════════════════════════════════════════════════════
# Evaluation: gate patterns per condition
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_gates(
    model: GatedAutoencoder,
    condition: str,
    n_samples: int = 500,
    img_size: int = 64,
    device: str = 'cpu',
    seed: int = 999,
) -> dict:
    """Evaluate gate activations and reconstruction error on a single condition.

    Args:
        model: Trained GatedAutoencoder.
        condition: Name from CONDITIONS (e.g. 'FullPose', 'Everything').
        n_samples: Number of samples to evaluate.
        img_size: Image size for the generated cat dataset.
        device: 'cpu' or 'cuda'.
        seed: Random seed for dataset generation.

    Returns:
        Dict with condition, n_active_levels, active_levels, gate_means, gate_stds,
        effective_depth_mean, effective_depth_std, recon_error_mean, recon_error_std.
    """
    model.eval()
    model.to(device)

    dataset = CatDataset(condition, n_samples=n_samples,
                         img_size=img_size, seed=seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_gates = []  # list of (B, n_layers) arrays
    all_recon = []

    for x, _ in tqdm(loader, desc=condition, leave=False, unit="batch"):
        x = x.to(device)
        x_recon, z, gates = model(x)

        if gates:
            gate_vals = torch.stack([g.mean(dim=0) if g.dim() > 0 else g
                                     for g in gates])  # (n_layers,)
            # Actually collect per-sample gates
            gate_matrix = torch.stack([g for g in gates], dim=0)  # (n_layers, B) or similar
            # Handle shape: gates[i] is (B,) after squeeze
            per_sample = torch.stack(gates, dim=1)  # (B, n_layers)
            all_gates.append(per_sample.cpu().numpy())

        recon_err = F.mse_loss(x_recon, x, reduction='none').mean(dim=[1,2,3])
        all_recon.append(recon_err.cpu().numpy())

    all_gates = np.concatenate(all_gates, axis=0)  # (N, n_layers)
    all_recon = np.concatenate(all_recon, axis=0)   # (N,)

    n_layers = all_gates.shape[1]
    active_levels = CONDITIONS[condition]

    result = {
        'condition': condition,
        'n_active_levels': len(active_levels),
        'active_levels': active_levels,
        'gate_means': all_gates.mean(axis=0).tolist(),
        'gate_stds': all_gates.std(axis=0).tolist(),
        'effective_depth_mean': float(all_gates.sum(axis=1).mean()),
        'effective_depth_std': float(all_gates.sum(axis=1).std()),
        'recon_error_mean': float(all_recon.mean()),
        'recon_error_std': float(all_recon.std()),
    }
    return result


import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════
# Plotting: the four predictions
# ═══════════════════════════════════════════════════════════

def plot_results(results: list, output_dir: str = 'results') -> None:
    """Plot the four key predictions and save to output_dir.

    Args:
        results: List of dicts from evaluate_gates (one per condition), with
            condition, n_active_levels, gate_means, effective_depth_mean/std,
            recon_error_mean/std.
        output_dir: Directory to write dynamic_depth_results.png.

    Returns:
        None.
    """
    os.makedirs(output_dir, exist_ok=True)

    conditions = [r['condition'] for r in results]
    n_levels = [r['n_active_levels'] for r in results]
    n_layers = len(results[0]['gate_means'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Compositional Dynamic Depth: Experimental Results',
                 fontsize=14, fontweight='bold')

    # ── Plot 1: Gate heatmap (conditions × layers) ──
    ax = axes[0, 0]
    gate_matrix = np.array([r['gate_means'] for r in results])
    im = ax.imshow(gate_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=1)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Condition')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([f"{c}\n({n})" for c, n in zip(conditions, n_levels)],
                       fontsize=7)
    ax.set_title('Prediction 1: Gate–Complexity Alignment')
    plt.colorbar(im, ax=ax, label='Mean gate α')

    # ── Plot 2: Effective depth vs complexity ──
    ax = axes[0, 1]
    d_eff = [r['effective_depth_mean'] for r in results]
    d_eff_std = [r['effective_depth_std'] for r in results]
    ax.errorbar(n_levels, d_eff, yerr=d_eff_std, fmt='o-',
                capsize=3, color='#d62728')
    ax.set_xlabel('Number of active generative levels')
    ax.set_ylabel('Effective depth D_eff = Σ α_ℓ')
    ax.set_title('Prediction 2: Depth–Complexity Curve')
    ax.grid(True, alpha=0.3)

    # ── Plot 3: Per-layer gate profiles ──
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(r['gate_means'], label=f"{r['condition']} ({r['n_active_levels']})",
                color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean gate value α_ℓ')
    ax.set_title('Prediction 3: Per-Layer Gate Profiles')
    ax.legend(fontsize=6, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # ── Plot 4: Reconstruction error vs complexity ──
    ax = axes[1, 1]
    recon = [r['recon_error_mean'] for r in results]
    recon_std = [r['recon_error_std'] for r in results]
    ax.errorbar(n_levels, recon, yerr=recon_std, fmt='s-',
                capsize=3, color='#1f77b4')
    ax.set_xlabel('Number of active generative levels')
    ax.set_ylabel('Reconstruction MSE')
    ax.set_title('Reconstruction Error vs Complexity')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dynamic_depth_results.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved results plot to {output_dir}/dynamic_depth_results.png")


# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

def main() -> None:
    """Parse CLI, set up job logging, and run the train-and-evaluate pipeline.

    Writes job_manifest.json in output_dir and updates jobs_registry (see
    --jobs_registry). On failure, the job is marked failed in both.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=2,
                       help='Residual blocks per stage')
    parser.add_argument('--n_stages', type=int, default=4,
                       help='Number of encoder stages')
    parser.add_argument('--n_train', type=int, default=5000)
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gate_penalty', type=float, default=0.01)
    parser.add_argument('--gate_init_bias', type=float, default=2.0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='DataLoader workers. Use 4 on Kaggle/GPU so GPU is not starved by on-the-fly image generation.')
    parser.add_argument('--jobs_registry', type=str, default='../jobs/jobs_registry.json',
                       help='Path to central jobs registry JSON (from scripts/ default: ../jobs/...)')
    parser.add_argument('--calibrate', action='store_true',
                       help='Run one epoch only and print suggested n_epochs for ~8h run')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Job logging: start ──
    job_id = _job_id()
    start_time = time.perf_counter()
    start_iso = datetime.now(timezone.utc).isoformat()
    args_dict = _args_to_dict(args)
    registry_path = os.path.abspath(args.jobs_registry)

    initial_record = {
        'job_id': job_id,
        'start_time_iso': start_iso,
        'end_time_iso': None,
        'duration_sec': None,
        'status': 'running',
        'args': args_dict,
        'output_dir': os.path.abspath(args.output_dir),
        'summary': None,
        'error': None,
    }
    _write_manifest(args.output_dir, initial_record)
    _update_registry_with_job(registry_path, job_id, initial_record)
    print(f"Job id: {job_id} | Registry: {registry_path}")

    try:
        _run_pipeline(args, device, job_id, start_time, start_iso, registry_path)
    except Exception as e:
        end_time = time.perf_counter()
        end_iso = datetime.now(timezone.utc).isoformat()
        duration_sec = end_time - start_time
        fail_record = {
            'job_id': job_id,
            'end_time_iso': end_iso,
            'duration_sec': round(duration_sec, 2),
            'status': 'failed',
            'error': str(e),
        }
        _write_manifest(args.output_dir, {**initial_record, **fail_record})
        _update_registry_with_job(registry_path, job_id, {**initial_record, **fail_record})
        raise


def _run_pipeline(args, device: str, job_id: str, start_time: float, start_iso: str, registry_path: str) -> None:
    """Execute the full train-and-evaluate pipeline. Updates job manifest and registry on success.

    Steps: (1) create dataset, (2) build model, (3) train (or calibrate with 1 epoch),
    (4) evaluate gates on all conditions, (5) plot, (6) write success manifest and registry.
    """
    initial_record = {
        'job_id': job_id,
        'start_time_iso': start_iso,
        'end_time_iso': None,
        'duration_sec': None,
        'status': 'running',
        'args': _args_to_dict(args),
        'output_dir': os.path.abspath(args.output_dir),
        'summary': None,
        'error': None,
    }

    # Job progress: steps 1–6 with approximate weights: 1=5%, 2=5%, 3=75%, 4=10%, 5=5%
    def _job_msg(pct: float, step: int, msg: str) -> None:
        print(f"\nJob: {pct:.0f}% | Step {step}/6: {msg}")
        print("=" * 60)

    # ── 1. Create training dataset (Everything condition) ──
    _job_msg(5, 1, "Creating training dataset (Everything condition)")
    train_dataset = CatDataset(
        'Everything', n_samples=args.n_train,
        img_size=args.img_size, seed=42
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
    )

    # ── 2. Build model ──
    _job_msg(10, 2, "Building Symmetry-Gated Autoencoder")
    n_total_layers = args.n_stages * args.n_blocks
    print(f"  {args.n_stages} stages × {args.n_blocks} blocks = "
          f"{n_total_layers} gated layers")

    model = GatedAutoencoder(
        img_size=args.img_size,
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        n_blocks_per_stage=args.n_blocks,
        n_stages=args.n_stages,
        gated=True,
        gate_init_bias=args.gate_init_bias,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")

    # ── 3. Train ──
    n_epochs_run = 1 if args.calibrate else args.n_epochs
    _job_msg(15, 3, f"Training for {n_epochs_run} epochs" +
             (" (calibration: 1 epoch)" if args.calibrate else ""))
    t_train_start = time.perf_counter()
    history = train(
        model, train_loader,
        n_epochs=n_epochs_run,
        lr=args.lr,
        gate_penalty=args.gate_penalty,
        device=device,
        job_pct_start=15.0,
        job_pct_end=90.0,
    )
    time_train_sec = time.perf_counter() - t_train_start

    if args.calibrate:
        time_per_epoch_sec = time_train_sec / n_epochs_run
        target_sec = 8 * 3600
        suggested_n_epochs = int(target_sec / time_per_epoch_sec * 0.95)
        print(f"\nCalibration: 1 epoch took {time_per_epoch_sec:.1f} s.")
        print(f"Suggested n_epochs for ~8 h run: {suggested_n_epochs}")
        end_iso = datetime.now(timezone.utc).isoformat()
        duration_sec = time_train_sec
        summary = {
            'calibration': True,
            'time_per_epoch_sec': round(time_per_epoch_sec, 2),
            'suggested_n_epochs_8h': suggested_n_epochs,
        }
        success_record = {
            **initial_record,
            'end_time_iso': end_iso,
            'duration_sec': round(duration_sec, 2),
            'status': 'success',
            'summary': summary,
        }
        _write_manifest(args.output_dir, success_record)
        _update_registry_with_job(registry_path, job_id, success_record)
        return

    # Save training history
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save model
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'gated_autoencoder.pt'))

    # ── 4. Evaluate on all conditions ──
    _job_msg(90, 4, "Evaluating gate patterns on all conditions")

    eval_conditions = [
        'Static', 'CameraOnly', 'CameraBody', 'CameraBodySpine',
        'FullPose', 'PoseHeadTail', 'PlusAppearance', 'Everything'
    ]

    results = []
    for cond in tqdm(
        eval_conditions,
        desc="Conditions",
        unit="cond",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    ):
        r = evaluate_gates(
            model, cond,
            n_samples=args.n_eval,
            img_size=args.img_size,
            device=device,
        )
        results.append(r)
        tqdm.write(
            f"  {cond}: D_eff={r['effective_depth_mean']:.2f} ± {r['effective_depth_std']:.2f} "
            f"| Recon MSE={r['recon_error_mean']:.4f}"
        )

    # Save results
    with open(os.path.join(args.output_dir, 'gate_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── 5. Plot ──
    _job_msg(95, 5, "Generating plots")
    plot_results(results, output_dir=args.output_dir)

    # ── 6. Summary ──
    _job_msg(100, 6, "Done – summary")
    print("SUMMARY: Depth-Complexity Relationship")
    print("=" * 60)
    print(f"{'Condition':<20s} {'Active':>7s} {'D_eff':>8s} {'Recon':>8s}")
    print("-" * 45)
    for r in results:
        print(f"{r['condition']:<20s} {r['n_active_levels']:>7d} "
              f"{r['effective_depth_mean']:>8.2f} "
              f"{r['recon_error_mean']:>8.4f}")

    # Check Prediction 2: monotone D_eff
    d_effs = [r['effective_depth_mean'] for r in results]
    monotone = all(d_effs[i] <= d_effs[i+1] + 0.5
                   for i in range(len(d_effs)-1))
    print(f"\nPrediction 2 (monotone D_eff): "
          f"{'SUPPORTED' if monotone else 'NOT CLEARLY SUPPORTED'}")

    print(f"\nAll outputs saved to: {args.output_dir}/")
    print("  - gated_autoencoder.pt (model weights)")
    print("  - training_history.json")
    print("  - gate_analysis.json")
    print("  - dynamic_depth_results.png (plots)")

    # ── Job logging: success ──
    end_iso = datetime.now(timezone.utc).isoformat()
    duration_sec = time.perf_counter() - start_time
    last_epoch = history[-1] if history else {}
    d_effs = [r['effective_depth_mean'] for r in results]
    monotone_d_eff = all(d_effs[i] <= d_effs[i + 1] + 0.5 for i in range(len(d_effs) - 1))
    summary = {
        'final_recon_loss': last_epoch.get('recon_loss'),
        'final_gate_loss': last_epoch.get('gate_loss'),
        'final_D_eff_mean': last_epoch.get('effective_depth'),
        'monotone_D_eff': monotone_d_eff,
        'per_condition': [
            {'condition': r['condition'], 'n_active_levels': r['n_active_levels'],
             'D_eff_mean': r['effective_depth_mean'], 'recon_error_mean': r['recon_error_mean']}
            for r in results
        ],
    }
    success_record = {
        **initial_record,
        'end_time_iso': end_iso,
        'duration_sec': round(duration_sec, 2),
        'status': 'success',
        'summary': summary,
    }
    _write_manifest(args.output_dir, success_record)
    _update_registry_with_job(registry_path, job_id, success_record)


if __name__ == '__main__':
    main()
