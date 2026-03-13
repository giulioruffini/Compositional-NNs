"""
Training and Evaluation for Compositional Dynamic Depth — v5
=============================================================

Key design: SEPARATE MODEL PER CONDITION.

Each condition defines a different Lie-pseudogroup action on the
generative process. The hypothesis: more compositional levels →
richer image manifold → the autoencoder needs more depth (more
open gates) to achieve good reconstruction.

We train one gated autoencoder per condition with the SAME architecture
and penalty. Each model independently discovers how many gates it needs.
D_eff should increase with the number of active generative levels.

No blur, no tricks — just the pure group-action hierarchy:
  Level 0: Identity (static)
  Level 1: Camera (SE(2) + scale)
  Level 2: Body rigid motion
  Level 3: Spine deformation
  Level 4: Limb joints
  Level 5: Head + tail joints
  Level 6: Appearance (color, stripes)
  Level 7: Background

Author: G. Ruffini / Technical Note companion code — v5
"""

import os
import sys
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from compositional_cat_v2 import (
    JointedCat, CONDITIONS, LEVEL_PARAMS, sample_params
)
from gated_resnet import GatedAutoencoder


# ═══════════════════════════════════════════════════════════
# Dataset — single condition, no blur
# ═══════════════════════════════════════════════════════════

class CatDataset(Dataset):
    """On-the-fly cat image generation for a single condition."""

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
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, len(self.active_levels)


# ═══════════════════════════════════════════════════════════
# Training — one model at a time
# ═══════════════════════════════════════════════════════════

def train(
    model: GatedAutoencoder,
    train_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    gate_penalty_max: float = 0.001,
    gate_warmup_epochs: int = 10,
    device: str = 'cpu',
    verbose: bool = True,
) -> list:
    """Train with progressive gate penalty."""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = []

    for epoch in range(n_epochs):
        model.train()

        if epoch < gate_warmup_epochs:
            gate_penalty = gate_penalty_max * (epoch / max(1, gate_warmup_epochs))
        else:
            gate_penalty = gate_penalty_max

        epoch_stats = {
            'recon_loss': 0, 'gate_loss': 0, 'total_loss': 0,
            'mean_gate': 0, 'effective_depth': 0, 'n_batches': 0,
            'gate_penalty': gate_penalty,
        }

        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            optimizer.zero_grad()
            loss, diag = model.compute_loss(x, gate_penalty=gate_penalty)
            loss.backward()
            optimizer.step()

            for k in ['recon_loss', 'gate_loss', 'total_loss',
                       'mean_gate', 'effective_depth']:
                epoch_stats[k] += diag[k]
            epoch_stats['n_batches'] += 1

        scheduler.step()

        nb = epoch_stats['n_batches']
        for k in ['recon_loss', 'gate_loss', 'total_loss',
                   'mean_gate', 'effective_depth']:
            epoch_stats[k] /= nb

        history.append(epoch_stats)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"recon={epoch_stats['recon_loss']:.4f} | "
                  f"D_eff={epoch_stats['effective_depth']:.2f} | "
                  f"mean_α={epoch_stats['mean_gate']:.3f}")

    return history


# ═══════════════════════════════════════════════════════════
# Evaluation
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
    """Evaluate gate activations on a specific condition."""
    model.eval()
    model.to(device)

    dataset = CatDataset(condition, n_samples=n_samples,
                         img_size=img_size, seed=seed)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    all_gates = []
    all_recon = []

    for x, _ in loader:
        x = x.to(device)
        x_recon, z, gates = model(x)

        if gates:
            per_sample = torch.stack(gates, dim=1)
            all_gates.append(per_sample.cpu().numpy())

        recon_err = F.mse_loss(x_recon, x, reduction='none').mean(dim=[1,2,3])
        all_recon.append(recon_err.cpu().numpy())

    all_gates = np.concatenate(all_gates, axis=0)
    all_recon = np.concatenate(all_recon, axis=0)

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


# ═══════════════════════════════════════════════════════════
# Plotting
# ═══════════════════════════════════════════════════════════

def plot_results(results: list, output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    conditions = [r['condition'] for r in results]
    n_levels = [r['n_active_levels'] for r in results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Compositional Dynamic Depth: Per-Condition Training (v5)',
                 fontsize=14, fontweight='bold')

    # Gate heatmap
    ax = axes[0, 0]
    gate_matrix = np.array([r['gate_means'] for r in results])
    im = ax.imshow(gate_matrix, aspect='auto', cmap='YlOrRd', vmin=0, vmax=1)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Condition')
    ax.set_yticks(range(len(conditions)))
    ax.set_yticklabels([f"{c}\n({n})" for c, n in zip(conditions, n_levels)],
                       fontsize=7)
    ax.set_title('Gate–Complexity Alignment')
    plt.colorbar(im, ax=ax, label='Mean gate prob')

    # D_eff vs complexity
    ax = axes[0, 1]
    d_eff = [r['effective_depth_mean'] for r in results]
    d_eff_std = [r['effective_depth_std'] for r in results]
    ax.errorbar(n_levels, d_eff, yerr=d_eff_std, fmt='o-',
                capsize=3, color='#d62728')
    ax.set_xlabel('Number of active generative levels')
    ax.set_ylabel('Effective depth D_eff = Σ α_ℓ')
    ax.set_title('Prediction: D_eff ∝ compositional depth')
    ax.grid(True, alpha=0.3)

    # Per-layer gate profiles
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    for i, r in enumerate(results):
        color = cmap(i / max(1, len(results) - 1))
        ax.plot(r['gate_means'], label=f"{r['condition']} ({r['n_active_levels']})",
                color=color, alpha=0.8, linewidth=1.5)
    ax.set_xlabel('Layer index')
    ax.set_ylabel('Mean gate probability')
    ax.set_title('Per-Layer Gate Profiles (separate models)')
    ax.legend(fontsize=6, loc='lower left', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # Recon error vs complexity
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


def plot_training_histories(all_histories: dict, output_dir: str = 'results'):
    """Plot training D_eff curves for all conditions."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.cm.viridis

    conditions = list(all_histories.keys())
    for i, cond in enumerate(conditions):
        history = all_histories[cond]
        color = cmap(i / max(1, len(conditions) - 1))
        epochs = range(1, len(history) + 1)

        n_lev = len(CONDITIONS[cond])
        axes[0].plot(epochs, [h['effective_depth'] for h in history],
                     color=color, label=f'{cond} ({n_lev})', alpha=0.8)
        axes[1].plot(epochs, [h['recon_loss'] for h in history],
                     color=color, label=f'{cond} ({n_lev})', alpha=0.8)

    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('D_eff')
    axes[0].set_title('Effective Depth During Training')
    axes[0].legend(fontsize=6, ncol=2); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Recon MSE')
    axes[1].set_title('Reconstruction Loss During Training')
    axes[1].legend(fontsize=6, ncol=2); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_dir}/training_curves.png")


# ═══════════════════════════════════════════════════════════
# Main pipeline — per-condition training
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compositional Dynamic Depth — v5 (per-condition training)'
    )
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=2,
                       help='Residual blocks per stage')
    parser.add_argument('--n_stages', type=int, default=4,
                       help='Number of encoder stages')
    parser.add_argument('--n_train', type=int, default=2000,
                       help='Training samples per condition')
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--n_epochs', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gate_penalty', type=float, default=0.002,
                       help='Gate penalty λ (same for all conditions)')
    parser.add_argument('--gate_warmup', type=int, default=15,
                       help='Epochs to ramp gate penalty from 0 to max')
    parser.add_argument('--gate_init_bias', type=float, default=0.0,
                       help='Gate bias init (0=neutral, >0=biased ON)')
    parser.add_argument('--gate_tau', type=float, default=1.0,
                       help='Gumbel-Softmax temperature')
    parser.add_argument('--penalty_sweep', nargs='*', type=float, default=None,
                       help='Sweep multiple penalty values (e.g. --penalty_sweep 0.001 0.003 0.005)')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    # Auto-detect GPU
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    eval_conditions = [
        'Static', 'CameraOnly', 'CameraBody', 'CameraBodySpine',
        'FullPose', 'PoseHeadTail', 'PlusAppearance', 'Everything'
    ]

    n_total_layers = args.n_stages * args.n_blocks

    # Penalty sweep: run full pipeline for each penalty
    penalties = args.penalty_sweep if args.penalty_sweep else [args.gate_penalty]

    for penalty_val in penalties:
        out_dir = args.output_dir if len(penalties) == 1 else \
                  os.path.join(args.output_dir, f'penalty_{penalty_val:.4f}')
        os.makedirs(out_dir, exist_ok=True)

        print("=" * 60)
        print(f"Compositional Dynamic Depth — Per-Condition Training (v5)")
        print(f"Gate penalty λ = {penalty_val}")
        print("=" * 60)
        print(f"  Architecture: {args.n_stages} stages × {args.n_blocks} blocks = "
              f"{n_total_layers} gated layers")
        print(f"  Image: {args.img_size}×{args.img_size}, "
              f"latent_ch={args.latent_dim}, base_ch={args.base_channels}")
        print(f"  Training: {args.n_train} samples, {args.n_epochs} epochs")
        print(f"  Device: {device}")
        print()

        run_pipeline(
            args, eval_conditions, penalty_val, out_dir, device
        )

    if len(penalties) > 1:
        plot_sweep_comparison(args.output_dir, penalties)


def run_pipeline(args, eval_conditions, gate_penalty, output_dir, device):
    """Run full per-condition training pipeline for one penalty value."""
    all_results = []
    all_histories = {}

    for cond in eval_conditions:
        n_lev = len(CONDITIONS[cond])
        print(f"{'─'*60}")
        print(f"Training: {cond} (levels={CONDITIONS[cond]}, dim={n_lev})")
        print(f"{'─'*60}")

        # Fresh model for each condition
        model = GatedAutoencoder(
            img_size=args.img_size,
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            n_blocks_per_stage=args.n_blocks,
            n_stages=args.n_stages,
            gated=True,
            gate_init_bias=args.gate_init_bias,
            gate_tau=args.gate_tau,
        )

        # Train dataset for this condition only
        train_dataset = CatDataset(
            cond, n_samples=args.n_train,
            img_size=args.img_size, seed=42,
        )
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size,
            shuffle=True, num_workers=0,
        )

        # Train
        history = train(
            model, train_loader,
            n_epochs=args.n_epochs,
            lr=args.lr,
            gate_penalty_max=gate_penalty,
            gate_warmup_epochs=args.gate_warmup,
            device=device,
        )
        all_histories[cond] = history

        # Evaluate
        r = evaluate_gates(
            model, cond,
            n_samples=args.n_eval,
            img_size=args.img_size,
            device=device,
        )
        all_results.append(r)

        print(f"  → D_eff = {r['effective_depth_mean']:.2f} ± "
              f"{r['effective_depth_std']:.2f} | "
              f"Recon = {r['recon_error_mean']:.4f}")
        gate_str = ' '.join(f'{g:.2f}' for g in r['gate_means'])
        print(f"    Gates: [{gate_str}]")

        # Save individual model
        torch.save(model.state_dict(),
                   os.path.join(output_dir, f'model_{cond}.pt'))
        print()

    # Save results
    with open(os.path.join(output_dir, 'gate_analysis.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    # Plots
    print("=" * 60)
    print("Generating plots")
    print("=" * 60)
    plot_results(all_results, output_dir=output_dir)
    plot_training_histories(all_histories, output_dir=output_dir)

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY: Depth vs Compositional Complexity (λ={gate_penalty})")
    print("=" * 60)
    print(f"{'Condition':<20s} {'Levels':>7s} {'D_eff':>8s} {'Recon':>8s}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['condition']:<20s} {r['n_active_levels']:>7d} "
              f"{r['effective_depth_mean']:>8.2f} "
              f"{r['recon_error_mean']:>8.4f}")

    d_effs = [r['effective_depth_mean'] for r in all_results]
    monotone = all(d_effs[i] <= d_effs[i+1] + 0.5
                   for i in range(len(d_effs)-1))
    print(f"\nMonotone D_eff: "
          f"{'YES ✓' if monotone else 'NOT CLEARLY'}")

    spread = d_effs[-1] - d_effs[0]
    print(f"D_eff spread (Everything − Static): {spread:.2f}")

    # Correlation
    n_levs = np.array([r['n_active_levels'] for r in all_results])
    d_arr = np.array(d_effs)
    if n_levs.std() > 0 and d_arr.std() > 0:
        corr = np.corrcoef(n_levs, d_arr)[0, 1]
        print(f"Pearson correlation (n_levels, D_eff): {corr:.3f}")

    print(f"\nAll outputs saved to: {output_dir}/")


def plot_sweep_comparison(base_dir: str, penalties: list):
    """Compare D_eff curves across penalty values."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Penalty Sweep: D_eff vs Compositional Complexity', fontsize=13)
    cmap = plt.cm.plasma

    for i, p in enumerate(penalties):
        pdir = os.path.join(base_dir, f'penalty_{p:.4f}')
        json_path = os.path.join(pdir, 'gate_analysis.json')
        if not os.path.exists(json_path):
            continue
        with open(json_path) as f:
            results = json.load(f)

        n_levels = [r['n_active_levels'] for r in results]
        d_effs = [r['effective_depth_mean'] for r in results]
        recons = [r['recon_error_mean'] for r in results]
        color = cmap(i / max(1, len(penalties) - 1))

        axes[0].plot(n_levels, d_effs, 'o-', color=color,
                     label=f'λ={p}', linewidth=2, markersize=6)
        axes[1].plot(n_levels, recons, 's-', color=color,
                     label=f'λ={p}', linewidth=2, markersize=6)

    axes[0].set_xlabel('Active generative levels')
    axes[0].set_ylabel('Effective depth D_eff')
    axes[0].set_title('D_eff scaling')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Active generative levels')
    axes[1].set_ylabel('Reconstruction MSE')
    axes[1].set_title('Reconstruction error')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'penalty_sweep.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved sweep comparison to {base_dir}/penalty_sweep.png")


if __name__ == '__main__':
    main()
