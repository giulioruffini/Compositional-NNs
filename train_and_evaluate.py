"""
Training and Evaluation for Compositional Dynamic Depth — v2
=============================================================

Key changes from v1:
  1. MIXED-CONDITION training: each batch draws from ALL complexity
     levels, so the model learns to allocate depth dynamically.
  2. Progressive gate penalty: ramps from 0 to λ_max over warmup epochs,
     letting the model first learn good representations before pruning.
  3. Uses compositional_cat_v2 (bigger cat, wider ranges).
  4. Gate module uses [AvgPool, StdPool] for richer conditioning.

Author: G. Ruffini / Technical Note companion code — v2
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
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
# Datasets
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


class MixedConditionDataset(Dataset):
    """
    Balanced sampling from ALL conditions.

    Each sample is drawn from a randomly chosen condition.
    This is the KEY FIX: the model sees inputs of varying complexity
    during training, so gate sparsity penalty selectively closes gates
    for simpler inputs while keeping them open for complex ones.
    """

    TRAIN_CONDITIONS = [
        'Static', 'CameraOnly', 'CameraBody', 'CameraBodySpine',
        'FullPose', 'PoseHeadTail', 'PlusAppearance', 'Everything',
    ]

    def __init__(
        self,
        n_samples_per_condition: int = 1000,
        img_size: int = 64,
        seed: int = 42,
    ):
        self.img_size = img_size
        self.conditions = self.TRAIN_CONDITIONS
        self.n_per_cond = n_samples_per_condition
        self.total = len(self.conditions) * n_samples_per_condition

        self.all_params = []
        self.all_n_levels = []
        self.all_condition_idx = []
        rng = np.random.RandomState(seed)

        for cond_idx, cond in enumerate(self.conditions):
            active = CONDITIONS[cond]
            for _ in range(n_samples_per_condition):
                self.all_params.append(sample_params(active, rng))
                self.all_n_levels.append(len(active))
                self.all_condition_idx.append(cond_idx)

        # Shuffle so conditions are mixed within batches
        order = np.random.RandomState(seed + 1).permutation(self.total)
        self.all_params = [self.all_params[i] for i in order]
        self.all_n_levels = [self.all_n_levels[i] for i in order]
        self.all_condition_idx = [self.all_condition_idx[i] for i in order]

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        cat = JointedCat()
        cat.params = self.all_params[idx]
        img = cat.render(img_size=self.img_size)
        arr = np.array(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, self.all_n_levels[idx]


# ═══════════════════════════════════════════════════════════
# Training with progressive gate penalty
# ═══════════════════════════════════════════════════════════

def train(
    model: GatedAutoencoder,
    train_loader: DataLoader,
    n_epochs: int = 50,
    lr: float = 1e-3,
    gate_penalty_max: float = 0.05,
    gate_warmup_epochs: int = 10,
    device: str = 'cpu',
    log_every: int = 50,
) -> list:
    """
    Train with progressive gate penalty.

    For the first `gate_warmup_epochs`, λ ramps linearly from 0 to
    gate_penalty_max. This lets the model first learn good feature
    representations, then progressively close unnecessary gates.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

    history = []

    for epoch in range(n_epochs):
        model.train()

        # Progressive gate penalty
        if epoch < gate_warmup_epochs:
            gate_penalty = gate_penalty_max * (epoch / max(1, gate_warmup_epochs))
        else:
            gate_penalty = gate_penalty_max

        epoch_stats = {
            'recon_loss': 0, 'gate_loss': 0, 'total_loss': 0,
            'mean_gate': 0, 'effective_depth': 0, 'n_batches': 0,
            'gate_penalty': gate_penalty,
        }

        for batch_idx, (x, n_levels) in enumerate(train_loader):
            x = x.to(device)
            n_lev = n_levels.to(device) if isinstance(n_levels, torch.Tensor) else torch.tensor(n_levels, device=device)
            optimizer.zero_grad()
            loss, diag = model.compute_loss(
                x, gate_penalty=gate_penalty, n_levels=n_lev
            )
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

        print(f"Epoch {epoch+1:3d}/{n_epochs} | "
              f"recon={epoch_stats['recon_loss']:.4f} | "
              f"gate={epoch_stats['gate_loss']:.4f} | "
              f"λ={gate_penalty:.4f} | "
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
            per_sample = torch.stack(gates, dim=1)  # (B, n_layers)
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
    fig.suptitle('Compositional Dynamic Depth: Experimental Results (v2)',
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
    ax.set_title('Prediction 1: Gate–Complexity Alignment')
    plt.colorbar(im, ax=ax, label='Mean gate α')

    # D_eff vs complexity
    ax = axes[0, 1]
    d_eff = [r['effective_depth_mean'] for r in results]
    d_eff_std = [r['effective_depth_std'] for r in results]
    ax.errorbar(n_levels, d_eff, yerr=d_eff_std, fmt='o-',
                capsize=3, color='#d62728')
    ax.set_xlabel('Number of active generative levels')
    ax.set_ylabel('Effective depth D_eff = Σ α_ℓ')
    ax.set_title('Prediction 2: Depth–Complexity Curve')
    ax.grid(True, alpha=0.3)

    # Per-layer gate profiles
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


def plot_training_history(history: list, output_dir: str = 'results'):
    """Plot training curves including gate penalty schedule."""
    os.makedirs(output_dir, exist_ok=True)

    epochs = range(1, len(history) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ax = axes[0]
    ax.plot(epochs, [h['recon_loss'] for h in history], label='Recon')
    ax.plot(epochs, [h['total_loss'] for h in history], label='Total', alpha=0.7)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.set_title('Training Loss'); ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [h['effective_depth'] for h in history], color='#d62728')
    ax.set_xlabel('Epoch'); ax.set_ylabel('D_eff')
    ax.set_title('Effective Depth During Training'); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, [h.get('gate_penalty', 0) for h in history], color='purple')
    ax.set_xlabel('Epoch'); ax.set_ylabel('λ (gate penalty)')
    ax.set_title('Gate Penalty Schedule'); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_dir}/training_curves.png")


# ═══════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compositional Dynamic Depth — v2 (mixed-condition training)'
    )
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--base_channels', type=int, default=32)
    parser.add_argument('--n_blocks', type=int, default=2,
                       help='Residual blocks per stage')
    parser.add_argument('--n_stages', type=int, default=4,
                       help='Number of encoder stages')
    parser.add_argument('--n_train_per_cond', type=int, default=1000,
                       help='Training samples PER CONDITION (total = 8x this)')
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gate_penalty', type=float, default=0.05,
                       help='Max gate penalty (ramped during warmup)')
    parser.add_argument('--gate_warmup', type=int, default=10,
                       help='Epochs to ramp gate penalty from 0 to max')
    parser.add_argument('--gate_init_bias', type=float, default=2.0)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = args.device
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Create MIXED training dataset ──
    print("=" * 60)
    print("Step 1: Creating MIXED-CONDITION training dataset")
    print("=" * 60)
    train_dataset = MixedConditionDataset(
        n_samples_per_condition=args.n_train_per_cond,
        img_size=args.img_size,
        seed=42,
    )
    total_train = len(train_dataset)
    print(f"  {len(MixedConditionDataset.TRAIN_CONDITIONS)} conditions × "
          f"{args.n_train_per_cond} samples = {total_train} total")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=0,
    )

    # ── 2. Build model ──
    print(f"\nStep 2: Building Symmetry-Gated Autoencoder (v2)")
    n_total_layers = args.n_stages * args.n_blocks
    print(f"  {args.n_stages} stages × {args.n_blocks} blocks = "
          f"{n_total_layers} gated layers")
    print(f"  Gate: [AvgPool, StdPool] → Linear → σ (v2 upgrade)")

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
    print(f"\nStep 3: Training for {args.n_epochs} epochs "
          f"(gate warmup: {args.gate_warmup} epochs)")
    print("=" * 60)
    history = train(
        model, train_loader,
        n_epochs=args.n_epochs,
        lr=args.lr,
        gate_penalty_max=args.gate_penalty,
        gate_warmup_epochs=args.gate_warmup,
        device=device,
    )

    # Save
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    torch.save(model.state_dict(),
               os.path.join(args.output_dir, 'gated_autoencoder.pt'))
    plot_training_history(history, args.output_dir)

    # ── 4. Evaluate on all conditions ──
    print("\n" + "=" * 60)
    print("Step 4: Evaluating gate patterns on all conditions")
    print("=" * 60)

    eval_conditions = [
        'Static', 'CameraOnly', 'CameraBody', 'CameraBodySpine',
        'FullPose', 'PoseHeadTail', 'PlusAppearance', 'Everything'
    ]

    results = []
    for cond in eval_conditions:
        print(f"\n  Evaluating: {cond} "
              f"(active levels: {CONDITIONS[cond]})")
        r = evaluate_gates(
            model, cond,
            n_samples=args.n_eval,
            img_size=args.img_size,
            device=device,
        )
        results.append(r)
        print(f"    D_eff = {r['effective_depth_mean']:.2f} ± "
              f"{r['effective_depth_std']:.2f}")
        print(f"    Recon MSE = {r['recon_error_mean']:.4f}")
        gate_str = ' '.join(f'{g:.2f}' for g in r['gate_means'])
        print(f"    Gates: [{gate_str}]")

    with open(os.path.join(args.output_dir, 'gate_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)

    # ── 5. Plot ──
    print("\n" + "=" * 60)
    print("Step 5: Generating plots")
    print("=" * 60)
    plot_results(results, output_dir=args.output_dir)

    # ── 6. Summary ──
    print("\n" + "=" * 60)
    print("SUMMARY: Depth-Complexity Relationship")
    print("=" * 60)
    print(f"{'Condition':<20s} {'Active':>7s} {'D_eff':>8s} {'Recon':>8s}")
    print("-" * 45)
    for r in results:
        print(f"{r['condition']:<20s} {r['n_active_levels']:>7d} "
              f"{r['effective_depth_mean']:>8.2f} "
              f"{r['recon_error_mean']:>8.4f}")

    d_effs = [r['effective_depth_mean'] for r in results]
    monotone = all(d_effs[i] <= d_effs[i+1] + 0.5
                   for i in range(len(d_effs)-1))
    print(f"\nPrediction 2 (monotone D_eff): "
          f"{'SUPPORTED' if monotone else 'NOT CLEARLY SUPPORTED'}")

    spread = d_effs[-1] - d_effs[0]
    print(f"D_eff spread (Everything - Static): {spread:.2f}")
    print(f"  (aim for >1.0 for clear differentiation)")

    print(f"\nAll outputs saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
