"""
Symmetry-Gated ResNet
=====================
A ResNet with learnable per-layer gates α_ℓ ∈ [0,1] that modulate
residual contributions:

    h_{ℓ+1} = h_ℓ + α_ℓ(x, h_ℓ) · Δ_ℓ(h_ℓ)

The gate α_ℓ is a scalar computed from the pooled hidden state.
Training includes a compute penalty λ Σ α_ℓ to encourage sparsity.

For the compositional-cat experiment, we train an autoencoder
(reconstruction task) so that the network must learn to represent
the full generative variation.

Author: G. Ruffini / Technical Note companion code
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# Gate module
# ═══════════════════════════════════════════════════════════

class LayerGate(nn.Module):
    """Computes α_ℓ = σ(w^T · Pool(h_ℓ) + b). One scalar gate per sample.

    Args:
        channels: Number of input channels (pooled to 1×1 then projected to scalar).
        init_bias: Bias for the linear layer; positive so gates start open (σ(b) ≈ 0.88).
    """
    def __init__(self, channels: int, init_bias: float = 2.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels, 1)
        # Initialise bias positive so gates start open
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns gate value α ∈ [0, 1], shape (B, 1, 1, 1) for broadcasting."""
        pooled = self.pool(x).flatten(1)       # (B, C)
        gate = torch.sigmoid(self.fc(pooled))   # (B, 1)
        return gate.view(-1, 1, 1, 1)


# ═══════════════════════════════════════════════════════════
# Gated residual block
# ═══════════════════════════════════════════════════════════

class GatedResBlock(nn.Module):
    """Residual block with multiplicative gate: h' = h + α(h) · Δ(h).

    Args:
        channels: Conv in/out channels.
        gate_init_bias: Passed to LayerGate so gates start mostly open.
    """
    def __init__(self, channels: int, gate_init_bias: float = 2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gate = LayerGate(channels, init_bias=gate_init_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (output, gate_value) with gate_value shape (B,) after squeeze."""
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        alpha = self.gate(x)  # (B, 1, 1, 1)
        out = x + alpha * residual
        out = F.relu(out)

        return out, alpha.squeeze()


class ResBlock(nn.Module):
    """Standard (un-gated) residual block for baseline."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        return F.relu(x + residual)


# ═══════════════════════════════════════════════════════════
# Encoder / Decoder
# ═══════════════════════════════════════════════════════════

class GatedEncoder(nn.Module):
    """Encoder: image → latent vector. Stages of gated residual blocks with downsampling.

    Each stage has n_blocks_per_stage blocks; between stages, spatial size is halved
    and channels are doubled. Gates from all blocks are returned for analysis.

    Args:
        img_size: Input spatial size (H=W).
        latent_dim: Bottleneck dimension.
        base_channels: Channels after stem; doubles each stage.
        n_blocks_per_stage: Residual blocks per stage.
        n_stages: Number of stages (total gated layers = n_stages * n_blocks_per_stage).
        gated: If False, use plain ResBlocks (no gates).
        gate_init_bias: Passed to each GatedResBlock.
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 64,
        base_channels: int = 32,
        n_blocks_per_stage: int = 2,
        n_stages: int = 4,
        gated: bool = True,
        gate_init_bias: float = 2.0,
    ):
        super().__init__()
        self.gated = gated
        self.n_stages = n_stages
        self.n_blocks_per_stage = n_blocks_per_stage

        # Initial conv
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )

        # Build stages with increasing channels
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = base_channels

        for stage in range(n_stages):
            blocks = nn.ModuleList()
            for b in range(n_blocks_per_stage):
                if gated:
                    blocks.append(GatedResBlock(ch, gate_init_bias))
                else:
                    blocks.append(ResBlock(ch))
            self.stages.append(blocks)

            # Downsample between stages (except last)
            if stage < n_stages - 1:
                next_ch = ch * 2
                self.downsamples.append(nn.Sequential(
                    nn.Conv2d(ch, next_ch, 3, stride=2, padding=1, bias=False),
                    nn.BatchNorm2d(next_ch),
                    nn.ReLU(),
                ))
                ch = next_ch
            else:
                self.downsamples.append(nn.Identity())

        # Final projection to latent
        self.final_channels = ch
        # Compute spatial size after downsampling
        spatial = img_size
        for _ in range(n_stages - 1):
            spatial = spatial // 2
        self.final_spatial = spatial

        self.fc = nn.Linear(ch * spatial * spatial, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Returns (latent_vector, list_of_gate_values). Each gate value shape (B,)."""
        gates = []
        h = self.stem(x)

        for stage_idx, (blocks, down) in enumerate(
            zip(self.stages, self.downsamples)
        ):
            for block in blocks:
                if self.gated:
                    h, alpha = block(h)
                    gates.append(alpha)
                else:
                    h = block(h)

            if stage_idx < self.n_stages - 1:
                h = down(h)

        z = self.fc(h.flatten(1))
        return z, gates


class Decoder(nn.Module):
    """Decoder: latent → reconstructed image. FC then transposed convs to upsample.

    Mirror of encoder spatial/channel schedule: starts at smallest spatial size and
    final_ch channels, upsamples (n_stages - 1) times, then 3×3 conv to RGB + sigmoid.

    Args:
        img_size: Target image spatial size.
        latent_dim: Bottleneck dimension.
        base_channels: Base channel count (final_ch = base * 2^(n_stages-1)).
        n_stages: Must match encoder for correct spatial size.
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 64,
        base_channels: int = 32,
        n_stages: int = 4,
    ):
        super().__init__()

        # Compute starting spatial size and channels
        final_ch = base_channels * (2 ** (n_stages - 1))
        spatial = img_size
        for _ in range(n_stages - 1):
            spatial = spatial // 2

        self.fc = nn.Linear(latent_dim, final_ch * spatial * spatial)
        self.initial_shape = (final_ch, spatial, spatial)

        # Build upsampling stages
        layers = []
        ch = final_ch
        for stage in range(n_stages - 1):
            next_ch = ch // 2
            layers.extend([
                nn.ConvTranspose2d(ch, next_ch, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(),
                nn.Conv2d(next_ch, next_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(),
            ])
            ch = next_ch

        # Final output
        layers.append(nn.Conv2d(ch, 3, 3, padding=1))
        layers.append(nn.Sigmoid())

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed image (B, 3, img_size, img_size) in [0, 1]."""
        h = self.fc(z)
        h = h.view(-1, *self.initial_shape)
        return self.net(h)


# ═══════════════════════════════════════════════════════════
# Full autoencoder
# ═══════════════════════════════════════════════════════════

class GatedAutoencoder(nn.Module):
    """Autoencoder with gated encoder. Loss = reconstruction + λ * Σ α_ℓ (compute penalty).

    Args:
        img_size: Input image size (H=W).
        latent_dim: Bottleneck dimension.
        base_channels: Encoder/decoder base channels.
        n_blocks_per_stage: Gated blocks per encoder stage.
        n_stages: Encoder/decoder stages.
        gated: If True, encoder uses GatedResBlocks and returns gate list.
        gate_init_bias: Passed to gated blocks.
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 64,
        base_channels: int = 32,
        n_blocks_per_stage: int = 2,
        n_stages: int = 4,
        gated: bool = True,
        gate_init_bias: float = 2.0,
    ):
        super().__init__()
        self.encoder = GatedEncoder(
            img_size=img_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            n_blocks_per_stage=n_blocks_per_stage,
            n_stages=n_stages,
            gated=gated,
            gate_init_bias=gate_init_bias,
        )
        self.decoder = Decoder(
            img_size=img_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            n_stages=n_stages,
        )
        self.gated = gated

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Returns (x_recon, z, gates). gates is list of per-layer gate tensors shape (B,)."""
        z, gates = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, gates

    def compute_loss(
        self,
        x: torch.Tensor,
        gate_penalty: float = 0.01,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Returns total loss and a dict of diagnostics.
        """
        x_recon, z, gates = self.forward(x)

        # Reconstruction loss
        recon_loss = F.mse_loss(x_recon, x)

        # Gate penalty: encourage sparsity
        if self.gated and len(gates) > 0:
            gate_sum = sum(g.mean() for g in gates)
            gate_loss = gate_penalty * gate_sum
            n_layers = len(gates)
            gate_mean = gate_sum.item() / n_layers
        else:
            gate_loss = torch.tensor(0.0, device=x.device)
            gate_mean = 1.0
            n_layers = 0

        total_loss = recon_loss + gate_loss

        diagnostics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'gate_loss': gate_loss.item(),
            'mean_gate': gate_mean,
            'effective_depth': gate_sum.item() if self.gated and len(gates) > 0 else float(n_layers),
            'gate_values': [g.mean().item() for g in gates] if gates else [],
        }
        return total_loss, diagnostics

    @property
    def n_gated_layers(self) -> int:
        if self.gated:
            return self.encoder.n_stages * self.encoder.n_blocks_per_stage
        return 0
