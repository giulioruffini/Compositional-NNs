"""
Symmetry-Gated ResNet — v3
===========================
Key fix: The non-gated parts of the encoder (stem, downsampling convs)
are now MINIMAL — just channel projections and spatial pooling.
ALL feature extraction capacity lives in the gated residual blocks.
This forces the model to keep gates open when processing is needed.

Architecture:
  - Stem: 1×1 conv (channel projection only, no spatial features)
  - Gated blocks: 3×3 convs (the ONLY spatial processing)
  - Downsampling: AvgPool2d + 1×1 conv (no spatial features)

Gate:
  α_ℓ = σ(W · [AvgPool(h_ℓ), StdPool(h_ℓ)] + b)

Author: G. Ruffini / Technical Note companion code — v3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# Gate module
# ═══════════════════════════════════════════════════════════

class LayerGate(nn.Module):
    """
    α_ℓ = σ(W · [AvgPool(h_ℓ), StdPool(h_ℓ)] + b)

    Spatial std gives the gate a direct signal of geometric complexity.
    """
    def __init__(self, channels: int, init_bias: float = 2.0):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2 * channels, 1)
        nn.init.zeros_(self.fc.weight)
        nn.init.constant_(self.fc.bias, init_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.pool(x).flatten(1)             # (B, C)
        std = x.flatten(2).std(dim=2)             # (B, C)
        features = torch.cat([avg, std], dim=1)   # (B, 2C)
        gate = torch.sigmoid(self.fc(features))   # (B, 1)
        return gate.view(-1, 1, 1, 1)


# ═══════════════════════════════════════════════════════════
# Gated residual block (unchanged from v2)
# ═══════════════════════════════════════════════════════════

class GatedResBlock(nn.Module):
    """h' = h + α(h) · Δ(h). The ONLY layers with 3×3 convolutions."""
    def __init__(self, channels: int, gate_init_bias: float = 2.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gate = LayerGate(channels, init_bias=gate_init_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))
        alpha = self.gate(x)
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
# Encoder — v3: minimal non-gated capacity
# ═══════════════════════════════════════════════════════════

class GatedEncoder(nn.Module):
    """
    Encoder with MINIMAL non-gated capacity.

    - Stem: 1×1 conv (channel projection, no spatial features)
    - Gated blocks: 3×3 convs (ALL spatial processing here)
    - Downsampling: AvgPool + 1×1 conv (no 3×3 spatial features)

    This forces the gated blocks to be the primary feature extractors.
    When gates close, the model loses spatial processing power.
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

        # MINIMAL stem: 1×1 conv only (no spatial features!)
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 1, bias=False),  # 1×1 only!
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )

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

            if stage < n_stages - 1:
                next_ch = ch * 2
                # MINIMAL downsample: AvgPool + 1×1 conv (no spatial features!)
                self.downsamples.append(nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(ch, next_ch, 1, bias=False),  # 1×1 only!
                    nn.BatchNorm2d(next_ch),
                    nn.ReLU(),
                ))
                ch = next_ch
            else:
                self.downsamples.append(nn.Identity())

        self.final_channels = ch
        spatial = img_size
        for _ in range(n_stages - 1):
            spatial = spatial // 2
        self.final_spatial = spatial

        self.fc = nn.Linear(ch * spatial * spatial, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
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


# ═══════════════════════════════════════════════════════════
# Decoder
# ═══════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    MINIMAL decoder: latent → reconstructed image.

    Uses bilinear upsampling + 1×1 conv for channel reduction.
    NO 3×3 convolutions — the decoder cannot compensate for a weak
    encoder. This forces the encoder's gated 3×3 blocks to extract
    good spatial features.
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 64,
        base_channels: int = 32,
        n_stages: int = 4,
    ):
        super().__init__()

        final_ch = base_channels * (2 ** (n_stages - 1))
        spatial = img_size
        for _ in range(n_stages - 1):
            spatial = spatial // 2

        self.fc = nn.Linear(latent_dim, final_ch * spatial * spatial)
        self.initial_shape = (final_ch, spatial, spatial)
        self.n_stages = n_stages

        # 1×1 channel reduction at each upsampling stage
        self.channel_reduce = nn.ModuleList()
        ch = final_ch
        for stage in range(n_stages - 1):
            next_ch = ch // 2
            self.channel_reduce.append(nn.Sequential(
                nn.Conv2d(ch, next_ch, 1, bias=False),  # 1×1 only!
                nn.BatchNorm2d(next_ch),
                nn.ReLU(),
            ))
            ch = next_ch

        # Final 1×1 to RGB
        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, 3, 1),  # 1×1 only!
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(-1, *self.initial_shape)

        for reduce in self.channel_reduce:
            h = F.interpolate(h, scale_factor=2, mode='bilinear',
                              align_corners=False)
            h = reduce(h)

        return self.to_rgb(h)


# ═══════════════════════════════════════════════════════════
# Full autoencoder
# ═══════════════════════════════════════════════════════════

class GatedAutoencoder(nn.Module):
    """Autoencoder with gated encoder. Loss = MSE + λ·Σα_ℓ."""
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

    def forward(self, x: torch.Tensor):
        z, gates = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, gates

    def compute_loss(
        self,
        x: torch.Tensor,
        gate_penalty: float = 0.01,
        n_levels: Optional[torch.Tensor] = None,
        max_levels: int = 7,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Loss = MSE + λ_eff · Σ_ℓ α_ℓ

        If n_levels is provided (per-sample complexity), the gate penalty
        is scaled: λ_eff(sample) = λ · (max_levels - n_levels + 1) / max_levels.
        Simpler inputs get stronger penalty → encourages them to close gates.
        Complex inputs get weaker penalty → allows them to keep gates open.
        """
        x_recon, z, gates = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)

        if self.gated and len(gates) > 0:
            # gates[ℓ] has shape (B,) — per-sample gate values
            gate_stack = torch.stack(gates, dim=1)  # (B, L)
            per_sample_sum = gate_stack.sum(dim=1)  # (B,)

            if n_levels is not None:
                # Per-sample scaling: simpler → stronger penalty
                n_lev = n_levels.float().to(x.device)
                scale = (max_levels - n_lev + 1) / max_levels  # (B,)
                # scale ranges from 1/7 (Everything) to 8/7 (Static)
                weighted_sum = (scale * per_sample_sum).mean()
            else:
                weighted_sum = per_sample_sum.mean()

            gate_loss = gate_penalty * weighted_sum
            n_layers = len(gates)
            gate_mean = per_sample_sum.mean().item() / n_layers
        else:
            gate_loss = torch.tensor(0.0, device=x.device)
            gate_mean = 1.0
            n_layers = 0
            per_sample_sum = torch.zeros(x.size(0), device=x.device)

        total_loss = recon_loss + gate_loss

        diagnostics = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'gate_loss': gate_loss.item(),
            'mean_gate': gate_mean,
            'effective_depth': per_sample_sum.mean().item() if self.gated and len(gates) > 0 else float(n_layers),
            'gate_values': [g.mean().item() for g in gates] if gates else [],
        }
        return total_loss, diagnostics

    @property
    def n_gated_layers(self) -> int:
        if self.gated:
            return self.encoder.n_stages * self.encoder.n_blocks_per_stage
        return 0
