"""
Symmetry-Gated ResNet — v5
===========================
Fully convolutional autoencoder with binary multiplicative gates.

Key insight: earlier versions had an FC bottleneck (encoder FC → latent vector
→ decoder FC → spatial map). The decoder FC could decode rich spatial maps
from a flat (non-spatial) code, circumventing the gates entirely.

v5 fix: FULLY CONVOLUTIONAL — no FC layers. The latent is a spatial feature
map (B, C_latent, S, S). When gates destroy spatial info → encoder outputs
constant maps → decoder (bilinear + 1×1 only) CANNOT recreate spatial detail.
The model MUST keep gates open to preserve spatial features.

Binary gates via Gumbel-Softmax force discrete ON/OFF decisions.
Gate penalty acts as a "cost per layer" budget.
Complex images need more layers → higher D_eff.

Architecture:
  Encoder:  1×1 stem → [gated 3×3 blocks + AvgPool+1×1 downsample]×N → 1×1 to latent
  Latent:   (B, latent_channels, S, S) — spatial feature map, not vector
  Decoder:  1×1 from latent → [bilinear ×2 + 1×1]×N → 1×1 to RGB

Gate:
  logit_ℓ = MLP([AvgPool(h_ℓ), StdPool(h_ℓ)])
  α_ℓ ∈ {0, 1} via Gumbel-Softmax
  ON:  h' = h + Δ(h)    — full residual with spatial features
  OFF: h' = μ(h)         — only channel means, spatial info destroyed

Author: G. Ruffini / Technical Note companion code — v5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# Gumbel-Softmax utilities
# ═══════════════════════════════════════════════════════════

def gumbel_sigmoid(logit: torch.Tensor, tau: float = 1.0, hard: bool = True) -> torch.Tensor:
    """
    Binary Gumbel-Softmax (Gumbel-Bernoulli).

    Samples from Bernoulli(σ(logit)) using reparametrization trick.
    """
    if not logit.requires_grad or not torch.is_grad_enabled():
        # Eval mode: hard threshold
        return (logit > 0.0).float()

    u = torch.rand_like(logit).clamp(1e-6, 1 - 1e-6)
    gumbel = torch.log(u) - torch.log(1 - u)
    y_soft = torch.sigmoid((logit + gumbel) / tau)

    if hard:
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


# ═══════════════════════════════════════════════════════════
# Gate module — binary via Gumbel-Softmax
# ═══════════════════════════════════════════════════════════

class LayerGate(nn.Module):
    """
    Binary gate: α_ℓ ∈ {0, 1} via Gumbel-Softmax.

    Input: [AvgPool(h), StdPool(h)] → MLP → logit → Gumbel-Bernoulli
    """
    def __init__(self, channels: int, init_bias: float = 0.0, tau: float = 1.0):
        super().__init__()
        self.tau = tau
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.gate_mlp = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.ReLU(),
            nn.Linear(channels, 1),
        )

        # Xavier init so gate CAN learn from features
        nn.init.xavier_normal_(self.gate_mlp[0].weight)
        nn.init.zeros_(self.gate_mlp[0].bias)
        nn.init.xavier_normal_(self.gate_mlp[2].weight)
        nn.init.constant_(self.gate_mlp[2].bias, init_bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        avg = self.pool(x).flatten(1)
        std = x.flatten(2).std(dim=2)
        features = torch.cat([avg, std], dim=1)

        logit = self.gate_mlp(features)
        logit_4d = logit.view(-1, 1, 1, 1)

        gate = gumbel_sigmoid(logit_4d, tau=self.tau, hard=True)
        prob = torch.sigmoid(logit).squeeze(-1)
        return gate, prob


# ═══════════════════════════════════════════════════════════
# Gated residual block — binary multiplicative gating
# ═══════════════════════════════════════════════════════════

class GatedResBlock(nn.Module):
    """
    Binary multiplicative gating.

    ON  (α=1):  h' = ReLU(h + Δ(h))     — full residual block
    OFF (α=0):  h' = ReLU(μ(h))          — spatial info destroyed
    """
    def __init__(self, channels: int, gate_init_bias: float = 0.0, gate_tau: float = 1.0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.gate = LayerGate(channels, init_bias=gate_init_bias, tau=gate_tau)
        self.spatial_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = F.relu(self.bn1(self.conv1(x)))
        residual = self.bn2(self.conv2(residual))

        gate_binary, gate_prob = self.gate(x)

        h_full = x + residual
        h_mean = self.spatial_pool(x).expand_as(x)

        out = gate_binary * h_full + (1.0 - gate_binary) * h_mean
        out = F.relu(out)

        return out, gate_prob


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
# Fully Convolutional Encoder
# ═══════════════════════════════════════════════════════════

class GatedEncoder(nn.Module):
    """
    Fully convolutional encoder — NO FC layers.

    Spatial information flows through spatial feature maps.
    No FC to circumvent gates.

    Output: (B, latent_channels, S, S) spatial latent.
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 8,      # now latent CHANNELS (spatial)
        base_channels: int = 32,
        n_blocks_per_stage: int = 2,
        n_stages: int = 4,
        gated: bool = True,
        gate_init_bias: float = 0.0,
        gate_tau: float = 1.0,
    ):
        super().__init__()
        self.gated = gated
        self.n_stages = n_stages
        self.n_blocks_per_stage = n_blocks_per_stage

        # MINIMAL stem: 1×1 conv only
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, 1, bias=False),
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
                    blocks.append(GatedResBlock(ch, gate_init_bias, gate_tau))
                else:
                    blocks.append(ResBlock(ch))
            self.stages.append(blocks)

            if stage < n_stages - 1:
                next_ch = ch * 2
                self.downsamples.append(nn.Sequential(
                    nn.AvgPool2d(2),
                    nn.Conv2d(ch, next_ch, 1, bias=False),
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

        # 1×1 conv to latent channels (NO FC!)
        self.to_latent = nn.Sequential(
            nn.Conv2d(ch, latent_dim, 1, bias=False),
            nn.BatchNorm2d(latent_dim),
        )
        self.latent_dim = latent_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        gate_probs = []
        h = self.stem(x)

        for stage_idx, (blocks, down) in enumerate(
            zip(self.stages, self.downsamples)
        ):
            for block in blocks:
                if self.gated:
                    h, prob = block(h)
                    gate_probs.append(prob)
                else:
                    h = block(h)

            if stage_idx < self.n_stages - 1:
                h = down(h)

        z = self.to_latent(h)  # (B, latent_ch, S, S) — spatial!
        return z, gate_probs


# ═══════════════════════════════════════════════════════════
# Fully Convolutional Decoder
# ═══════════════════════════════════════════════════════════

class Decoder(nn.Module):
    """
    Fully convolutional decoder — NO FC layers.

    latent (B, latent_ch, S, S) → bilinear upsample + 1×1 conv → RGB

    Because there's no FC, the decoder CANNOT create spatial detail
    that wasn't present in the encoder output. It can only:
    - Upsample (bilinear interpolation — smooth, no new detail)
    - Mix channels (1×1 conv — per-pixel, no spatial processing)
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 8,      # latent channels
        base_channels: int = 32,
        n_stages: int = 4,
    ):
        super().__init__()

        final_ch = base_channels * (2 ** (n_stages - 1))
        self.n_stages = n_stages

        # 1×1 from latent channels to decoder channels (NO FC!)
        self.from_latent = nn.Sequential(
            nn.Conv2d(latent_dim, final_ch, 1, bias=False),
            nn.BatchNorm2d(final_ch),
            nn.ReLU(),
        )

        self.channel_reduce = nn.ModuleList()
        ch = final_ch
        for stage in range(n_stages - 1):
            next_ch = ch // 2
            self.channel_reduce.append(nn.Sequential(
                nn.Conv2d(ch, next_ch, 1, bias=False),
                nn.BatchNorm2d(next_ch),
                nn.ReLU(),
            ))
            ch = next_ch

        self.to_rgb = nn.Sequential(
            nn.Conv2d(ch, 3, 1),
            nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)

        for reduce in self.channel_reduce:
            h = F.interpolate(h, scale_factor=2, mode='bilinear',
                              align_corners=False)
            h = reduce(h)

        return self.to_rgb(h)


# ═══════════════════════════════════════════════════════════
# Full autoencoder
# ═══════════════════════════════════════════════════════════

class GatedAutoencoder(nn.Module):
    """
    Fully convolutional autoencoder with binary-gated encoder.

    Loss = MSE + λ · mean(Σ_ℓ prob_ℓ)
    """
    def __init__(
        self,
        img_size: int = 128,
        latent_dim: int = 8,
        base_channels: int = 32,
        n_blocks_per_stage: int = 2,
        n_stages: int = 4,
        gated: bool = True,
        gate_init_bias: float = 0.0,
        gate_tau: float = 1.0,
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
            gate_tau=gate_tau,
        )
        self.decoder = Decoder(
            img_size=img_size,
            latent_dim=latent_dim,
            base_channels=base_channels,
            n_stages=n_stages,
        )
        self.gated = gated

    def forward(self, x: torch.Tensor):
        z, gate_probs = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z, gate_probs

    def compute_loss(
        self,
        x: torch.Tensor,
        gate_penalty: float = 0.01,
        n_levels: Optional[torch.Tensor] = None,
        max_levels: int = 7,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Loss = MSE + λ · mean(Σ_ℓ prob_ℓ)

        Penalty on gate probabilities (smooth) while forward uses hard gates.
        """
        x_recon, z, gate_probs = self.forward(x)
        recon_loss = F.mse_loss(x_recon, x)

        if self.gated and len(gate_probs) > 0:
            prob_stack = torch.stack(gate_probs, dim=1)
            per_sample_sum = prob_stack.sum(dim=1)

            gate_loss = gate_penalty * per_sample_sum.mean()

            n_layers = len(gate_probs)
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
            'effective_depth': per_sample_sum.mean().item() if self.gated and len(gate_probs) > 0 else float(n_layers),
            'gate_values': [g.mean().item() for g in gate_probs] if gate_probs else [],
        }
        return total_loss, diagnostics

    @property
    def n_gated_layers(self) -> int:
        if self.gated:
            return self.encoder.n_stages * self.encoder.n_blocks_per_stage
        return 0
