r"""
=============================================================================
BharatSRM-Net v4: Heteroscedastic Uncertainty Head
=============================================================================
Mathematical Formulation:
  s_i = \log \sigma_i^2 (per-pixel, per-band log variance)
  Loss: L_conf = (1/N) \sum_i [ \exp(-s_i) * ||y_i - \hat{y}_i||^2 + s_i ]

Role:
  Jointly predicts spatial uncertainty alongside super-resolution output,
  flagging unobserved or inferred fine details (e.g. under clouds/shadows or high-frequency edges).
=============================================================================
"""

import torch
from torch import nn


class UncertaintyHead(nn.Module):
    r"""Predicts per-pixel, per-band log-variance uncertainty map (s = log \sigma^2)."""

    def __init__(
        self,
        in_channels: int = 64,
        out_bands: int = 4,
        scale_factor: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.out_bands = out_bands

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, out_bands * (scale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),  # (B, out_bands, 4H, 4W)
            nn.Conv2d(out_bands, out_bands, kernel_size=3, padding=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Fused feature representation of shape (B, in_channels, H, W)

        Returns:
            log_variance: Per-pixel, per-band log variance tensor of shape (B, 4, 4H, 4W)
        """
        log_variance = self.net(features)
        # Clamp for numerical stability during exponentiation in loss
        log_variance = torch.clamp(log_variance, min=-8.0, max=5.0)
        return log_variance

    @staticmethod
    def get_variance(log_variance: torch.Tensor) -> torch.Tensor:
        r"""Returns standard variance \sigma^2 = exp(s)."""
        return torch.exp(log_variance)

    @staticmethod
    def get_std(log_variance: torch.Tensor) -> torch.Tensor:
        r"""Returns standard deviation \sigma = exp(0.5 * s)."""
        return torch.exp(0.5 * log_variance)
