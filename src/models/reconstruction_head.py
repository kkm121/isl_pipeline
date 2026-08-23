"""
=============================================================================
BharatSRM-Net v4: Reconstruction Head (Sub-Pixel Super-Resolution)
=============================================================================
Formulation:
  PixelShuffle(s=4) upscale: (B, C * s^2, H, W) -> (B, C, 4H, 4W)
  Outputs 4 native Sentinel-2 bands: Red (B4), Green (B3), Blue (B2), NIR (B8)
  Nominal Ground Sampling Distance: 10m / 4 = 2.5m grid (<4m target requirement).
=============================================================================
"""

import torch
from torch import nn


class ReconstructionHead(nn.Module):
    """Sub-pixel super-resolution reconstruction head producing 4-band RGBN imagery."""

    def __init__(
        self,
        in_channels: int = 64,
        out_bands: int = 4,
        scale_factor: int = 4,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.out_bands = out_bands

        # Deep refinement layers prior to upscaling
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, hidden_dim), num_channels=hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # PixelShuffle Projection: Maps hidden_dim -> (out_bands * scale_factor^2)
        # For scale_factor=4, scale_factor^2 = 16. For 4 bands: 4 * 16 = 64 channels.
        self.pixel_shuffle_proj = nn.Conv2d(
            hidden_dim,
            out_bands * (scale_factor**2),
            kernel_size=3,
            padding=1,
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # High-resolution spectral polish layer
        self.hr_polish = nn.Sequential(
            nn.Conv2d(out_bands, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_bands, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Normalized surface reflectance bounded in [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Fused feature representation of shape (B, in_channels, H, W)

        Returns:
            sr_image: Reconstructed 4-band HR image of shape (B, 4, 4H, 4W)
        """
        x = self.refinement(features)
        x = self.pixel_shuffle_proj(x)
        x = self.pixel_shuffle(x)  # (B, out_bands, 4H, 4W)
        sr_image = self.hr_polish(x)
        return sr_image
