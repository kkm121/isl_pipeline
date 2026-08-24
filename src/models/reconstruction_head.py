"""
=============================================================================
BharatSRM-Net v4: High-Fidelity Reconstruction Head (Sub-Pixel Super-Resolution)
=============================================================================
Formulation:
  PixelShuffle(s=4) upscale with ICNR sub-pixel anti-aliasing initialization.
  Global Residual Learning:
    \hat{y} = \text{Bicubic}(x_{\text{LRRGBN}}) + \Delta y_{\text{residual}}
  Outputs 4 native Sentinel-2 bands: Red (B4), Green (B3), Blue (B2), NIR (B8)
  Nominal Ground Sampling Distance: 10m / 4 = 2.5m grid (<4m target requirement).
=============================================================================
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Channel Attention for multi-spectral band gating."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(2, channels // reduction), kernel_size=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(2, channels // reduction), channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


class HighFrequencyPolishBlock(nn.Module):
    """Multi-scale residual refinement block preserving fine edge transitions without periodic phase noise."""

    def __init__(self, channels: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.ca = ChannelAttention(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.relu(self.conv1(x))
        res = self.relu(self.conv2(res))
        res = self.ca(res)
        res = self.conv3(res)
        return x + res


def icnr_init(conv_weight: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
    """ICNR (Initialization to Convolution Transpose) to eliminate sub-pixel checkerboard/dot artifacts."""
    out_channels, in_channels, kh, kw = conv_weight.shape
    sub_channels = max(1, out_channels // (scale_factor**2))
    sub_weight = nn.init.kaiming_normal_(torch.empty(sub_channels, in_channels, kh, kw))
    repeated = sub_weight.repeat_interleave(scale_factor**2, dim=0)
    return repeated[:out_channels]


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
        self.pixel_shuffle_proj = nn.Conv2d(
            hidden_dim,
            out_bands * (scale_factor**2),
            kernel_size=3,
            padding=1,
        )
        # Apply ICNR initialization to prevent sub-pixel dot patterns
        with torch.no_grad():
            self.pixel_shuffle_proj.weight.copy_(icnr_init(self.pixel_shuffle_proj.weight, scale_factor))

        self.pixel_shuffle = nn.PixelShuffle(scale_factor)

        # High-resolution multi-scale residual polish block
        self.hr_polish = HighFrequencyPolishBlock(channels=out_bands, hidden_dim=32)

    def forward(
        self, features: torch.Tensor, lr_base: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            features: Fused feature representation of shape (B, in_channels, H, W)
            lr_base: Optional (B, 4, H, W) low-res RGBN input for global residual learning

        Returns:
            sr_image: Reconstructed 4-band HR image of shape (B, 4, 4H, 4W)
        """
        x = self.refinement(features)
        x = self.pixel_shuffle_proj(x)
        res = self.pixel_shuffle(x)  # (B, out_bands, 4H, 4W)
        res = self.hr_polish(res)

        if lr_base is not None:
            # Global residual connection over bicubic base with zero-mean channel centering
            base_hr = F.interpolate(
                lr_base,
                scale_factor=self.scale_factor,
                mode="bicubic",
                align_corners=False,
            )
            # Center residual per-channel to guarantee ZERO color cast / spectral drift
            res_centered = res - torch.mean(res, dim=(-2, -1), keepdim=True)
            sr_image = torch.clamp(base_hr + 0.25 * torch.tanh(res_centered), 0.0, 1.0)
        else:
            sr_image = torch.sigmoid(res)

        return sr_image
