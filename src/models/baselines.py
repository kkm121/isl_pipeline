"""
=============================================================================
BharatSRM-Net v4: Benchmark Baselines & Ablation Models
=============================================================================
Architectures:
  1. BicubicSR: Bicubic interpolation baseline (s=4) extracting native bands.
  2. EDSRBaseline: Enhanced Deep Residual Network for 10-band -> 4-band SR (s=4).
  3. SRResNetBaseline: Standard SRResNet with BatchNorm + PReLU + PixelShuffle (s=4).
  4. A0_BaseNet: Ablation A0 (Single-stream dilated encoder, standard Conv2d, no context, no masking).
  5. A1_ContextNet: Ablation A1 (BaseNet + Context stream + AC-FEM fusion + degradation loss).
=============================================================================
"""

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .ac_fem import ACFEM
from .encoder import ContextEncoder, DilatedResidualBlock, LightweightWindowAttention
from .reconstruction_head import ReconstructionHead


class BicubicSR(nn.Module):
    """
    Bicubic Interpolation Baseline for Satellite Super-Resolution.
    Upsamples 10m input bands to 2.5m nominal GSD (scale factor s=4).
    Extracts the 4 primary RGBN bands [B4, B3, B2, B8] or first 4 bands.
    """

    def __init__(
        self,
        in_bands: int = 10,
        out_bands: int = 4,
        scale_factor: int = 4,
        band_indices: list[int] | None = None,
    ):
        super().__init__()
        self.in_bands = in_bands
        self.out_bands = out_bands
        self.scale_factor = scale_factor
        self.band_indices = band_indices

    def forward(
        self,
        x_spectral: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        context_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for BicubicSR.

        Args:
            x_spectral: (B, C, H, W) Sentinel-2 multispectral tensor
            validity_mask: Ignored (kept for pipeline API uniformity)
            context_dem: Ignored (kept for pipeline API uniformity)

        Returns:
            Dict containing:
                - 'sr_image': (B, out_bands, s*H, s*W) Reconstructed HR image
                - 'fused_features': (B, out_bands, H, W) Extracted LR sub-bands
        """
        if self.band_indices is not None:
            x_sub = x_spectral[:, self.band_indices, :, :]
        elif x_spectral.size(1) >= self.out_bands:
            x_sub = x_spectral[:, : self.out_bands, :, :]
        else:
            x_sub = x_spectral

        sr_image = F.interpolate(
            x_sub,
            scale_factor=self.scale_factor,
            mode="bicubic",
            align_corners=False,
        )
        sr_image = torch.clamp(sr_image, 0.0, 1.0)

        return {
            "sr_image": sr_image,
            "fused_features": x_sub,
        }


class EDSRResBlock(nn.Module):
    """
    Enhanced Deep Residual Block without Batch Normalization (Lim et al. 2017).
    Consists of Conv2d -> ReLU -> Conv2d with residual scaling.
    """

    def __init__(self, n_feats: int = 64, kernel_size: int = 3, res_scale: float = 1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(
            n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.conv2(self.relu(self.conv1(x)))
        return x + res * self.res_scale


class EDSRBaseline(nn.Module):
    """
    Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR).
    Adapted for 10-band Sentinel-2 multispectral input -> 4-band RGBN output at 4x resolution.
    Removes Batch Normalization layers to preserve absolute radiometric scale.
    """

    def __init__(
        self,
        in_spectral_bands: int = 10,
        out_sr_bands: int = 4,
        scale_factor: int = 4,
        n_feats: int = 64,
        n_resblocks: int = 8,
        res_scale: float = 1.0,
    ):
        super().__init__()
        self.in_spectral_bands = in_spectral_bands
        self.out_sr_bands = out_sr_bands
        self.scale_factor = scale_factor

        # Input feature projection
        self.head = nn.Conv2d(in_spectral_bands, n_feats, kernel_size=3, padding=1)

        # Deep residual trunk
        self.body = nn.Sequential(
            *[
                EDSRResBlock(n_feats=n_feats, kernel_size=3, res_scale=res_scale)
                for _ in range(n_resblocks)
            ]
        )
        self.body_tail = nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1)

        # Sub-pixel upsampler (s=4)
        self.upsampler = nn.Sequential(
            nn.Conv2d(n_feats, n_feats * (scale_factor**2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.ReLU(inplace=True),
        )

        # High-resolution output tail
        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_sr_bands, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_spectral: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        context_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for EDSR baseline.
        """
        f_head = self.head(x_spectral)
        f_body = self.body_tail(self.body(f_head)) + f_head
        f_up = self.upsampler(f_body)
        sr_image = self.tail(f_up)

        return {
            "sr_image": sr_image,
            "fused_features": f_body,
        }


class SRResNetBlock(nn.Module):
    """
    Standard SRResNet Residual Block (Ledig et al. 2017).
    Conv2d -> BatchNorm2d -> PReLU -> Conv2d -> BatchNorm2d + skip.
    """

    def __init__(self, n_feats: int = 64, kernel_size: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(n_feats),
            nn.PReLU(),
            nn.Conv2d(
                n_feats,
                n_feats,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(n_feats),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class SRResNetBaseline(nn.Module):
    """
    Super-Resolution Residual Network (SRResNet) Baseline.
    Architecture:
      Conv2d + PReLU -> N Residual Blocks with BatchNorm -> Conv2d + BatchNorm + Skip
      -> PixelShuffle upsampling stages -> Conv2d + Sigmoid.
    """

    def __init__(
        self,
        in_spectral_bands: int = 10,
        out_sr_bands: int = 4,
        scale_factor: int = 4,
        n_feats: int = 64,
        n_resblocks: int = 8,
    ):
        super().__init__()
        self.in_spectral_bands = in_spectral_bands
        self.out_sr_bands = out_sr_bands
        self.scale_factor = scale_factor

        self.head = nn.Sequential(
            nn.Conv2d(in_spectral_bands, n_feats, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        self.body = nn.Sequential(
            *[SRResNetBlock(n_feats=n_feats, kernel_size=3) for _ in range(n_resblocks)]
        )

        self.body_tail = nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_feats),
        )

        # 4x PixelShuffle upsampler
        if scale_factor == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
                nn.Conv2d(n_feats, n_feats * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.PReLU(),
            )
        else:
            self.upsampler = nn.Sequential(
                nn.Conv2d(
                    n_feats, n_feats * (scale_factor**2), kernel_size=3, padding=1
                ),
                nn.PixelShuffle(scale_factor),
                nn.PReLU(),
            )

        self.tail = nn.Sequential(
            nn.Conv2d(n_feats, out_sr_bands, kernel_size=9, padding=4),
            nn.Sigmoid(),
        )

    def forward(
        self,
        x_spectral: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        context_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for SRResNet baseline.
        """
        f_head = self.head(x_spectral)
        f_body = self.body_tail(self.body(f_head)) + f_head
        f_up = self.upsampler(f_body)
        sr_image = self.tail(f_up)

        return {
            "sr_image": sr_image,
            "fused_features": f_body,
        }


class A0_BaseNet(nn.Module):
    """
    Ablation A0: Single-stream multispectral encoder with dilated convolutions.
    Features:
      - Standard Conv2d (NO PartialConv2d cloud masking).
      - NO Context stream (no CartoDEM elevation/slope).
      - NO AC-FEM fusion module.
      - NO Uncertainty head (plain reconstruction only).
      - Dilated Residual Blocks (rates {1, 2, 4, 8}) + optional Window Attention.
      - Sub-pixel ReconstructionHead (s=4).
    """

    def __init__(
        self,
        in_spectral_bands: int = 10,
        out_sr_bands: int = 4,
        scale_factor: int = 4,
        base_channels: int = 64,
        dilation_rates: list[int] | None = None,
        use_window_attention: bool = True,
    ):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        self.in_spectral_bands = in_spectral_bands
        self.out_sr_bands = out_sr_bands
        self.scale_factor = scale_factor
        self.use_window_attention = use_window_attention

        # Standard conv layers without partial convolutions
        self.in_conv1 = nn.Conv2d(
            in_spectral_bands, base_channels, kernel_size=3, padding=1
        )
        self.in_conv2 = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, padding=1
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.dilated_blocks = nn.ModuleList(
            [DilatedResidualBlock(base_channels, dilation=r) for r in dilation_rates]
        )

        if use_window_attention:
            self.attn = LightweightWindowAttention(
                base_channels, num_heads=4, window_size=8
            )
        else:
            self.attn = nn.Identity()

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                base_channels, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.reconstruction_head = ReconstructionHead(
            in_channels=base_channels,
            out_bands=out_sr_bands,
            scale_factor=scale_factor,
            hidden_dim=128,
        )

    def forward(
        self,
        x_spectral: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        context_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for Ablation A0.
        """
        feat = self.relu(self.in_conv1(x_spectral))
        feat = self.relu(self.in_conv2(feat))

        for block in self.dilated_blocks:
            feat = block(feat)

        if self.use_window_attention:
            feat = self.attn(feat)

        f_spec = self.out_proj(feat)
        sr_image = self.reconstruction_head(f_spec)

        updated_mask = (
            validity_mask
            if validity_mask is not None
            else torch.ones(
                x_spectral.size(0),
                1,
                x_spectral.size(2),
                x_spectral.size(3),
                device=x_spectral.device,
                dtype=x_spectral.dtype,
            )
        )

        return {
            "sr_image": sr_image,
            "fused_features": f_spec,
            "updated_mask": updated_mask,
        }


class A1_ContextNet(nn.Module):
    """
    Ablation A1: BaseNet + Context Stream + AC-FEM Fusion Module + Degradation Loss.
    Features:
      - Standard Conv2d encoder (NO PartialConv2d cloud masking).
      - Dual-stream ContextEncoder (CartoDEM elevation & slope).
      - AC-FEM cross-stream fusion module.
      - Sub-pixel ReconstructionHead (s=4).
    """

    def __init__(
        self,
        in_spectral_bands: int = 10,
        out_sr_bands: int = 4,
        scale_factor: int = 4,
        base_channels: int = 64,
        context_channels: int = 32,
        dilation_rates: list[int] | None = None,
        use_window_attention: bool = True,
    ):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        self.in_spectral_bands = in_spectral_bands
        self.out_sr_bands = out_sr_bands
        self.scale_factor = scale_factor
        self.use_window_attention = use_window_attention

        # Standard conv layers without partial convolutions
        self.in_conv1 = nn.Conv2d(
            in_spectral_bands, base_channels, kernel_size=3, padding=1
        )
        self.in_conv2 = nn.Conv2d(
            base_channels, base_channels, kernel_size=3, padding=1
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        self.dilated_blocks = nn.ModuleList(
            [DilatedResidualBlock(base_channels, dilation=r) for r in dilation_rates]
        )

        if use_window_attention:
            self.attn = LightweightWindowAttention(
                base_channels, num_heads=4, window_size=8
            )
        else:
            self.attn = nn.Identity()

        self.out_proj = nn.Sequential(
            nn.Conv2d(
                base_channels, base_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Context stream encoder (CartoDEM elevation & slope)
        self.context_encoder = ContextEncoder(
            in_channels=2, out_channels=context_channels
        )

        # AC-FEM Feature Fusion Module
        self.ac_fem = ACFEM(
            spec_channels=base_channels,
            prior_channels=context_channels,
            out_channels=base_channels,
        )

        # Sub-pixel reconstruction head
        self.reconstruction_head = ReconstructionHead(
            in_channels=base_channels,
            out_bands=out_sr_bands,
            scale_factor=scale_factor,
            hidden_dim=128,
        )

    def forward(
        self,
        x_spectral: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        context_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass for Ablation A1.
        """
        feat = self.relu(self.in_conv1(x_spectral))
        feat = self.relu(self.in_conv2(feat))

        for block in self.dilated_blocks:
            feat = block(feat)

        if self.use_window_attention:
            feat = self.attn(feat)

        f_spec = self.out_proj(feat)

        if context_dem is None:
            context_dem = torch.zeros(
                x_spectral.size(0),
                2,
                x_spectral.size(2),
                x_spectral.size(3),
                device=x_spectral.device,
                dtype=x_spectral.dtype,
            )

        f_prior = self.context_encoder(context_dem)
        f_fused = self.ac_fem(f_spec, f_prior, validity_mask=validity_mask)

        sr_image = self.reconstruction_head(f_fused)

        return {
            "sr_image": sr_image,
            "fused_features": f_fused,
            "spectral_features": f_spec,
            "context_features": f_prior,
        }
