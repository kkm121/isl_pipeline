"""
=============================================================================
BharatSRM-Net v4: Master Neural Architecture
=============================================================================
Unified Architecture uniting:
  - Masked Multispectral Encoder (PartialConv2D + Dilated Residual Blocks + Window Attention)
  - Context Encoder (CartoDEM 30m Elevation & Slope)
  - AC-FEM (Adaptive Cloud-Aware Feature Enhancement Module)
  - Reconstruction Head (PixelShuffle s=4, 4-band RGBN output)
  - Heteroscedastic Uncertainty Head (Per-pixel, per-band log-variance)
  - Downstream Task Heads (Roads, LULC, Disaster Change Detection)
=============================================================================
"""


import torch
from torch import nn

from .ac_fem import ACFEM
from .downstream_heads import BuiltUpLULCHead, ChangeDamageHead, RuralRoadExtractionHead
from .encoder import ContextEncoder, MaskedMultispectralEncoder
from .reconstruction_head import ReconstructionHead
from .uncertainty_head import UncertaintyHead


class BharatSRMNetV4(nn.Module):
    """
    BharatSRM-Net v4: Physically-Consistent, Uncertainty-Aware Super-Resolution Network
    for Indian Medium-Resolution Satellite Imagery (SIH 2026 PS 26142 - NTRO).
    """

    def __init__(
        self,
        in_spectral_bands: int = 10,
        out_sr_bands: int = 4,
        scale_factor: int = 4,
        base_channels: int = 64,
        context_channels: int = 32,
        use_context_stream: bool = True,
        use_window_attention: bool = True,
        include_downstream_heads: bool = True,
    ):
        super().__init__()
        self.in_spectral_bands = in_spectral_bands
        self.out_sr_bands = out_sr_bands
        self.scale_factor = scale_factor
        self.use_context_stream = use_context_stream
        self.include_downstream_heads = include_downstream_heads

        # 1. Masked Multispectral Encoder
        self.spectral_encoder = MaskedMultispectralEncoder(
            in_channels=in_spectral_bands,
            base_channels=base_channels,
            use_window_attention=use_window_attention,
        )

        # 2. Context Encoder (CartoDEM elevation/slope)
        if use_context_stream:
            self.context_encoder = ContextEncoder(
                in_channels=2, out_channels=context_channels
            )
            # AC-FEM Feature Fusion Module
            self.ac_fem = ACFEM(
                spec_channels=base_channels,
                prior_channels=context_channels,
                out_channels=base_channels,
            )
        else:
            self.context_encoder = None
            self.ac_fem = None

        # 3. Super-Resolution Reconstruction Head (s=4)
        self.reconstruction_head = ReconstructionHead(
            in_channels=base_channels,
            out_bands=out_sr_bands,
            scale_factor=scale_factor,
            hidden_dim=128,
        )

        # 4. Heteroscedastic Uncertainty Head
        self.uncertainty_head = UncertaintyHead(
            in_channels=base_channels,
            out_bands=out_sr_bands,
            scale_factor=scale_factor,
            hidden_dim=64,
        )

        # 5. Downstream Heads
        if include_downstream_heads:
            # HR Feature upsampler for downstream decoders
            self.feat_upsampler = nn.Sequential(
                nn.Conv2d(base_channels, base_channels * (scale_factor**2), kernel_size=3, padding=1),
                nn.PixelShuffle(scale_factor),
                nn.LeakyReLU(0.2, inplace=True),
            )
            self.road_head = RuralRoadExtractionHead(in_channels=base_channels, image_bands=out_sr_bands)
            self.lulc_head = BuiltUpLULCHead(in_channels=base_channels, image_bands=out_sr_bands, num_classes=5)
            self.change_head = ChangeDamageHead(in_channels=base_channels, image_bands=out_sr_bands)
        else:
            self.feat_upsampler = None
            self.road_head = None
            self.lulc_head = None
            self.change_head = None

    def forward(
        self,
        x_spectral: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
        context_dem: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        r"""
        Forward pass of BharatSRM-Net v4.

        Args:
            x_spectral: (B, 10, H, W) Sentinel-2 multispectral tile
            validity_mask: (B, 1, H, W) S2cloudless + QA60 validity mask (1=valid, 0=cloud/shadow)
            context_dem: (B, 2, H, W) CartoDEM elevation/slope tensor (Optional)

        Returns:
            Dict containing:
                - 'sr_image': (B, 4, 4H, 4W) Super-resolved 4-band RGBN imagery
                - 'log_variance': (B, 4, 4H, 4W) Per-pixel, per-band log-variance
                - 'variance': (B, 4, 4H, 4W) \sigma^2 uncertainty map
                - 'std': (B, 4, 4H, 4W) Standard deviation map
                - 'fused_features': (B, 64, H, W) Latent feature representation
                - 'updated_mask': (B, 1, H, W) Updated validity mask
        """
        # Step 1: Extract spectral features under partial conv cloud masking
        f_spec, updated_mask = self.spectral_encoder(x_spectral, validity_mask)

        # Step 2: Context fusion via AC-FEM if context stream is active
        if self.use_context_stream and self.context_encoder is not None and self.ac_fem is not None:
            if context_dem is None:
                # Default zero context if DEM not supplied
                context_dem = torch.zeros(
                    x_spectral.size(0), 2, x_spectral.size(2), x_spectral.size(3),
                    device=x_spectral.device, dtype=x_spectral.dtype
                )
            f_prior = self.context_encoder(context_dem)
            f_fused = self.ac_fem(f_spec, f_prior, validity_mask=updated_mask)
        else:
            f_fused = f_spec

        # Step 3: Reconstruction and Uncertainty prediction
        sr_image = self.reconstruction_head(f_fused)
        log_var = self.uncertainty_head(f_fused)
        variance = self.uncertainty_head.get_variance(log_var)
        std_dev = self.uncertainty_head.get_std(log_var)

        output: dict[str, torch.Tensor] = {
            "sr_image": sr_image,
            "log_variance": log_var,
            "variance": variance,
            "std": std_dev,
            "fused_features": f_fused,
            "updated_mask": updated_mask,
        }

        # Step 4: Optional downstream feature upsampling
        if self.include_downstream_heads and self.feat_upsampler is not None:
            output["features_hr"] = self.feat_upsampler(f_fused)

        return output

    def predict_downstream_road(self, features_hr: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        """Runs the downstream rural road extraction head."""
        if self.road_head is None:
            raise RuntimeError("Road downstream head is not initialized.")
        return self.road_head(features_hr, sr_image)

    def predict_downstream_lulc(self, features_hr: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        """Runs the downstream LULC boundary disaggregation head."""
        if self.lulc_head is None:
            raise RuntimeError("LULC downstream head is not initialized.")
        return self.lulc_head(features_hr, sr_image)

    def predict_downstream_change(
        self,
        sr_t1: torch.Tensor,
        sr_t2: torch.Tensor,
        feat_t1: torch.Tensor,
        feat_t2: torch.Tensor,
        unc_t1: torch.Tensor,
        unc_t2: torch.Tensor,
    ) -> torch.Tensor:
        """Runs the downstream bi-temporal disaster damage / change detection head."""
        if self.change_head is None:
            raise RuntimeError("Change downstream head is not initialized.")
        return self.change_head(sr_t1, sr_t2, feat_t1, feat_t2, unc_t1, unc_t2)
