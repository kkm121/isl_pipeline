"""
=============================================================================
BharatSRM-Net v4: Adaptive Cloud-Aware Feature Enhancement Module (AC-FEM)
=============================================================================
Mathematical Formulation:
  M_channel = \\sigma( W_2 \\cdot \text{ReLU}( W_1 \\cdot \text{AvgPool}(F_{spec} \\parallel F_{prior}) ) )
  F_{fused} = M_{channel} \\odot \text{Conv}_{3\times 3}(F_{spec} \\parallel F_{prior}) + (1 - M) \\cdot F_{prior}

Role:
  Mask-aware feature weighting: Down-weights unreliable spectral features under
  cloud cover and up-weights context features in those regions.
=============================================================================
"""


import torch
from torch import nn


class ACFEM(nn.Module):
    """Adaptive Cloud-Aware Feature Enhancement Module."""

    def __init__(self, spec_channels: int, prior_channels: int, out_channels: int, reduction: int = 4):
        super().__init__()
        in_total = spec_channels + prior_channels
        
        # Channel attention projection (Squeeze-and-Excitation over concatenated streams)
        hidden_dim = max(16, in_total // reduction)
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_total, hidden_dim, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Spatial cross-attention
        self.q_proj = nn.Conv2d(spec_channels, out_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(prior_channels, out_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(prior_channels, out_channels, kernel_size=1)

        # Spatial cross-stream fusion convolution
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(in_total + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Prior feature projection to match out_channels if necessary
        if prior_channels != out_channels:
            self.prior_proj = nn.Conv2d(prior_channels, out_channels, kernel_size=1)
        else:
            self.prior_proj = nn.Identity()

    def forward(
        self,
        f_spec: torch.Tensor,
        f_prior: torch.Tensor,
        validity_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for AC-FEM.

        Args:
            f_spec: Multispectral feature maps of shape (B, C_spec, H, W)
            f_prior: Context prior feature maps of shape (B, C_prior, H, W)
            validity_mask: Binary validity mask of shape (B, 1, H, W) (1=valid, 0=cloud/shadow)

        Returns:
            f_fused: Fused feature representation of shape (B, C_out, H, W)
        """
        B, C_spec, H, W = f_spec.shape

        # Concatenate spectral evidence and context prior
        f_concat_orig = torch.cat([f_spec, f_prior], dim=1)

        # Channel attention weighting vector
        m_channel = self.channel_gate(f_concat_orig)  # (B, C_out, 1, 1)

        # Linear Spatial Cross-Attention in FP32 (O(N * C^2) compute, O(C^2) memory footprint)
        # Prevents quadratic O(N^2) memory explosion on large satellite rasters
        q = self.q_proj(f_spec).view(B, -1, H * W).permute(0, 2, 1).float()  # (B, HW, C)
        k = self.k_proj(f_prior).view(B, -1, H * W).permute(0, 2, 1).float()  # (B, HW, C)
        v = self.v_proj(f_prior).view(B, -1, H * W).permute(0, 2, 1).float()  # (B, HW, C)
        
        q_norm = torch.softmax(q, dim=-1)   # (B, HW, C)
        k_norm = torch.softmax(k, dim=-2)   # (B, HW, C)
        
        # Context summary: (B, C, HW) @ (B, HW, C) -> (B, C, C)
        context = torch.bmm(k_norm.transpose(-2, -1), v)  # (B, C, C)
        
        # Projected cross-attention features: (B, HW, C) @ (B, C, C) -> (B, HW, C)
        ca_out = torch.bmm(q_norm, context).permute(0, 2, 1).to(dtype=f_spec.dtype).view(B, -1, H, W)

        # Concatenate with cross-attention output
        f_concat = torch.cat([f_spec, f_prior, ca_out], dim=1)

        # Convolved fusion features
        f_conv = self.spatial_fusion(f_concat)   # (B, C_out, H, W)
        f_weighted = m_channel * f_conv

        # Prior projected stream
        f_prior_proj = self.prior_proj(f_prior)  # (B, C_out, H, W)

        # Mask-gated combination:
        # If validity_mask is provided, in invalid regions (1 - M), prioritize context prior
        if validity_mask is not None:
            # Interpolate validity_mask to feature spatial dimensions if needed
            if validity_mask.shape[2:] != f_spec.shape[2:]:
                m = nn.functional.interpolate(
                    validity_mask, size=f_spec.shape[2:], mode="nearest"
                )
            else:
                m = validity_mask
            
            # F_fused = (M * F_weighted) + (1 - M) * F_prior
            f_fused = (m * f_weighted) + ((1.0 - m) * f_prior_proj)
        else:
            f_fused = f_weighted

        return f_fused
