r"""
=============================================================================
BharatSRM-Net v4: Partial Convolution Layer for Cloud and Shadow Masking
=============================================================================
Formulation:
  x' = W^T (X \odot M) * (\sum 1 / \sum M) + b, if \sum M > 0 else 0
  M' = 1 if \sum M > 0 else 0

Reference:
  Liu et al., "Image Inpainting for Irregular Holes Using Partial Convolutions"
=============================================================================
"""


import torch
import torch.nn.functional as F
from torch import nn


class PartialConv2d(nn.Module):
    """2D Partial Convolution layer for irregular cloud/shadow validity mask handling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = True,
        multi_channel_mask: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.multi_channel_mask = multi_channel_mask

        # Learnable feature convolution weights
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter("bias", None)

        # Fixed mask convolution kernel (all ones) for counting valid pixels
        mask_in_channels = in_channels if multi_channel_mask else 1
        self.register_buffer(
            "mask_kernel",
            torch.ones(1, mask_in_channels, kernel_size, kernel_size),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=2.236)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for Partial Convolution.
        
        Args:
            x: Input feature tensor of shape (B, C_in, H, W)
            mask: Binary validity mask of shape (B, 1, H, W) or (B, C_in, H, W) where 1=valid, 0=cloud/corrupted.
        
        Returns:
            out_features: Output feature tensor of shape (B, C_out, H_out, W_out)
            updated_mask: Updated validity mask of shape (B, 1, H_out, W_out)
        """
        if mask is None:
            mask = torch.ones(
                x.size(0), 1, x.size(2), x.size(3), dtype=x.dtype, device=x.device
            )

        # Ensure mask matches dtype
        mask = mask.to(dtype=x.dtype)

        # Mask-weighted sum of valid inputs in receptive field
        # mask_sum shape: (B, 1, H_out, W_out)
        with torch.no_grad():
            mask_in = mask if self.multi_channel_mask else mask[:, :1, :, :]
            mask_sum = F.conv2d(
                mask_in,
                self.mask_kernel,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
            )
            # Normalization scale: (\sum 1 / \sum M)
            kernel_elements = self.kernel_size * self.kernel_size * (self.in_channels if self.multi_channel_mask else 1)
            mask_ratio = kernel_elements / (mask_sum + 1e-8)
            new_mask = (mask_sum > 0).to(dtype=x.dtype)
            mask_ratio = mask_ratio * new_mask

        # Masked input features: X \odot M
        if mask.size(1) == 1 and x.size(1) > 1:
            masked_x = x * mask
        else:
            masked_x = x * mask

        # Standard convolution over masked features
        raw_out = F.conv2d(
            masked_x,
            self.weight,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Apply renormalization scale factor
        scaled_out = raw_out * mask_ratio

        if self.bias is not None:
            bias_view = self.bias.view(1, -1, 1, 1)
            out_features = (scaled_out + bias_view) * new_mask
        else:
            out_features = scaled_out * new_mask

        return out_features, new_mask
