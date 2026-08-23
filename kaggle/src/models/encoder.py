"""
=============================================================================
BharatSRM-Net v4: Masked Multispectral & Context Encoders
=============================================================================
Architecture:
  - Masked Multispectral Encoder: PartialConv2D (cloud-aware) + Dilated Residual Blocks (r in {1, 2, 4, 8})
    + Optional Window Self-Attention Block.
  - Context Encoder: Conv2D over CartoDEM elevation/slope only (no road-proximity leakage).
=============================================================================
"""


import torch
from torch import nn

from .partial_conv import PartialConv2d


class DilatedResidualBlock(nn.Module):
    """Residual block with dilated 2D convolutions to expand receptive field without pooling."""

    def __init__(self, channels: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return self.relu(out + res)


class LightweightWindowAttention(nn.Module):
    """Lightweight local window self-attention for non-local spatial-spectral dependencies."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        # Pad to multiple of window_size if necessary
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        if pad_h > 0 or pad_w > 0:
            x = nn.functional.pad(x, (0, pad_w, 0, pad_h))
            Hp, Wp = H + pad_h, W + pad_w
        else:
            Hp, Wp = H, W

        # Reshape to windows: (B * num_windows, window_size*window_size, C)
        x_perm = x.permute(0, 2, 3, 1)  # (B, Hp, Wp, C)
        x_win = x_perm.reshape(
            B,
            Hp // self.window_size,
            self.window_size,
            Wp // self.window_size,
            self.window_size,
            C,
        )
        x_win = x_win.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)

        # Norm + QKV
        x_norm = self.norm(x_win)
        qkv = self.qkv(x_norm).reshape(-1, self.window_size * self.window_size, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out_win = (attn @ v).transpose(1, 2).reshape(-1, self.window_size * self.window_size, C)
        out_win = self.proj(out_win)

        # Restore full shape
        out = out_win.view(
            B,
            Hp // self.window_size,
            Wp // self.window_size,
            self.window_size,
            self.window_size,
            C,
        ).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return out + x


class MaskedMultispectralEncoder(nn.Module):
    """Dual-stage masked multispectral encoder with Partial Convolutions & Dilated Residual blocks."""

    def __init__(
        self,
        in_channels: int = 10,
        base_channels: int = 64,
        dilation_rates: list[int] | None = None,
        use_window_attention: bool = True,
    ):
        super().__init__()
        if dilation_rates is None:
            dilation_rates = [1, 2, 4, 8]

        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_window_attention = use_window_attention

        # Step 1: Initial Partial Convolution for cloud/shadow masking
        self.pconv1 = PartialConv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.pconv2 = PartialConv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

        # Step 2: Multi-scale Dilated Residual Blocks (r in {1, 2, 4, 8})
        self.dilated_blocks = nn.ModuleList([
            DilatedResidualBlock(base_channels, dilation=r) for r in dilation_rates
        ])

        # Step 3: Optional Window Self-Attention Block
        if use_window_attention:
            self.attn = LightweightWindowAttention(base_channels, num_heads=4, window_size=8)
        else:
            self.attn = nn.Identity()

        # Step 4: Output feature fusion projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self, x: torch.Tensor, validity_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, 10, H, W) Sentinel-2 multispectral tensor
            validity_mask: (B, 1, H, W) Cloud validity mask (1=clear, 0=cloud/shadow)

        Returns:
            f_spec: (B, base_channels, H, W) Extracted spectral-spatial feature maps
            updated_mask: (B, 1, H, W) Updated validity mask
        """
        # Partial convolution passes
        feat, m1 = self.pconv1(x, validity_mask)
        feat = self.relu(feat)
        feat, m2 = self.pconv2(feat, m1)
        feat = self.relu(feat)

        # Multi-scale dilated blocks
        for block in self.dilated_blocks:
            feat = block(feat)

        # Window attention
        if self.use_window_attention:
            feat = self.attn(feat)

        f_spec = self.out_proj(feat)
        return f_spec, m2


class ContextEncoder(nn.Module):
    """Context encoder for low-frequency CartoDEM topographic elevation/slope."""

    def __init__(self, in_channels: int = 2, out_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        # dem: (B, 2, H, W) [Elevation, Slope]
        return self.net(dem)
