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
        self.gn1 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            bias=False,
        )
        self.gn2 = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.drop(out)
        out = self.gn2(self.conv2(out))
        return self.relu(out + res)


class LightweightWindowAttention(nn.Module):
    """Lightweight local window self-attention for non-local spatial-spectral dependencies."""

    def __init__(self, dim: int, num_heads: int = 4, window_size: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.scale = (dim // num_heads) ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        ws = self.window_size

        # Pad spatial dimensions to multiple of window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            _, _, hp, wp = x.shape
        else:
            hp, wp = h, w

        # Reshape to local non-overlapping spatial windows
        # (B, C, H, W) -> (B * num_windows, window_size * window_size, C)
        x_windows = (
            x.view(b, c, hp // ws, ws, wp // ws, ws)
            .permute(0, 2, 4, 3, 5, 1)
            .contiguous()
            .reshape(-1, ws * ws, c)
        )

        num_win = x_windows.shape[0]
        qkv = (
            self.qkv(x_windows)
            .reshape(num_win, ws * ws, 3, self.num_heads, c // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0].float(), qkv[1].float(), qkv[2].float()

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out_win = (attn @ v).to(dtype=x.dtype).transpose(1, 2).reshape(num_win, ws * ws, c)
        out_win = self.proj(out_win)

        # Reverse window partitioning back to full feature tensor
        out = (
            out_win.view(b, hp // ws, wp // ws, ws, ws, c)
            .permute(0, 5, 1, 3, 2, 4)
            .contiguous()
            .reshape(b, c, hp, wp)
        )

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :h, :w]

        return out + x


class MaskedMultispectralEncoder(nn.Module):
    """
    Multispectral Encoder combining:
      1. Gated Partial Convolutions for cloud/shadow masking
      2. Dilated residual blocks for receptive field expansion (r=1, 2, 4, 8)
      3. Lightweight Window Attention for spectral cross-band dependencies
    """

    def __init__(
        self,
        in_channels: int = 10,
        base_channels: int = 64,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        use_window_attention: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.use_window_attention = use_window_attention

        # Step 1: Initial PartialConv layers
        self.pconv1 = PartialConv2d(in_channels, base_channels // 2, kernel_size=3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pconv2 = PartialConv2d(base_channels // 2, base_channels, kernel_size=3, padding=1)

        # Step 2: Dilated residual blocks (r in {1, 2, 4, 8})
        self.dilated_blocks = nn.ModuleList([
            DilatedResidualBlock(base_channels, dilation=r)
            for r in dilation_rates
        ])

        # Step 3: Optional local window self-attention
        if use_window_attention:
            self.attn = LightweightWindowAttention(base_channels, num_heads=4, window_size=8)
        else:
            self.attn = nn.Identity()

        # Step 4: Output feature fusion projection
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=min(8, base_channels), num_channels=base_channels),
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
        c_mid = max(2, out_channels // 2)
        c_out = max(2, out_channels)
        g_mid = 8 if c_mid % 8 == 0 else (4 if c_mid % 4 == 0 else (2 if c_mid % 2 == 0 else 1))
        g_out = 8 if c_out % 8 == 0 else (4 if c_out % 4 == 0 else (2 if c_out % 2 == 0 else 1))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c_mid, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=g_mid, num_channels=c_mid),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=g_out, num_channels=c_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=g_out, num_channels=c_out),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, dem: torch.Tensor) -> torch.Tensor:
        # dem: (B, 2, H, W) [Elevation, Slope]
        return self.net(dem)
