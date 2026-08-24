"""
=============================================================================
BharatSRM-Net v4: High-Fidelity Residual Pretraining on Dual Tesla T4 GPUs
=============================================================================
Dataset: jucor1/worldstrat (3,928 Real Sentinel-2 L2A / SPOT 6/7 1.5m Scene Pairs)
Architecture: BharatSRMNetV4 with Global Residual Learning & Laplacian Edge Loss
=============================================================================
"""

import os
import glob
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# 1. Architecture Definitions
class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "mask_kernel",
            torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1]),
            persistent=False,
        )
        self.slide_winsize = self.kernel_size[0] * self.kernel_size[1]
        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, x, mask=None):
        if mask is None:
            raw_out = super().forward(x)
            return raw_out, None
        with torch.no_grad():
            self.mask_kernel = self.mask_kernel.to(device=x.device, dtype=x.dtype)
            mask_sum = F.conv2d(
                mask,
                self.mask_kernel,
                bias=None,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=1,
            )
            self.update_mask = torch.clamp(mask_sum, 0.0, 1.0)
            self.mask_ratio = self.slide_winsize / (mask_sum + 1e-8)
            self.mask_ratio = self.mask_ratio * self.update_mask
        raw_out = super().forward(x * mask)
        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = (raw_out - bias_view) * self.mask_ratio + bias_view
            output = output * self.update_mask
        else:
            output = raw_out * self.mask_ratio
        return output, self.update_mask

class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.gn1 = nn.GroupNorm(min(8, channels), channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False)
        self.gn2 = nn.GroupNorm(min(8, channels), channels)

    def forward(self, x):
        res = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(out + res)

class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(2, channels // reduction), 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(2, channels // reduction), channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)

class HighFrequencyPolishBlock(nn.Module):
    def __init__(self, channels: int = 4, hidden_dim: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, hidden_dim, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.ca = ChannelAttention(hidden_dim)
        self.conv3 = nn.Conv2d(hidden_dim, channels, 3, padding=1)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.relu(self.conv2(res))
        res = self.ca(res)
        res = self.conv3(res)
        return x + res

class HighFidelityReconstructionHead(nn.Module):
    def __init__(self, in_channels: int = 64, out_bands: int = 4, scale_factor: int = 4, hidden_dim: int = 128):
        super().__init__()
        self.scale_factor = scale_factor
        self.refinement = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, hidden_dim), hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.pixel_shuffle_proj = nn.Conv2d(hidden_dim, out_bands * (scale_factor**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.hr_polish = HighFrequencyPolishBlock(out_bands, 32)

    def forward(self, features, lr_base=None):
        x = self.refinement(features)
        x = self.pixel_shuffle_proj(x)
        res = self.pixel_shuffle(x)
        res = self.hr_polish(res)
        if lr_base is not None:
            base_hr = F.interpolate(lr_base, scale_factor=self.scale_factor, mode="bicubic", align_corners=False)
            return torch.clamp(base_hr + torch.tanh(res) * 0.5, 0.0, 1.0)
        return torch.sigmoid(res)

class HeteroscedasticUncertaintyHead(nn.Module):
    def __init__(self, in_channels: int = 64, out_bands: int = 4, scale_factor: int = 4):
        super().__init__()
        self.pixel_shuffle_proj = nn.Conv2d(in_channels, out_bands * (scale_factor**2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.log_var_refine = nn.Sequential(
            nn.Conv2d(out_bands, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, out_bands, 3, padding=1),
        )

    def forward(self, features):
        x = self.pixel_shuffle_proj(features)
        x = self.pixel_shuffle(x)
        return torch.clamp(self.log_var_refine(x), -8.0, 5.0)

class HighFidelityBharatSRMNet(nn.Module):
    def __init__(self, in_bands: int = 10, out_bands: int = 4, base_channels: int = 64):
        super().__init__()
        self.pconv1 = PartialConv2d(in_bands, base_channels // 2, 3, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.pconv2 = PartialConv2d(base_channels // 2, base_channels, 3, padding=1)
        self.dilated_blocks = nn.ModuleList([
            DilatedResidualBlock(base_channels, dilation=r) for r in (1, 2, 4, 8)
        ])
        self.out_proj = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, base_channels), base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.reconstruction_head = HighFidelityReconstructionHead(base_channels, out_bands, scale_factor=4)
        self.uncertainty_head = HeteroscedasticUncertaintyHead(base_channels, out_bands, scale_factor=4)

    def forward(self, x, mask=None):
        f, m = self.pconv1(x, mask)
        f = self.relu(f)
        f, m = self.pconv2(f, m)
        f = self.relu(f)
        for block in self.dilated_blocks:
            f = block(f)
        f = self.out_proj(f)
        lr_base = x[:, [2, 1, 0, 3], :, :] if x.size(1) >= 10 else x[:, :4, :, :]
        sr_image = self.reconstruction_head(f, lr_base=lr_base)
        log_var = self.uncertainty_head(f)
        return {"sr_image": sr_image, "log_variance": log_var}

# 2. Losses
class StructuralSSIMLoss(nn.Module):
    def __init__(self, in_channels: int = 4):
        super().__init__()
        self.in_channels = in_channels
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).repeat(in_channels, 1, 1, 1)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).repeat(in_channels, 1, 1, 1)
        laplacian = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).repeat(in_channels, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplacian", laplacian)

    def forward(self, pred, target):
        sobel_x = self.sobel_x.to(pred.device, pred.dtype)
        sobel_y = self.sobel_y.to(pred.device, pred.dtype)
        lap = self.laplacian.to(pred.device, pred.dtype)
        gx_p = F.conv2d(pred, sobel_x, padding=1, groups=self.in_channels)
        gy_p = F.conv2d(pred, sobel_y, padding=1, groups=self.in_channels)
        lap_p = F.conv2d(pred, lap, padding=1, groups=self.in_channels)
        gx_t = F.conv2d(target, sobel_x, padding=1, groups=self.in_channels)
        gy_t = F.conv2d(target, sobel_y, padding=1, groups=self.in_channels)
        lap_t = F.conv2d(target, lap, padding=1, groups=self.in_channels)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t) + 0.5 * F.l1_loss(lap_p, lap_t)

class HighFidelityCompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.struct = StructuralSSIMLoss(in_channels=4)

    def forward(self, pred, target, log_var):
        l_char = torch.mean(torch.sqrt((pred - target)**2 + 1e-6))
        # SAM loss
        dot = torch.sum(pred * target, dim=1)
        norm_p = torch.sqrt(torch.sum(pred * pred, dim=1) + 1e-7)
        norm_t = torch.sqrt(torch.sum(target * target, dim=1) + 1e-7)
        cos = torch.clamp(dot / (norm_p * norm_t + 1e-7), -0.9999, 0.9999)
        l_sam = torch.mean(torch.acos(cos))
        l_edge = self.struct(pred, target)
        # NLL loss
        prec = torch.exp(-log_var)
        l_conf = torch.mean(prec * (pred - target)**2 + log_var)
        total = 1.0 * l_char + 0.2 * l_sam + 0.8 * l_edge + 0.01 * l_conf
        return total, l_char, l_sam, l_edge

def main():
    print("=== BHARATSRM-NET V4: HIGH-FIDELITY RESIDUAL TRAINING ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | GPUs: {torch.cuda.device_count()}")

    model = HighFidelityBharatSRMNet(in_bands=10, out_bands=4, base_channels=64).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion = HighFidelityCompositeLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = GradScaler()

    print("[OK] High-Fidelity training pipeline initialized.")

if __name__ == "__main__":
    main()
