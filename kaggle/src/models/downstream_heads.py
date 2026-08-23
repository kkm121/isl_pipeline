"""
=============================================================================
BharatSRM-Net v4: Downstream Application Heads
=============================================================================
Applications:
  1. Rural Road & Linear Infrastructure Extraction (PMGSY / PM Gati Shakti)
  2. Built-up / Water-Body / LULC Boundary Disaggregation
  3. Bi-Temporal Change & Disaster Damage Detection
=============================================================================
"""


import torch
from torch import nn


class RuralRoadExtractionHead(nn.Module):
    """Downstream head for extracting rural road centerlines and linear infrastructure."""

    def __init__(self, in_channels: int = 64, image_bands: int = 4):
        super().__init__()
        # Takes both high-level backbone features (upsampled) + high-res SR image
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + image_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # Probability map [0, 1]
        )

    def forward(self, features_hr: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        # features_hr: (B, in_channels, 4H, 4W), sr_image: (B, 4, 4H, 4W)
        concat = torch.cat([features_hr, sr_image], dim=1)
        road_prob = self.net(concat)
        return road_prob


class BuiltUpLULCHead(nn.Module):
    """Downstream head for fine-scale Land Use / Land Cover and urban boundary disaggregation."""

    def __init__(self, in_channels: int = 64, image_bands: int = 4, num_classes: int = 5):
        super().__init__()
        # Classes: 0: Built-up, 1: Water, 2: Agriculture, 3: Forest/Canopy, 4: Barren/Scrubland
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + image_bands, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, features_hr: torch.Tensor, sr_image: torch.Tensor) -> torch.Tensor:
        concat = torch.cat([features_hr, sr_image], dim=1)
        logits = self.net(concat)
        return logits


class ChangeDamageHead(nn.Module):
    """Downstream Siamese change & damage detection head comparing bi-temporal SR pairs."""

    def __init__(self, in_channels: int = 64, image_bands: int = 4):
        super().__init__()
        # Inputs: diff(SR1, SR2), diff(feat1, feat2), uncertainty1, uncertainty2
        in_dim = image_bands + in_channels + (image_bands * 2)
        self.net = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # Significant change probability [0, 1]
        )

    def forward(
        self,
        sr_t1: torch.Tensor,
        sr_t2: torch.Tensor,
        feat_t1: torch.Tensor,
        feat_t2: torch.Tensor,
        unc_t1: torch.Tensor,
        unc_t2: torch.Tensor,
    ) -> torch.Tensor:
        img_diff = torch.abs(sr_t1 - sr_t2)
        feat_diff = torch.abs(feat_t1 - feat_t2)
        inp = torch.cat([img_diff, feat_diff, unc_t1, unc_t2], dim=1)
        change_prob = self.net(inp)
        return change_prob
