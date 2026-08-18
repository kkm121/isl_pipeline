"""
SOTA Isolated Sign Spatial-Temporal Transformer (Tier-1 Architecture).

Combines Squeeze-and-Excitation 1D Convolutions (SE-Conv1D) for local temporal
motion bursts with Transformer Multi-Head Self-Attention for global sign semantics.
Supports 328-dimensional multimodal landmark feature representations.
"""

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding for sequence temporal ordering."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return x + self.pe[:, : x.size(1), :]


class SE1DConvBlock(nn.Module):
    """Squeeze-and-Excitation 1D Convolutional Residual Block.

    Captures localized joint kinematic trajectories and dynamically
    re-calibrates temporal feature channels via squeeze-and-excitation.
    """

    def __init__(self, channels: int, reduction: int = 4, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, max(1, channels // reduction), 1),
            nn.GELU(),
            nn.Conv1d(max(1, channels // reduction), channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x
        out = self.drop(self.act(self.norm(self.conv1(x))))
        out = self.conv2(out)
        w = self.se(out)
        out = out * w
        return out + res


class SOTASignTransformer(nn.Module):
    """SOTA Spatial-Temporal Sign Transformer for ISL recognition.

    Inputs: (B, T=150, 328) multimodal features (coordinates + velocities + distance pairs)
    Outputs: (B, num_classes) classification logits (default 263 classes)
    """

    def __init__(
        self,
        in_features: int = 328,
        num_classes: int = 263,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.15,
        max_len: int = 200,
    ):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.d_model = d_model

        self.in_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)

        # 2x SE-Conv1D Blocks (Local temporal motion bursts)
        self.se_block1 = SE1DConvBlock(d_model, dropout=dropout)
        self.se_block2 = SE1DConvBlock(d_model, dropout=dropout)

        # Transformer Encoder Layers (Global sign semantics)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-Pooling Classification Head (MeanPool + MaxPool -> d_model * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout * 2),
            nn.Linear(dim_feedforward, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (B, T, in_features=328)
            src_key_padding_mask: Optional boolean mask (B, T) where True indicates padding.

        Returns:
            Logits of shape (B, num_classes=263)
        """
        # Linear projection + Positional Encoding
        feat = self.in_proj(x)
        feat = self.pos_enc(feat)

        # Pass through SE-Conv blocks: (B, T, C) -> (B, C, T) -> (B, T, C)
        feat_c = feat.transpose(1, 2)
        feat_c = self.se_block1(feat_c)
        feat_c = self.se_block2(feat_c)
        feat = feat_c.transpose(1, 2)

        # Pass through Transformer Encoder
        encoded = self.transformer(feat, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Multi-Pooling Head (Mean + Max pooling along temporal dimension)
        if src_key_padding_mask is not None:
            # Valid positions are False in PyTorch src_key_padding_mask
            valid_mask = (~src_key_padding_mask).unsqueeze(-1).float()
            masked_encoded = encoded * valid_mask
            mean_pool = masked_encoded.sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1.0)
            masked_for_max = encoded.masked_fill(src_key_padding_mask.unsqueeze(-1), -1e9)
            max_pool, _ = masked_for_max.max(dim=1)
        else:
            mean_pool = encoded.mean(dim=1)
            max_pool, _ = encoded.max(dim=1)

        multi_pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, d_model * 2)
        return self.classifier(multi_pooled)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================================
# Training Utilities & Loss Helpers
# ===========================================================================


def mixup_data(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Applies MixUp regularization to input sequence batch."""
    if alpha > 0:
        lam = float(np.random.beta(alpha, alpha))
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(
    criterion: nn.Module,
    pred: torch.Tensor,
    y_a: torch.Tensor,
    y_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Computes MixUp loss between blended targets."""
    return lam * criterion(pred, y_a) + (1.0 - lam) * criterion(pred, y_b)


def calculate_accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Tuple[int, ...] = (1, 5),
) -> List[float]:
    """Computes Top-K classification accuracy percentages.

    Args:
        output: Predicted logits of shape (B, num_classes)
        target: Ground truth class indices of shape (B,)
        topk: Tuple of top-k values to evaluate (e.g. (1, 5))

    Returns:
        List of accuracy percentages [top1_acc, top5_acc, ...] in [0.0, 100.0]
    """
    with torch.no_grad():
        num_classes = output.size(1)
        batch_size = target.size(0)
        if batch_size == 0:
            return [0.0] * len(topk)

        res = []
        for k in topk:
            actual_k = min(k, num_classes)
            _, pred = output.topk(actual_k, dim=1, largest=True, sorted=True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            correct_k = correct.float().sum().item()
            acc = (correct_k / batch_size) * 100.0
            res.append(acc)
        return res
