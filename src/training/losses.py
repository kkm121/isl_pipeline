r"""
=============================================================================
BharatSRM-Net v4: Physical & Spectral Composite Loss Formulation
=============================================================================
Objective:
  L_core = \lambda_1 L_rec + \lambda_2 L_spec + \lambda_3 L_degrade + \lambda_4 L_struct + \lambda_5 L_conf

Components:
  1. L_rec: Charbonnier smooth L1 loss (robust to tile boundary registration noise).
  2. L_spec: Spectral Angle Mapper (SAM) angular loss (enforces spectral fidelity).
  3. L_degrade: Sensor MTF/PSF cycle-consistency degradation constraint.
  4. L_struct: SSIM + Sobel gradient edge preservation.
  5. L_conf: Heteroscedastic regression loss training the per-pixel log-variance.
=============================================================================
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


class CharbonnierLoss(nn.Module):
    r"""Charbonnier (Smooth L1) loss: sqrt(||y - \hat{y}||^2 + \epsilon^2)."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps_sq = eps**2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps_sq)
        return torch.mean(loss)


class SpectralAngleMapperLoss(nn.Module):
    """Spectral Angle Mapper (SAM) loss in radians measuring spectral distortion independent of brightness."""

    def __init__(self, eps: float = 1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Cast to float32 for numerical stability under AMP FP16
        pred = pred.float()
        target = target.float()
        dot = torch.sum(pred * target, dim=1)  # (B, H, W)
        norm_pred = torch.sqrt(torch.sum(pred * pred, dim=1) + self.eps)
        norm_target = torch.sqrt(torch.sum(target * target, dim=1) + self.eps)

        cos_angle = dot / (norm_pred * norm_target + self.eps)
        # Clamp to [-1.0 + eps, 1.0 - eps] in FP32 to prevent infinite gradients from arccos derivative -1/sqrt(1-x^2) at x=1.0
        cos_angle = torch.clamp(cos_angle, min=-1.0 + self.eps, max=1.0 - self.eps)
        sam_rad = torch.acos(cos_angle)
        return torch.mean(sam_rad)


class DegradationConsistencyLoss(nn.Module):
    r"""
    Sensor MTF/PSF Degradation Consistency Loss:
    Downsampling the SR reconstruction under a modeled sensor point-spread function
    must reconstruct the observed low-resolution input.
    """

    def __init__(
        self,
        num_bands: int = 4,
        scale_factor: int = 4,
        kernel_size: int = 7,
        psf_sigma: float = 1.2,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.scale_factor = scale_factor
        self.kernel_size = kernel_size
        self.psf_sigma = psf_sigma

        # Precompute 2D Gaussian PSF kernel for MTF modeling
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g_1d = torch.exp(-(coords**2) / (2 * psf_sigma**2))
        g_2d = g_1d.unsqueeze(1) @ g_1d.unsqueeze(0)
        g_2d = g_2d / g_2d.sum()

        # Reshape to depthwise conv kernel: (num_bands, 1, K, K)
        kernel = g_2d.unsqueeze(0).unsqueeze(0).repeat(num_bands, 1, 1, 1)
        self.register_buffer("psf_kernel", kernel)
        self.padding = kernel_size // 2

    def forward(self, sr_image: torch.Tensor, lr_observed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr_image: (B, 4, 4H, 4W) Super-resolved prediction [R, G, B, NIR]
            lr_observed: (B, 4, H, W) or (B, 10, H, W) Observed Sentinel-2 input
        """
        sr_image = sr_image.float()
        lr_observed = lr_observed.float()

        # If LR has 10 bands [B2(Blue), B3(Green), B4(Red), B8(NIR), ...], extract and order to match SR [R, G, B, NIR]:
        if lr_observed.size(1) >= 10:
            # Index 2 = B4 (Red), Index 1 = B3 (Green), Index 0 = B2 (Blue), Index 3 = B8 (NIR)
            lr_4bands = lr_observed[:, [2, 1, 0, 3], :, :]
        elif lr_observed.size(1) > self.num_bands:
            lr_4bands = lr_observed[:, :self.num_bands, :, :]
        else:
            lr_4bands = lr_observed

        # Apply sensor PSF blur with reflect padding to prevent dark edge halos
        padded_sr = F.pad(sr_image, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        psf_kernel = self.psf_kernel.to(device=sr_image.device, dtype=sr_image.dtype)
        blurred = F.conv2d(
            padded_sr,
            psf_kernel,
            padding=0,
            groups=self.num_bands,
        )

        # Apply box downsampling with stride = scale_factor (s=4)
        downsampled_lr = F.avg_pool2d(
            blurred, kernel_size=self.scale_factor, stride=self.scale_factor
        )

        # L1 consistency constraint
        l_degrade = F.l1_loss(downsampled_lr, lr_4bands)
        return l_degrade


class StructuralSSIMLoss(nn.Module):
    """Structural Similarity (SSIM) + Sobel Gradient Edge Preservation Loss."""

    def __init__(self, window_size: int = 11, in_channels: int = 4, data_range: float = 1.0):
        super().__init__()
        self.window_size = window_size
        self.in_channels = in_channels
        self.data_range = data_range

        # 1D Gaussian kernel for SSIM window
        sigma = 1.5
        gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2)) for x in range(window_size)])
        gauss = gauss / gauss.sum()
        _2D_window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = _2D_window.unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.register_buffer("window", window)

        # Sobel edge filters
        sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        sobel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).unsqueeze(0).unsqueeze(0).repeat(in_channels, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        padd = self.window_size // 2
        window = self.window.to(device=img1.device, dtype=img1.dtype)
        mu1 = F.conv2d(img1, window, padding=padd, groups=self.in_channels)
        mu2 = F.conv2d(img2, window, padding=padd, groups=self.in_channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # F.relu guards against negative variance from floating-point imprecision
        sigma1_sq = F.relu(F.conv2d(img1 * img1, window, padding=padd, groups=self.in_channels) - mu1_sq)
        sigma2_sq = F.relu(F.conv2d(img2 * img2, window, padding=padd, groups=self.in_channels) - mu2_sq)
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=self.in_channels) - mu1_mu2

        # Scale constants by data_range (L) per SSIM spec: C1 = (K1*L)^2, C2 = (K2*L)^2
        C1 = (0.01 * self.data_range) ** 2
        C2 = (0.03 * self.data_range) ** 2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        ssim_term = 1.0 - self._ssim(pred, target)

        # Sobel edge gradient term
        sobel_x = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
        sobel_y = self.sobel_y.to(device=pred.device, dtype=pred.dtype)
        gx_pred = F.conv2d(pred, sobel_x, padding=1, groups=self.in_channels)
        gy_pred = F.conv2d(pred, sobel_y, padding=1, groups=self.in_channels)
        gx_tgt = F.conv2d(target, sobel_x, padding=1, groups=self.in_channels)
        gy_tgt = F.conv2d(target, sobel_y, padding=1, groups=self.in_channels)

        edge_loss = F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt)
        return ssim_term + 0.5 * edge_loss


class HeteroscedasticUncertaintyLoss(nn.Module):
    r"""
    Heteroscedastic Aleatoric Uncertainty Loss:
      L_conf = (1/N) \sum_i [ \exp(-s_i) * ||y_i - \hat{y}_i||^2 + s_i ], where s_i = \log \sigma_i^2.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, log_variance: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        target = target.float()
        log_variance = log_variance.float()
        # diff_sq: (B, 4, H, W)
        diff_sq = (pred - target) ** 2
        # loss = exp(-s) * diff_sq + s
        precision = torch.exp(-log_variance)
        loss = precision * diff_sq + log_variance
        return torch.mean(loss)


class CompositeBharatSRMLoss(nn.Module):
    r"""
    Master Composite Loss Function for BharatSRM-Net v4:
      L_core = \lambda_1 L_rec + \lambda_2 L_spec + \lambda_3 L_degrade + \lambda_4 L_struct + \lambda_5 L_conf
    """

    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_spec: float = 0.1,
        lambda_degrade: float = 0.5,
        lambda_struct: float = 0.2,
        lambda_conf: float = 0.05,
    ):
        super().__init__()
        self.lambda_rec = lambda_rec
        self.lambda_spec = lambda_spec
        self.lambda_degrade = lambda_degrade
        self.lambda_struct = lambda_struct
        self.lambda_conf = lambda_conf

        self.charbonnier = CharbonnierLoss(eps=1e-3)
        self.sam = SpectralAngleMapperLoss(eps=1e-7)
        self.degrade = DegradationConsistencyLoss(num_bands=4, scale_factor=4)
        self.struct = StructuralSSIMLoss(window_size=11, in_channels=4)
        self.conf = HeteroscedasticUncertaintyLoss()

    def forward(
        self,
        sr_pred: torch.Tensor,
        hr_target: torch.Tensor,
        lr_input: torch.Tensor,
        log_variance: torch.Tensor,
        epoch: int = 1,
        warmup_epochs: int = 3,
    ) -> dict[str, torch.Tensor]:
        """
        Calculates all 5 loss components with loss term warmup for degradation and uncertainty.
        Enforces float32 full precision to prevent FP16 gradient underflow/overflow.
        """
        sr_pred = sr_pred.float()
        hr_target = hr_target.float()
        lr_input = lr_input.float()
        log_variance = log_variance.float()

        # Loss term warmup factor (0 -> 1 over first warmup_epochs)
        if warmup_epochs <= 0:
            warmup_factor = 1.0
        elif epoch <= warmup_epochs:
            warmup_factor = float(epoch) / float(warmup_epochs)
        else:
            warmup_factor = 1.0

        l_rec = self.charbonnier(sr_pred, hr_target)
        l_spec = self.sam(sr_pred, hr_target)
        l_degrade = self.degrade(sr_pred, lr_input)
        l_struct = self.struct(sr_pred, hr_target)
        l_conf = self.conf(sr_pred, hr_target, log_variance)

        # Composite total loss
        l_total = (
            self.lambda_rec * l_rec
            + self.lambda_spec * l_spec
            + (self.lambda_degrade * warmup_factor) * l_degrade
            + self.lambda_struct * l_struct
            + (self.lambda_conf * warmup_factor) * l_conf
        )

        return {
            "loss_total": l_total,
            "loss_rec": l_rec,
            "loss_spec": l_spec,
            "loss_degrade": l_degrade,
            "loss_struct": l_struct,
            "loss_conf": l_conf,
            "warmup_factor": torch.tensor(warmup_factor),
        }

SSIMLoss = StructuralSSIMLoss
