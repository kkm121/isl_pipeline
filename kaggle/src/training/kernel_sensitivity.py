"""
=============================================================================
BharatSRM-Net v4: Sensor MTF/PSF Modeling & Kernel Sensitivity Framework
=============================================================================
Gate 3 & Section 4.4 Implementation:
  - Models realistic point-spread functions (PSF) and modulation transfer functions (MTF)
    for Sentinel-2 MSI and high-resolution Earth Observation sensors.
  - Architectures:
    1. GaussianPSFKernel: 2D Gaussian approximation of optical MTF.
    2. SincWindowedPSFKernel: Sinc-windowed / Airy disc diffraction approximation.
  - Evaluator:
    KernelSensitivityEvaluator: Compares L_degrade and reconstruction metrics (PSNR, SAM, SSIM)
    across diverse PSF kernels to verify cycle-consistency robustness against PSF mismatches.
=============================================================================
"""

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.evaluation.metrics import calculate_psnr, calculate_sam, calculate_ssim


class GaussianPSFKernel(nn.Module):
    r"""
    2D Isotropic Gaussian Point Spread Function (PSF) Kernel.
    Models effective sensor MTF as a Gaussian blur filter:
      G(x, y) = \frac{1}{2\pi \sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
    """

    def __init__(
        self,
        kernel_size: int = 7,
        sigma: float = 1.2,
        num_bands: int = 4,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.kernel_size = kernel_size
        self.sigma = sigma
        self.num_bands = num_bands
        self.padding = kernel_size // 2

        # Generate 2D Gaussian Kernel
        coords = torch.arange(kernel_size).float() - (kernel_size - 1) / 2.0
        g_1d = torch.exp(-(coords**2) / (2.0 * (sigma**2)))
        g_2d = g_1d.unsqueeze(1) * g_1d.unsqueeze(0)
        g_2d = g_2d / g_2d.sum()

        # Reshape to depthwise conv format: (num_bands, 1, K, K)
        kernel = g_2d.unsqueeze(0).unsqueeze(0).repeat(num_bands, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def get_kernel(self) -> torch.Tensor:
        """Returns the normalized (num_bands, 1, K, K) PSF kernel tensor."""
        return self.kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies depthwise PSF blur to input imagery without spatial downsampling.

        Args:
            x: (B, C, H, W) where C >= num_bands (first num_bands are blurred)
        """
        if x.size(1) > self.num_bands:
            x_target = x[:, : self.num_bands, :, :]
        else:
            x_target = x

        blurred = F.conv2d(
            x_target,
            self.kernel,
            padding=self.padding,
            groups=self.num_bands,
        )
        return blurred

    def degrade(self, sr_image: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
        r"""
        Simulates physical sensor downsampling degradation:
          \hat{x}_LR = Downsample_s( Blur_PSF( \hat{y}_SR ) )

        Args:
            sr_image: (B, 4, s*H, s*W) Super-resolved image
            scale_factor: Super-resolution scale factor (default s=4)

        Returns:
            downsampled_lr: (B, 4, H, W) Simulated LR observation
        """
        blurred = self.forward(sr_image)
        downsampled = F.avg_pool2d(
            blurred, kernel_size=scale_factor, stride=scale_factor
        )
        return downsampled


class SincWindowedPSFKernel(nn.Module):
    r"""
    Sinc-Windowed / Airy Disc Point Spread Function (PSF) Kernel.
    Models diffraction-limited optical MTF with radial sinc envelope and apodization window:
      h(r) = \frac{\sin(\pi \cdot r \cdot f_c)}{\pi \cdot r \cdot f_c} \cdot w(r)
    Supported window types: 'hann', 'hamming', 'blackman', 'lanczos', 'none'.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        cutoff_freq: float = 0.5,
        window_type: str = "hann",
        num_bands: int = 4,
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError(f"kernel_size must be odd, got {kernel_size}")

        self.kernel_size = kernel_size
        self.cutoff_freq = cutoff_freq
        self.window_type = window_type.lower()
        self.num_bands = num_bands
        self.padding = kernel_size // 2

        # 2D radial coordinate grid
        center = (kernel_size - 1) / 2.0
        y, x = torch.meshgrid(
            torch.arange(kernel_size).float() - center,
            torch.arange(kernel_size).float() - center,
            indexing="ij",
        )
        r = torch.sqrt(x**2 + y**2)
        r_max = center * math.sqrt(2.0)

        # 1. Radial Sinc
        # sinc(0) = 1, sinc(pi * r * fc)
        pi_r_fc = math.pi * r * cutoff_freq
        sinc_term = torch.where(
            r == 0,
            torch.ones_like(r),
            torch.sin(pi_r_fc) / torch.clamp(pi_r_fc, min=1e-8),
        )

        # 2. Windowing function w(r)
        r_norm = torch.clamp(r / r_max, 0.0, 1.0)
        if self.window_type == "hann":
            window = 0.5 + 0.5 * torch.cos(math.pi * r_norm)
        elif self.window_type == "hamming":
            window = 0.54 + 0.46 * torch.cos(math.pi * r_norm)
        elif self.window_type == "blackman":
            window = (
                0.42
                + 0.5 * torch.cos(math.pi * r_norm)
                + 0.08 * torch.cos(2.0 * math.pi * r_norm)
            )
        elif self.window_type == "lanczos":
            pi_r_norm = math.pi * r_norm
            window = torch.where(
                r_norm == 0,
                torch.ones_like(r_norm),
                torch.sin(pi_r_norm) / torch.clamp(pi_r_norm, min=1e-8),
            )
        elif self.window_type == "none":
            window = torch.ones_like(r)
        else:
            raise ValueError(f"Unsupported window_type: {window_type}")

        # Combine, ensure non-negativity for physical photon PSF, and normalize
        kernel_2d = torch.clamp(sinc_term * window, min=0.0)
        sum_k = kernel_2d.sum()
        if sum_k > 0:
            kernel_2d = kernel_2d / sum_k
        else:
            kernel_2d = torch.ones_like(kernel_2d) / (kernel_size * kernel_size)

        # Reshape to depthwise conv format: (num_bands, 1, K, K)
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0).repeat(num_bands, 1, 1, 1)
        self.register_buffer("kernel", kernel)

    def get_kernel(self) -> torch.Tensor:
        """Returns the normalized (num_bands, 1, K, K) PSF kernel tensor."""
        return self.kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies depthwise PSF blur to input imagery."""
        if x.size(1) > self.num_bands:
            x_target = x[:, : self.num_bands, :, :]
        else:
            x_target = x

        blurred = F.conv2d(
            x_target,
            self.kernel,
            padding=self.padding,
            groups=self.num_bands,
        )
        return blurred

    def degrade(self, sr_image: torch.Tensor, scale_factor: int = 4) -> torch.Tensor:
        """Simulates physical sensor downsampling degradation."""
        blurred = self.forward(sr_image)
        downsampled = F.avg_pool2d(
            blurred, kernel_size=scale_factor, stride=scale_factor
        )
        return downsampled


class KernelSensitivityEvaluator:
    r"""
    Evaluator quantifying super-resolution model robustness to PSF kernel mismatch.
    Section 4.4 & Gate 3 Specification:
      Tests cycle-consistency degradation loss L_degrade across diverse PSF kernels:
        - Nominal Gaussian (\sigma=1.2)
        - Narrow Gaussian (\sigma=0.8)
        - Wide Gaussian (\sigma=1.8)
        - Sinc-Hann Diffraction (\fc=0.5)
        - Sinc-Hamming Diffraction (\fc=0.5)
        - Sinc-Blackman Diffraction (\fc=0.6)
    """

    def __init__(
        self,
        scale_factor: int = 4,
        num_bands: int = 4,
        kernel_size: int = 7,
    ):
        self.scale_factor = scale_factor
        self.num_bands = num_bands
        self.kernel_size = kernel_size

    def generate_default_kernel_suite(self) -> dict[str, nn.Module]:
        """Generates the standardized suite of 6 PSF kernels for sensitivity analysis."""
        return {
            "Gaussian_nominal": GaussianPSFKernel(
                kernel_size=self.kernel_size, sigma=1.2, num_bands=self.num_bands
            ),
            "Gaussian_narrow": GaussianPSFKernel(
                kernel_size=self.kernel_size, sigma=0.8, num_bands=self.num_bands
            ),
            "Gaussian_wide": GaussianPSFKernel(
                kernel_size=self.kernel_size, sigma=1.8, num_bands=self.num_bands
            ),
            "Sinc_Hann": SincWindowedPSFKernel(
                kernel_size=self.kernel_size,
                cutoff_freq=0.5,
                window_type="hann",
                num_bands=self.num_bands,
            ),
            "Sinc_Hamming": SincWindowedPSFKernel(
                kernel_size=self.kernel_size,
                cutoff_freq=0.5,
                window_type="hamming",
                num_bands=self.num_bands,
            ),
            "Sinc_Blackman": SincWindowedPSFKernel(
                kernel_size=self.kernel_size,
                cutoff_freq=0.6,
                window_type="blackman",
                num_bands=self.num_bands,
            ),
        }

    def compute_degradation_loss(
        self,
        sr_image: torch.Tensor,
        lr_observed: torch.Tensor,
        psf_kernel: nn.Module,
    ) -> float:
        """
        Computes L1 degradation consistency loss:
          L_degrade = || Downsample_s(Blur_K(sr_image)) - lr_observed ||_1
        """
        # Ensure kernel is on the same device as sr_image
        if hasattr(psf_kernel, "kernel"):
            psf_kernel = psf_kernel.to(sr_image.device)

        if lr_observed.size(1) > self.num_bands:
            lr_ref = lr_observed[:, : self.num_bands, :, :]
        else:
            lr_ref = lr_observed

        with torch.no_grad():
            lr_simulated = psf_kernel.degrade(
                sr_image, scale_factor=self.scale_factor
            )
            l1_diff = F.l1_loss(lr_simulated, lr_ref)
        return float(l1_diff.item())

    def evaluate_kernel_sensitivity(
        self,
        sr_pred: torch.Tensor,
        lr_observed: torch.Tensor,
        hr_target: torch.Tensor | None = None,
        kernels: dict[str, nn.Module] | None = None,
    ) -> dict[str, Any]:
        """
        Runs sensitivity evaluation across all kernels in the suite.

        Args:
            sr_pred: (B, 4, s*H, s*W) Super-resolved prediction
            lr_observed: (B, 10, H, W) or (B, 4, H, W) Observed LR input
            hr_target: Optional (B, 4, s*H, s*W) High-resolution reference
            kernels: Optional custom dictionary of named PSF kernels

        Returns:
            Dict containing:
                - 'per_kernel_metrics': dict of metrics for each kernel
                - 'nominal_degrade_loss': L_degrade on nominal Gaussian kernel
                - 'mean_degrade_loss': Average L_degrade across all kernels
                - 'std_degrade_loss': Standard deviation of L_degrade
                - 'max_degrade_loss': Maximum degradation loss
                - 'sensitivity_ratio': (max - min) / nominal
                - 'gate3_passed': Boolean indicating if sensitivity is within Gate 3 tolerance
        """
        if kernels is None:
            kernels = self.generate_default_kernel_suite()

        per_kernel_metrics: dict[str, dict[str, float]] = {}
        degrade_losses: list[float] = []

        for name, kernel in kernels.items():
            l_deg = self.compute_degradation_loss(sr_pred, lr_observed, kernel)
            degrade_losses.append(l_deg)

            kernel_dict = {
                "L_degrade": l_deg,
            }

            # If HR target is provided, compute baseline reconstruction metrics
            if hr_target is not None:
                psnr_dict = calculate_psnr(sr_pred, hr_target)
                sam_val = calculate_sam(sr_pred, hr_target)
                ssim_dict = calculate_ssim(sr_pred, hr_target)
                kernel_dict["PSNR_mean"] = psnr_dict["PSNR_mean"]
                kernel_dict["SAM_deg"] = sam_val
                kernel_dict["SSIM_mean"] = ssim_dict["SSIM_mean"]

            per_kernel_metrics[name] = kernel_dict

        nominal_loss = per_kernel_metrics.get("Gaussian_nominal", {}).get(
            "L_degrade", degrade_losses[0]
        )
        mean_loss = float(np.mean(degrade_losses))
        std_loss = float(np.std(degrade_losses))
        max_loss = float(np.max(degrade_losses))
        min_loss = float(np.min(degrade_losses))
        sensitivity_ratio = float(
            (max_loss - min_loss) / (nominal_loss + 1e-8)
        )

        # Gate 3 criterion: std of degradation loss across kernel variations < 0.05
        # and nominal degradation loss < 0.15
        gate3_passed = bool(std_loss < 0.05 and nominal_loss < 0.15)

        return {
            "per_kernel_metrics": per_kernel_metrics,
            "nominal_degrade_loss": nominal_loss,
            "mean_degrade_loss": mean_loss,
            "std_degrade_loss": std_loss,
            "max_degrade_loss": max_loss,
            "min_degrade_loss": min_loss,
            "sensitivity_ratio": sensitivity_ratio,
            "gate3_passed": gate3_passed,
        }
