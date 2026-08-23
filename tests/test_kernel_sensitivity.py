"""
Unit and Integration Tests for Sensor PSF Modeling & Kernel Sensitivity Evaluator
"""

import pytest
import torch

from src.training.kernel_sensitivity import (
    GaussianPSFKernel,
    KernelSensitivityEvaluator,
    SincWindowedPSFKernel,
)


def test_gaussian_psf_kernel_properties():
    kernel_module = GaussianPSFKernel(kernel_size=7, sigma=1.2, num_bands=4)
    kernel = kernel_module.get_kernel()

    assert kernel.shape == (4, 1, 7, 7)
    # Check normalization: sum across spatial dimensions equals 1.0
    for b in range(4):
        assert torch.isclose(kernel[b, 0].sum(), torch.tensor(1.0), atol=1e-5)
    # Check non-negativity
    assert torch.all(kernel >= 0.0)


def test_gaussian_psf_degradation():
    kernel_module = GaussianPSFKernel(kernel_size=7, sigma=1.2, num_bands=4)
    sr_img = torch.rand(2, 4, 32, 32)
    lr_sim = kernel_module.degrade(sr_img, scale_factor=4)

    assert lr_sim.shape == (2, 4, 8, 8)
    assert torch.all(lr_sim >= 0.0) and torch.all(lr_sim <= 1.0)


@pytest.mark.parametrize("win_type", ["hann", "hamming", "blackman", "lanczos", "none"])
def test_sinc_windowed_psf_windows(win_type: str):
    kernel_module = SincWindowedPSFKernel(
        kernel_size=7, cutoff_freq=0.5, window_type=win_type, num_bands=4
    )
    kernel = kernel_module.get_kernel()

    assert kernel.shape == (4, 1, 7, 7)
    for b in range(4):
        assert torch.isclose(kernel[b, 0].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.all(kernel >= 0.0)


def test_sinc_windowed_psf_degradation():
    kernel_module = SincWindowedPSFKernel(
        kernel_size=7, cutoff_freq=0.5, window_type="hann", num_bands=4
    )
    sr_img = torch.rand(2, 4, 32, 32)
    lr_sim = kernel_module.degrade(sr_img, scale_factor=4)

    assert lr_sim.shape == (2, 4, 8, 8)


def test_kernel_sensitivity_evaluator():
    evaluator = KernelSensitivityEvaluator(scale_factor=4, num_bands=4, kernel_size=7)
    suite = evaluator.generate_default_kernel_suite()
    assert len(suite) == 6

    # Create dummy SR, LR, and HR images
    sr_pred = torch.rand(2, 4, 32, 32)
    lr_obs = torch.rand(2, 10, 8, 8)
    hr_target = torch.rand(2, 4, 32, 32)

    results = evaluator.evaluate_kernel_sensitivity(
        sr_pred=sr_pred,
        lr_observed=lr_obs,
        hr_target=hr_target,
    )

    assert "per_kernel_metrics" in results
    assert len(results["per_kernel_metrics"]) == 6
    assert "mean_degrade_loss" in results
    assert "std_degrade_loss" in results
    assert "sensitivity_ratio" in results
    assert "gate3_passed" in results

    # Verify per-kernel metric contents
    for k_name, k_dict in results["per_kernel_metrics"].items():
        assert "L_degrade" in k_dict
        assert "PSNR_mean" in k_dict
        assert "SAM_deg" in k_dict
        assert k_dict["L_degrade"] >= 0.0
