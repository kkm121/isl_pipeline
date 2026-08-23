"""
Unit Tests for Remote Sensing Evaluation Metrics & Calibration
"""

import pytest
import torch
import math

from src.evaluation.cloud_stratified_eval import CloudStratifiedEvaluator
from src.evaluation.metrics import (
    calculate_psnr,
    calculate_rmse,
    calculate_sam,
    calculate_ssim,
    calculate_ergas,
)
from src.evaluation.uncertainty_calibration import UncertaintyCalibrationEvaluator


def test_psnr_identical_clamped():
    p = torch.ones(1, 4, 32, 32)
    t = torch.ones(1, 4, 32, 32)
    psnr = calculate_psnr(p, t)
    assert psnr["PSNR_mean"] == 100.0


def test_psnr_known_mse():
    # RMSE = 0.1, MSE = 0.01. Data range = 1.0. 
    # PSNR = 10 * log10(1^2 / 0.01) = 10 * log10(100) = 20.0
    p = torch.full((1, 1, 32, 32), 0.1)
    t = torch.full((1, 1, 32, 32), 0.0)
    psnr = calculate_psnr(p, t)
    assert psnr["PSNR_mean"] == pytest.approx(20.0, rel=1e-4)


def test_sam_identical_zero():
    p = torch.rand(1, 4, 32, 32) + 0.1
    t = p.clone()
    sam = calculate_sam(p, t)
    assert sam == pytest.approx(0.0, abs=1e-2)

def test_sam_orthogonal_ninety():
    p = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    t = torch.tensor([[[[0.0]], [[1.0]], [[0.0]], [[0.0]]]])
    sam = calculate_sam(p, t)
    assert sam == pytest.approx(90.0, rel=1e-2)

def test_ergas_perfect():
    p = torch.ones(1, 4, 32, 32)
    t = torch.ones(1, 4, 32, 32)
    ergas = calculate_ergas(p, t, scale_factor=4)
    assert ergas == pytest.approx(0.0, abs=1e-4)

def test_rmse_known_value():
    p = torch.full((1, 1, 32, 32), 0.5)
    t = torch.full((1, 1, 32, 32), 0.0)
    rmse = calculate_rmse(p, t)
    assert rmse["RMSE_mean"] == pytest.approx(0.5, rel=1e-4)

def test_ssim_identical():
    p = torch.rand(1, 4, 32, 32)
    t = p.clone()
    ssim = calculate_ssim(p, t)
    assert ssim["SSIM_mean"] == pytest.approx(1.0, abs=1e-4)

def test_cloud_stratified_clear_vs_edge():
    evaluator = CloudStratifiedEvaluator(cloud_prob_threshold=0.40)
    t = torch.zeros(1, 4, 32, 32)
    
    p = torch.zeros(1, 4, 32, 32)
    p[:, :, :, :16] = 0.5
    
    cp = torch.zeros(1, 1, 32, 32)
    cp[:, :, :, 16:] = 1.0

    results = evaluator.evaluate_stratified(p, t, cp)
    
    assert results["clear"]["RMSE"] == pytest.approx(0.5, rel=1e-4)
    if "cloud_core" in results:
        assert results["cloud_core"]["RMSE"] == pytest.approx(0.0, abs=1e-4)


def test_uncertainty_calibration_perfect():
    calibrator = UncertaintyCalibrationEvaluator(num_bins=5)
    
    # create perfect calibration: error squared perfectly matches variance
    # MSE = Var
    var = torch.linspace(0.01, 0.25, 32 * 32 * 4).reshape(1, 4, 32, 32)
    # error = sqrt(var), so error^2 = var
    error = torch.sqrt(var)
    t = torch.zeros(1, 4, 32, 32)
    p = error

    res = calibrator.compute_reliability_curve(p, t, var)
    # perfectly calibrated means high spread_skill_correlation
    assert res["spread_skill_correlation"] > 0.99
