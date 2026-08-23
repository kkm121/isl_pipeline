"""
Unit Tests for Remote Sensing Evaluation Metrics & Calibration
"""

import pytest
import torch

from src.evaluation.cloud_stratified_eval import CloudStratifiedEvaluator
from src.evaluation.metrics import (
    calculate_psnr,
    calculate_rmse,
    calculate_sam,
)
from src.evaluation.uncertainty_calibration import UncertaintyCalibrationEvaluator


def test_metrics_identical_inputs():
    p = torch.ones(1, 4, 32, 32)
    t = torch.ones(1, 4, 32, 32)

    psnr = calculate_psnr(p, t)
    assert psnr["PSNR_mean"] == 100.0

    sam = calculate_sam(p, t)
    assert sam == pytest.approx(0.0, abs=1e-3)

    rmse = calculate_rmse(p, t)
    assert rmse["RMSE_mean"] == pytest.approx(0.0, abs=1e-5)


def test_cloud_stratified_evaluator():
    evaluator = CloudStratifiedEvaluator(cloud_prob_threshold=0.40)
    p = torch.rand(1, 4, 32, 32)
    t = torch.rand(1, 4, 32, 32)

    # Left half clear (0.0), right half cloud (1.0)
    cp = torch.zeros(1, 1, 32, 32)
    cp[:, :, :, 16:] = 1.0

    results = evaluator.evaluate_stratified(p, t, cp)
    assert "clear" in results
    assert "cloud_core" in results
    assert results["clear"]["pixel_count"] > 0


def test_uncertainty_calibration():
    calibrator = UncertaintyCalibrationEvaluator(num_bins=5)
    p = torch.ones(1, 4, 32, 32)
    t = torch.ones(1, 4, 32, 32) + torch.linspace(0.01, 0.5, 32 * 32 * 4).reshape(1, 4, 32, 32)
    var = torch.linspace(0.01, 0.25, 32 * 32 * 4).reshape(1, 4, 32, 32)

    res = calibrator.compute_reliability_curve(p, t, var)
    assert res["num_bins"] == 5
    assert len(res["binned_predicted_variance"]) > 0
    assert res["spread_skill_correlation"] > 0.0
