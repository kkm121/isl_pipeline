import pytest
import torch

from src.evaluation.leave_one_out import (
    INDIAN_AOIS,
    LeaveOneRegionOutEvaluator,
)
from src.models.baselines import BicubicSR


def test_loro_create_folds():
    evaluator = LeaveOneRegionOutEvaluator()
    folds = evaluator.create_folds()

    assert len(folds) == 4
    for i, fold in enumerate(folds):
        assert fold["fold_idx"] == i
        assert fold["held_out_region"] == INDIAN_AOIS[i]
        assert len(fold["train_regions"]) == 3
        assert fold["held_out_region"] not in fold["train_regions"]
        assert "metadata" in fold


def test_evaluate_region_predictions():
    evaluator = LeaveOneRegionOutEvaluator(scale_factor=4.0)

    # Identical target and pred
    target = torch.rand(2, 4, 32, 32)
    pred = target.clone()

    metrics = evaluator.evaluate_region_predictions(pred, target, "western_ghats")
    assert "PSNR_mean" in metrics
    assert "SAM_deg" in metrics
    assert "ERGAS" in metrics
    assert "RMSE_mean" in metrics
    assert metrics["PSNR_mean"] >= 90.0  # Perfect match
    assert metrics["SAM_deg"] < 0.05


def test_evaluate_model_on_region():
    evaluator = LeaveOneRegionOutEvaluator(scale_factor=4.0)
    model = BicubicSR(in_bands=10, out_bands=4, scale_factor=4)

    lr = torch.rand(2, 10, 16, 16)
    hr = torch.rand(2, 4, 64, 64)

    metrics = evaluator.evaluate_model_on_region(
        model=model,
        aoi_name="indo_gangetic",
        lr_tensor=lr,
        hr_tensor=hr,
    )

    assert "PSNR_mean" in metrics
    assert "SAM_deg" in metrics
    assert "ERGAS" in metrics
    assert "RMSE_mean" in metrics


def test_aggregate_loro_results_and_report():
    evaluator = LeaveOneRegionOutEvaluator()

    mock_results = {
        "indo_gangetic": {"PSNR_mean": 33.5, "SAM_deg": 2.1, "ERGAS": 3.2, "RMSE_mean": 0.021},
        "peri_urban": {"PSNR_mean": 34.2, "SAM_deg": 1.9, "ERGAS": 2.9, "RMSE_mean": 0.019},
        "western_ghats": {"PSNR_mean": 31.8, "SAM_deg": 2.8, "ERGAS": 3.8, "RMSE_mean": 0.026},
        "rajasthan": {"PSNR_mean": 35.1, "SAM_deg": 1.7, "ERGAS": 2.6, "RMSE_mean": 0.017},
    }

    summary = evaluator.aggregate_loro_results(mock_results)

    assert "mean_psnr" in summary
    assert "std_psnr" in summary
    assert summary["hardest_region"] == "western_ghats"  # Lowest PSNR (31.8)
    assert summary["best_region"] == "rajasthan"        # Highest PSNR (35.1)

    # Test markdown report generation
    report = evaluator.format_markdown_report(summary)
    assert "indo_gangetic" in report
    assert "western_ghats" in report
    assert "Cross-Regional Mean" in report
    assert "Hardest Region" in report


def test_generalization_gap():
    evaluator = LeaveOneRegionOutEvaluator()
    in_domain = {"PSNR_mean": 35.0, "SAM_deg": 1.8, "ERGAS": 2.7}
    held_out = {"PSNR_mean": 32.5, "SAM_deg": 2.4, "ERGAS": 3.4}

    gap = evaluator.compute_generalization_gap(in_domain, held_out)
    assert gap["delta_psnr_db"] == pytest.approx(2.5)
    assert gap["delta_sam_deg"] == pytest.approx(0.6)
    assert gap["delta_ergas"] == pytest.approx(0.7)
