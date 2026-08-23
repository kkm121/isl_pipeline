r"""
=============================================================================
BharatSRM-Net v4: Uncertainty Calibration & Reliability Protocol
=============================================================================
Section 9.6 Protocol:
  1. Reliability Check: Bins test-set pixels by predicted variance \sigma^2 into quantiles,
     and computes the actual MSE in each bin to verify linearity.
  2. Spread-Skill Check: Spatial correlation between predicted uncertainty and actual error.
=============================================================================
"""

from typing import Any

import numpy as np
import torch


class UncertaintyCalibrationEvaluator:
    """Evaluates whether the predicted uncertainty map is statistically calibrated."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def compute_reliability_curve(
        self,
        sr_pred: torch.Tensor,
        hr_target: torch.Tensor,
        predicted_variance: torch.Tensor,
    ) -> dict[str, Any]:
        r"""
        Computes binned reliability curve of predicted variance \sigma^2 vs actual squared error.
        """
        p = sr_pred.detach().cpu().numpy().flatten()
        t = hr_target.detach().cpu().numpy().flatten()
        var = predicted_variance.detach().cpu().numpy().flatten()

        actual_sq_error = (p - t) ** 2

        # Sort by predicted variance and partition into quantile bins
        quantiles = np.linspace(0, 100, self.num_bins + 1)
        bin_edges = np.percentile(var, quantiles)

        binned_pred_var = []
        binned_actual_mse = []
        bin_counts = []

        for i in range(self.num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == self.num_bins - 1:
                mask = (var >= low) & (var <= high)
            else:
                mask = (var >= low) & (var < high)

            if mask.sum() > 0:
                mean_pred_v = float(np.mean(var[mask]))
                mean_act_mse = float(np.mean(actual_sq_error[mask]))
                binned_pred_var.append(mean_pred_v)
                binned_actual_mse.append(mean_act_mse)
                bin_counts.append(int(mask.sum()))

        # Correlation between predicted uncertainty and empirical error
        correlation = float(np.corrcoef(var, actual_sq_error)[0, 1]) if len(var) > 1 else 0.0
        
        # Temperature scaling calibration factor
        mean_var = float(np.mean(var))
        mean_err = float(np.mean(actual_sq_error))
        temp_scale = float(mean_err / (mean_var + 1e-8))

        return {
            "num_bins": self.num_bins,
            "binned_predicted_variance": binned_pred_var,
            "binned_actual_mse": binned_actual_mse,
            "bin_pixel_counts": bin_counts,
            "spread_skill_correlation": correlation,
            "is_monotonic": bool(np.all(np.diff(binned_actual_mse) >= -1e-5)),
            "temperature_scaling_factor": temp_scale,
            "calibrated_mean_variance": mean_var * temp_scale,
            "empirical_mean_mse": mean_err,
        }
