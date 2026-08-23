r"""
=============================================================================
BharatSRM-Net v4: Uncertainty Calibration & Reliability Suite
=============================================================================
Provides:
  1. TemperatureScalingCalibrator: Parametric temperature scaling (Moment & NLL optimal)
  2. IsotonicUncertaintyCalibrator: Non-parametric isotonic regression via PAVA
  3. UncertaintyCalibrationEvaluator: Reliability curves, ENCE, UCE, and Z-score metrics
=============================================================================
"""

from typing import Any, Literal
import numpy as np
import torch


def pava_isotonic_regression(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
    Fits a non-decreasing monotonic step function minimizing sum((y - g(x))^2).
    """
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    values = list(y_sorted.astype(np.float64))

    stack_v: list[float] = []
    stack_w: list[float] = []
    stack_count: list[int] = []

    for v in values:
        cur_v = v
        cur_w = 1.0
        cur_c = 1

        while stack_v and stack_v[-1] >= cur_v:
            prev_v = stack_v.pop()
            prev_w = stack_w.pop()
            prev_c = stack_count.pop()

            cur_v = (prev_w * prev_v + cur_w * cur_v) / (prev_w + cur_w)
            cur_w = prev_w + cur_w
            cur_c = prev_c + cur_c

        stack_v.append(cur_v)
        stack_w.append(cur_w)
        stack_count.append(cur_c)

    fitted_y = np.empty_like(y_sorted, dtype=np.float64)
    idx = 0
    for v, count in zip(stack_v, stack_count):
        fitted_y[idx : idx + count] = v
        idx += count

    return x_sorted, fitted_y


class TemperatureScalingCalibrator:
    r"""
    Calibrates heteroscedastic variance \sigma^2 via temperature scaling.
    Transforms raw variance \sigma_{raw}^2 -> \hat{\sigma}^2 = T * \sigma_{raw}^2
    such that \mathbb{E}[\hat{\sigma}^2] = MSE (Moment matching) or \mathbb{E}[z^2] = 1 (NLL optimal).
    """

    def __init__(self, method: Literal["moment", "nll", "affine"] = "moment"):
        self.method = method
        self.temperature: float = 1.0
        self.affine_a: float = 1.0
        self.affine_b: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        sr_pred: torch.Tensor | np.ndarray,
        hr_target: torch.Tensor | np.ndarray,
        raw_variance: torch.Tensor | np.ndarray,
    ) -> "TemperatureScalingCalibrator":
        """Fits calibration parameters on a validation / calibration set."""
        p = (sr_pred.detach().cpu().numpy() if isinstance(sr_pred, torch.Tensor) else np.asarray(sr_pred)).ravel()
        t = (hr_target.detach().cpu().numpy() if isinstance(hr_target, torch.Tensor) else np.asarray(hr_target)).ravel()
        var = (raw_variance.detach().cpu().numpy() if isinstance(raw_variance, torch.Tensor) else np.asarray(raw_variance)).ravel()

        eps = 1e-8
        var = np.clip(var, eps, None)
        sq_err = (p - t) ** 2
        mse = float(np.mean(sq_err))
        mean_var = float(np.mean(var))

        if self.method == "moment":
            self.temperature = float(mse / (mean_var + eps))
            self.affine_a = 1.0
            self.affine_b = float(np.log(max(self.temperature, 1e-8)))
        elif self.method == "nll":
            z_sq = sq_err / var
            self.temperature = float(np.mean(z_sq))
            self.affine_a = 1.0
            self.affine_b = float(np.log(max(self.temperature, 1e-8)))
        elif self.method == "affine":
            s = np.log(var)
            target_s = np.log(np.clip(sq_err, 1e-6, None))
            A = np.vstack([s, np.ones_like(s)]).T
            sol, _, _, _ = np.linalg.lstsq(A, target_s, rcond=None)
            self.affine_a = float(max(sol[0], 0.01))
            self.affine_b = float(sol[1])
            self.temperature = float(np.exp(self.affine_b))

        self.is_fitted = True
        return self

    def calibrate(self, raw_variance: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        """Applies fitted temperature scaling to raw variance tensor or array."""
        if not self.is_fitted:
            raise RuntimeError("TemperatureScalingCalibrator must be fitted before calling calibrate().")

        if isinstance(raw_variance, torch.Tensor):
            if self.method == "affine":
                log_v = torch.log(torch.clamp(raw_variance, min=1e-8))
                cal_log_v = self.affine_a * log_v + self.affine_b
                return torch.exp(cal_log_v)
            return raw_variance * self.temperature
        else:
            if self.method == "affine":
                log_v = np.log(np.clip(raw_variance, 1e-8, None))
                cal_log_v = self.affine_a * log_v + self.affine_b
                return np.exp(cal_log_v)
            return raw_variance * self.temperature


class IsotonicUncertaintyCalibrator:
    r"""
    Non-parametric monotonic calibration mapping raw predicted variance to empirical MSE.
    Uses Pool Adjacent Violators Algorithm (PAVA).
    """

    def __init__(self, max_calibration_samples: int = 200000):
        self.max_calibration_samples = max_calibration_samples
        self.x_breakpoints: np.ndarray | None = None
        self.y_breakpoints: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        sr_pred: torch.Tensor | np.ndarray,
        hr_target: torch.Tensor | np.ndarray,
        raw_variance: torch.Tensor | np.ndarray,
    ) -> "IsotonicUncertaintyCalibrator":
        p = (sr_pred.detach().cpu().numpy() if isinstance(sr_pred, torch.Tensor) else np.asarray(sr_pred)).ravel()
        t = (hr_target.detach().cpu().numpy() if isinstance(hr_target, torch.Tensor) else np.asarray(hr_target)).ravel()
        var = (raw_variance.detach().cpu().numpy() if isinstance(raw_variance, torch.Tensor) else np.asarray(raw_variance)).ravel()

        if len(var) > self.max_calibration_samples:
            idx = np.random.choice(len(var), self.max_calibration_samples, replace=False)
            p, t, var = p[idx], t[idx], var[idx]

        sq_err = (p - t) ** 2
        x_sorted, y_fitted = pava_isotonic_regression(var, sq_err)

        unique_x, unique_idx = np.unique(x_sorted, return_index=True)
        self.x_breakpoints = unique_x
        self.y_breakpoints = y_fitted[unique_idx]
        self.is_fitted = True
        return self

    def calibrate(self, raw_variance: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
        if not self.is_fitted or self.x_breakpoints is None or self.y_breakpoints is None:
            raise RuntimeError("IsotonicUncertaintyCalibrator must be fitted before calibrate().")

        is_torch = isinstance(raw_variance, torch.Tensor)
        v_np = raw_variance.detach().cpu().numpy() if is_torch else np.asarray(raw_variance)
        shape = v_np.shape

        cal_flat = np.interp(
            v_np.ravel(),
            self.x_breakpoints,
            self.y_breakpoints,
            left=self.y_breakpoints[0],
            right=self.y_breakpoints[-1],
        )
        cal_np = cal_flat.reshape(shape)

        if is_torch:
            return torch.from_numpy(cal_np).to(device=raw_variance.device, dtype=raw_variance.dtype)
        return cal_np


class UncertaintyCalibrationEvaluator:
    """Evaluates whether predicted uncertainty is statistically and probabilistically calibrated."""

    def __init__(self, num_bins: int = 10):
        self.num_bins = num_bins

    def compute_reliability_curve(
        self,
        sr_pred: torch.Tensor | np.ndarray,
        hr_target: torch.Tensor | np.ndarray,
        predicted_variance: torch.Tensor | np.ndarray,
    ) -> dict[str, Any]:
        """Computes binned reliability curve, ENCE, UCE, and spread-skill correlation."""
        p = (sr_pred.detach().cpu().numpy() if isinstance(sr_pred, torch.Tensor) else np.asarray(sr_pred)).ravel()
        t = (hr_target.detach().cpu().numpy() if isinstance(hr_target, torch.Tensor) else np.asarray(hr_target)).ravel()
        var = (predicted_variance.detach().cpu().numpy() if isinstance(predicted_variance, torch.Tensor) else np.asarray(predicted_variance)).ravel()

        actual_sq_error = (p - t) ** 2
        overall_mse = float(np.mean(actual_sq_error))
        overall_mean_var = float(np.mean(var))

        quantiles = np.linspace(0, 100, self.num_bins + 1)
        bin_edges = np.percentile(var, quantiles)

        binned_pred_var = []
        binned_actual_mse = []
        binned_rmv = []
        binned_rmse = []
        bin_counts = []
        norm_errors = []
        uce_errors = []

        total_pixels = len(var)

        for i in range(self.num_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            mask = (var >= low) & (var <= high) if i == self.num_bins - 1 else (var >= low) & (var < high)
            n_k = int(mask.sum())

            if n_k > 0:
                mean_v = float(np.mean(var[mask]))
                mean_mse = float(np.mean(actual_sq_error[mask]))
                rmv = float(np.sqrt(max(mean_v, 0.0)))
                rmse = float(np.sqrt(max(mean_mse, 0.0)))

                binned_pred_var.append(mean_v)
                binned_actual_mse.append(mean_mse)
                binned_rmv.append(rmv)
                binned_rmse.append(rmse)
                bin_counts.append(n_k)

                norm_errors.append(abs(rmv - rmse) / (rmse + 1e-8))
                uce_errors.append((n_k / total_pixels) * abs(mean_v - mean_mse))

        ence = float(np.mean(norm_errors)) * 100.0 if norm_errors else 0.0
        uce = float(np.sum(uce_errors)) if uce_errors else 0.0
        correlation = float(np.corrcoef(var, actual_sq_error)[0, 1]) if len(var) > 1 else 0.0

        z = (p - t) / np.sqrt(np.clip(var, 1e-8, None))
        z_mean = float(np.mean(z))
        z_var = float(np.var(z))

        return {
            "num_bins": self.num_bins,
            "overall_mean_predicted_variance": overall_mean_var,
            "overall_empirical_mse": overall_mse,
            "variance_to_mse_ratio": overall_mean_var / (overall_mse + 1e-8),
            "binned_predicted_variance": binned_pred_var,
            "binned_actual_mse": binned_actual_mse,
            "binned_rmv": binned_rmv,
            "binned_rmse": binned_rmse,
            "bin_pixel_counts": bin_counts,
            "spread_skill_correlation": correlation,
            "ence_percent": ence,
            "uce": uce,
            "standardized_residual_mean": z_mean,
            "standardized_residual_variance": z_var,
            "is_monotonic": bool(np.all(np.diff(binned_actual_mse) >= -1e-5)),
        }
