"""
=============================================================================
BharatSRM-Net v4: Remote Sensing Quantitative Metric Evaluators
=============================================================================
Standard Earth Observation Metrics:
  1. PSNR (Peak Signal-to-Noise Ratio, dB)
  2. SSIM (Structural Similarity Index)
  3. SAM (Spectral Angle Mapper, degrees)
  4. ERGAS (Relative Dimensionless Global Error in Synthesis)
  5. RMSE (Root Mean Squared Error)
  6. Degradation Consistency Error (L_degrade)
=============================================================================
"""


import numpy as np
import torch


def calculate_psnr(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    data_range: float = 1.0,
    eps: float = 1e-10,
) -> dict[str, float]:
    """Calculates Peak Signal-to-Noise Ratio (PSNR) in dB per band and overall mean."""
    if isinstance(pred, torch.Tensor):
        p = pred.detach().cpu().numpy()
    else:
        p = np.array(pred)

    if isinstance(target, torch.Tensor):
        t = target.detach().cpu().numpy()
    else:
        t = np.array(target)

    # Ensure shape (B, C, H, W) or (C, H, W)
    if p.ndim == 3:
        p = p[np.newaxis, ...]
        t = t[np.newaxis, ...]

    _, C, _, _ = p.shape
    band_names = ["Red", "Green", "Blue", "NIR"][:C]
    band_psnr = {}

    for c in range(C):
        mse_c = np.mean((p[:, c, :, :] - t[:, c, :, :]) ** 2)
        if mse_c < eps:
            psnr_c = 100.0
        else:
            psnr_c = 10.0 * np.log10((data_range ** 2) / mse_c)
        band_psnr[f"PSNR_{band_names[c] if c < len(band_names) else f'B{c}'}"] = float(psnr_c)

    overall_mse = np.mean((p - t) ** 2)
    overall_psnr = 100.0 if overall_mse < eps else float(10.0 * np.log10((data_range ** 2) / overall_mse))
    band_psnr["PSNR_mean"] = overall_psnr

    return band_psnr


def calculate_sam(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    eps: float = 1e-7,
) -> float:
    """Calculates Spectral Angle Mapper (SAM) in degrees (lower is better, 0° is identical)."""
    if isinstance(pred, torch.Tensor):
        p = pred.detach().cpu().numpy()
    else:
        p = np.array(pred)

    if isinstance(target, torch.Tensor):
        t = target.detach().cpu().numpy()
    else:
        t = np.array(target)

    if p.ndim == 3:
        p = p[np.newaxis, ...]
        t = t[np.newaxis, ...]

    # Dot product along spectral dimension C (axis 1)
    dot = np.sum(p * t, axis=1)
    norm_p = np.sqrt(np.sum(p * p, axis=1))
    norm_t = np.sqrt(np.sum(t * t, axis=1))

    denom = norm_p * norm_t
    cos_theta = np.where(denom > eps, dot / np.maximum(denom, eps), 1.0)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    sam_deg = np.arccos(cos_theta) * (180.0 / np.pi)
    return float(np.mean(sam_deg))


def calculate_ergas(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    scale_factor: float = 4.0,
    eps: float = 1e-8,
) -> float:
    r"""
    Calculates Relative Dimensionless Global Error in Synthesis (ERGAS):
      ERGAS = 100 * (1/s) * sqrt( (1/K) * \sum_{k=1}^K (RMSE_k^2 / \mu_k^2) )
    """
    if isinstance(pred, torch.Tensor):
        p = pred.detach().cpu().numpy()
    else:
        p = np.array(pred)

    if isinstance(target, torch.Tensor):
        t = target.detach().cpu().numpy()
    else:
        t = np.array(target)

    if p.ndim == 3:
        p = p[np.newaxis, ...]
        t = t[np.newaxis, ...]

    _, C, _, _ = p.shape
    sum_ratio = 0.0

    for c in range(C):
        rmse_c = np.sqrt(np.mean((p[:, c, :, :] - t[:, c, :, :]) ** 2))
        mu_c = np.mean(t[:, c, :, :]) + eps
        sum_ratio += (rmse_c / mu_c) ** 2

    ergas = 100.0 * (1.0 / scale_factor) * np.sqrt((1.0 / C) * sum_ratio)
    return float(ergas)


def calculate_rmse(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Calculates Root Mean Squared Error (RMSE) per band and mean."""
    if isinstance(pred, torch.Tensor):
        p = pred.detach().cpu().numpy()
    else:
        p = np.array(pred)

    if isinstance(target, torch.Tensor):
        t = target.detach().cpu().numpy()
    else:
        t = np.array(target)

    if p.ndim == 3:
        p = p[np.newaxis, ...]
        t = t[np.newaxis, ...]

    _, C, _, _ = p.shape
    band_names = ["Red", "Green", "Blue", "NIR"][:C]
    res = {}

    for c in range(C):
        rmse_c = float(np.sqrt(np.mean((p[:, c, :, :] - t[:, c, :, :]) ** 2)))
        res[f"RMSE_{band_names[c] if c < len(band_names) else f'B{c}'}"] = rmse_c

    res["RMSE_mean"] = float(np.sqrt(np.mean((p - t) ** 2)))
    return res


def evaluate_all_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_factor: float = 4.0,
) -> dict[str, float]:
    """Computes all primary quantitative remote sensing super-resolution metrics."""
    metrics: dict[str, float] = {}

    # PSNR
    psnr_dict = calculate_psnr(pred, target)
    metrics.update(psnr_dict)

    # SAM (Spectral Angle Mapper)
    metrics["SAM_deg"] = calculate_sam(pred, target)

    # ERGAS
    metrics["ERGAS"] = calculate_ergas(pred, target, scale_factor=scale_factor)

    # RMSE
    rmse_dict = calculate_rmse(pred, target)
    metrics.update(rmse_dict)

    return metrics
