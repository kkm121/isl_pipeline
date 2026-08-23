"""
=============================================================================
BharatSRM-Net v4: Cloud-Stratified Evaluation Framework
=============================================================================
Section 9.1.1 Specification:
  Evaluates super-resolution reconstruction metrics stratified by:
    1. Clear Pixels (cloud probability < threshold)
    2. Cloud-Edge Pixels (boundary buffer where PartialConv mask propagation is weakest)
    3. Cloud-Shadow Pixels (shadow component of validity mask)
=============================================================================
"""


import numpy as np
import torch
from scipy.ndimage import binary_dilation


class CloudStratifiedEvaluator:
    """Evaluates reconstruction quality across distinct cloud and atmospheric strata."""

    def __init__(self, cloud_prob_threshold: float = 0.40, edge_buffer_pixels: int = 5):
        self.cloud_prob_threshold = cloud_prob_threshold
        self.edge_buffer_pixels = edge_buffer_pixels

    def compute_strata_masks(
        self,
        cloud_prob_map: np.ndarray,
        shadow_mask: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Derives boolean masks for: 'clear', 'cloud_edge', 'cloud_shadow'.

        Args:
            cloud_prob_map: (H, W) array of cloud probabilities [0.0, 1.0]
            shadow_mask: (H, W) boolean array for detected cloud shadow
        """
        # 1. Cloud core
        cloud_core = cloud_prob_map >= self.cloud_prob_threshold

        # 2. Cloud edge (boundary buffer zone)
        dilated = binary_dilation(cloud_core, iterations=self.edge_buffer_pixels)
        cloud_edge = dilated & (~cloud_core)

        # 3. Cloud shadow
        if shadow_mask is None:
            # Approximate shadow as dark non-cloud adjacent areas
            shadow_mask = np.zeros_like(cloud_core, dtype=bool)

        # 4. Clear pixels
        clear_pixels = (~cloud_core) & (~cloud_edge) & (~shadow_mask)

        return {
            "clear": clear_pixels,
            "cloud_edge": cloud_edge,
            "cloud_shadow": shadow_mask,
            "cloud_core": cloud_core,
        }

    def evaluate_stratified(
        self,
        sr_pred: torch.Tensor,
        hr_target: torch.Tensor,
        cloud_prob_hr: torch.Tensor,
    ) -> dict[str, dict[str, float]]:
        """
        Calculates PSNR, SAM, and RMSE across each cloud stratum.
        """
        p = sr_pred.detach().cpu().numpy()
        t = hr_target.detach().cpu().numpy()
        cp = cloud_prob_hr.detach().cpu().numpy()

        if p.ndim == 4:
            p = p[0]
            t = t[0]
            cp = cp[0, 0] if cp.ndim == 4 else cp[0]

        strata = self.compute_strata_masks(cp)
        results: dict[str, dict[str, float]] = {}

        for stratum_name, mask in strata.items():
            if mask.sum() < 10:
                # Not enough pixels in this stratum
                continue

            # Masked evaluation
            p_stratum = p[:, mask]
            t_stratum = t[:, mask]

            mse = np.mean((p_stratum - t_stratum) ** 2)
            psnr = float(10.0 * np.log10(1.0 / (mse + 1e-10)))
            rmse = float(np.sqrt(mse))

            # SAM on masked pixels
            dot = np.sum(p_stratum * t_stratum, axis=0)
            norm_p = np.sqrt(np.sum(p_stratum ** 2, axis=0) + 1e-7)
            norm_t = np.sqrt(np.sum(t_stratum ** 2, axis=0) + 1e-7)
            cos_theta = np.clip(dot / (norm_p * norm_t + 1e-7), -1.0, 1.0)
            sam_deg = float(np.mean(np.arccos(cos_theta) * (180.0 / np.pi)))

            results[stratum_name] = {
                "pixel_count": int(mask.sum()),
                "PSNR_dB": psnr,
                "SAM_deg": sam_deg,
                "RMSE": rmse,
            }

        return results
