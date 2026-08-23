"""
=============================================================================
BharatSRM-Net v4: Leave-One-Region-Out (LORO) Cross-Validation Framework
=============================================================================
Section 9.5 Specification:
  Evaluates out-of-domain geographic generalization across 4 Indian Areas of Interest (AOIs):
    1. 'indo_gangetic': Alluvial agricultural plains, severe aerosol/haze.
    2. 'peri_urban': Dense urban fringe, high spatial frequency infrastructure.
    3. 'western_ghats': Tropical forest canopy, rugged high-relief terrain, heavy cloud.
    4. 'rajasthan': Arid desert / scrubland, bright high-albedo sand, low moisture.

Protocol:
  4-Fold cross-validation where in each fold, 3 AOIs form the training/validation set
  and 1 unseen AOI is evaluated as the strictly blind test benchmark.
=============================================================================
"""

from typing import Any

import numpy as np
import torch
from torch import nn

from .metrics import (
    calculate_ergas,
    calculate_psnr,
    calculate_rmse,
    calculate_sam,
    evaluate_all_metrics,
)

INDIAN_AOIS: list[str] = [
    "indo_gangetic",
    "peri_urban",
    "western_ghats",
    "rajasthan",
]

AOI_METADATA: dict[str, dict[str, str]] = {
    "indo_gangetic": {
        "landscape": "Alluvial Agricultural Plains",
        "atmospheric": "High Aerosol Optical Depth / Haze / Crop Burning",
        "terrain": "Flat (<50m elevation delta)",
    },
    "peri_urban": {
        "landscape": "Dense Built-up / Urban-Rural Fringe",
        "atmospheric": "Moderate Smog / Urban Heat Island",
        "terrain": "Low-to-moderate relief",
    },
    "western_ghats": {
        "landscape": "Tropical Forest Canopy / Plantation",
        "atmospheric": "Frequent Orographic Cloud & Shadow",
        "terrain": "High relief / Steep Slopes (>500m delta)",
    },
    "rajasthan": {
        "landscape": "Arid Desert / Shrubland / Salt Flats",
        "atmospheric": "High Solar Irradiance / Dust",
        "terrain": "Undulating Sand Dunes & Rocky Plains",
    },
}


class LeaveOneRegionOutEvaluator:
    """
    Manages Leave-One-Region-Out cross-validation and regional generalization benchmarking.
    """

    def __init__(
        self,
        aois: list[str] | None = None,
        scale_factor: float = 4.0,
    ):
        self.aois = aois if aois is not None else list(INDIAN_AOIS)
        self.scale_factor = scale_factor

    def create_folds(self) -> list[dict[str, Any]]:
        """
        Generates the 4 LORO fold configurations.

        Returns:
            List of fold dicts containing fold index, held-out test region, and train regions.
        """
        folds = []
        for i, held_out in enumerate(self.aois):
            train_regions = [r for r in self.aois if r != held_out]
            folds.append(
                {
                    "fold_idx": i,
                    "held_out_region": held_out,
                    "train_regions": train_regions,
                    "test_region": held_out,
                    "metadata": AOI_METADATA.get(held_out, {}),
                }
            )
        return folds

    def evaluate_region_predictions(
        self,
        sr_pred: torch.Tensor,
        hr_target: torch.Tensor,
        aoi_name: str,
    ) -> dict[str, float]:
        """
        Computes all standard remote sensing metrics for a specific regional prediction batch.

        Args:
            sr_pred: (B, 4, H_hr, W_hr) Super-resolved prediction
            hr_target: (B, 4, H_hr, W_hr) High-resolution ground truth
            aoi_name: Name of the evaluated AOI region

        Returns:
            Dict containing regional PSNR, SAM, ERGAS, and RMSE metrics.
        """
        metrics = evaluate_all_metrics(
            sr_pred, hr_target, scale_factor=self.scale_factor
        )
        return metrics

    def evaluate_model_on_region(
        self,
        model: nn.Module,
        aoi_name: str,
        lr_tensor: torch.Tensor,
        hr_tensor: torch.Tensor,
        mask_tensor: torch.Tensor | None = None,
        dem_tensor: torch.Tensor | None = None,
    ) -> dict[str, float]:
        """
        Evaluates a PyTorch model on a specific held-out region.
        """
        model.eval()
        device = next(model.parameters()).device if list(model.parameters()) else "cpu"
        lr_in = lr_tensor.to(device)
        mask_in = mask_tensor.to(device) if mask_tensor is not None else None
        dem_in = dem_tensor.to(device) if dem_tensor is not None else None

        with torch.no_grad():
            out = model(lr_in, validity_mask=mask_in, context_dem=dem_in)
            if isinstance(out, dict):
                sr_pred = out["sr_image"]
            else:
                sr_pred = out

        return self.evaluate_region_predictions(sr_pred.cpu(), hr_tensor.cpu(), aoi_name)

    def aggregate_loro_results(
        self,
        per_region_results: dict[str, dict[str, float]],
    ) -> dict[str, Any]:
        """
        Aggregates per-region metric dicts into cross-regional statistics.

        Args:
            per_region_results: Dict mapping AOI name -> metric dictionary

        Returns:
            Dict containing per-region breakdown, cross-regional mean & std, and hardest/best region.
        """
        psnr_means = [m["PSNR_mean"] for m in per_region_results.values()]
        sam_values = [m["SAM_deg"] for m in per_region_results.values()]
        ergas_values = [m["ERGAS"] for m in per_region_results.values()]
        rmse_values = [m["RMSE_mean"] for m in per_region_results.values()]

        # Hardest region has lowest PSNR / highest SAM
        sorted_by_psnr = sorted(
            per_region_results.items(), key=lambda item: item[1]["PSNR_mean"]
        )
        hardest_aoi = sorted_by_psnr[0][0]
        best_aoi = sorted_by_psnr[-1][0]

        summary = {
            "per_region": per_region_results,
            "mean_psnr": float(np.mean(psnr_means)),
            "std_psnr": float(np.std(psnr_means)),
            "mean_sam": float(np.mean(sam_values)),
            "std_sam": float(np.std(sam_values)),
            "mean_ergas": float(np.mean(ergas_values)),
            "std_ergas": float(np.std(ergas_values)),
            "mean_rmse": float(np.mean(rmse_values)),
            "std_rmse": float(np.std(rmse_values)),
            "hardest_region": hardest_aoi,
            "best_region": best_aoi,
            "regional_spread_psnr": float(np.max(psnr_means) - np.min(psnr_means)),
        }
        return summary

    def compute_generalization_gap(
        self,
        in_domain_metrics: dict[str, float],
        held_out_metrics: dict[str, float],
    ) -> dict[str, float]:
        """
        Computes domain generalization gap between in-domain validation and held-out test region.
        """
        gap_psnr = float(
            in_domain_metrics.get("PSNR_mean", 0.0)
            - held_out_metrics.get("PSNR_mean", 0.0)
        )
        gap_sam = float(
            held_out_metrics.get("SAM_deg", 0.0)
            - in_domain_metrics.get("SAM_deg", 0.0)
        )
        gap_ergas = float(
            held_out_metrics.get("ERGAS", 0.0)
            - in_domain_metrics.get("ERGAS", 0.0)
        )

        return {
            "delta_psnr_db": gap_psnr,
            "delta_sam_deg": gap_sam,
            "delta_ergas": gap_ergas,
        }

    def format_markdown_report(
        self,
        summary_results: dict[str, Any],
    ) -> str:
        """
        Generates a formatted GitHub-style Markdown report summarizing LORO evaluation.
        """
        lines = [
            "### BharatSRM-Net v4: Leave-One-Region-Out (LORO) Cross-Validation Report",
            "",
            "| AOI / Held-Out Region | Landscape Type | PSNR (dB) | SAM (deg) | ERGAS | RMSE (mean) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]

        per_region = summary_results.get("per_region", {})
        for aoi, metrics in per_region.items():
            meta = AOI_METADATA.get(aoi, {})
            landscape = meta.get("landscape", "Unknown")
            psnr = metrics.get("PSNR_mean", 0.0)
            sam = metrics.get("SAM_deg", 0.0)
            ergas = metrics.get("ERGAS", 0.0)
            rmse = metrics.get("RMSE_mean", 0.0)
            lines.append(
                f"| **{aoi}** | {landscape} | {psnr:.2f} | {sam:.2f}° | {ergas:.2f} | {rmse:.4f} |"
            )

        mean_psnr = summary_results.get("mean_psnr", 0.0)
        std_psnr = summary_results.get("std_psnr", 0.0)
        mean_sam = summary_results.get("mean_sam", 0.0)
        std_sam = summary_results.get("std_sam", 0.0)
        mean_ergas = summary_results.get("mean_ergas", 0.0)
        std_ergas = summary_results.get("std_ergas", 0.0)
        mean_rmse = summary_results.get("mean_rmse", 0.0)
        std_rmse = summary_results.get("std_rmse", 0.0)
        hardest = summary_results.get("hardest_region", "N/A")
        best = summary_results.get("best_region", "N/A")

        lines.append(
            f"| **Cross-Regional Mean** | *All 4 Indian AOIs* | **{mean_psnr:.2f} ± {std_psnr:.2f}** | **{mean_sam:.2f}° ± {std_sam:.2f}°** | **{mean_ergas:.2f} ± {std_ergas:.2f}** | **{mean_rmse:.4f} ± {std_rmse:.4f}** |"
        )
        lines.append("")
        lines.append(f"- **Hardest Region**: `{hardest}` (lowest cross-validation PSNR)")
        lines.append(f"- **Best Generalizing Region**: `{best}`")
        lines.append(
            f"- **Regional PSNR Spread**: `{summary_results.get('regional_spread_psnr', 0.0):.2f} dB`"
        )

        return "\n".join(lines)
