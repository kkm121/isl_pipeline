"""
=============================================================================
BharatSRM-Net v4: Gate 3 Sensor PSF Kernel Sensitivity Pilot Runner
=============================================================================
Evaluates the robustness of the pretrained BharatSRM-Net v4 super-resolution
reconstruction against optical PSF / MTF sensor variations.
=============================================================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.bharatsrm_net import BharatSRMNetV4
from src.training.kernel_sensitivity import KernelSensitivityEvaluator

def run_gate3_evaluation():
    print("=" * 80)
    print("=== BharatSRM-Net v4: Gate 3 Sensor PSF Kernel Sensitivity Pilot ===")
    print("=" * 80)
    
    # 1. Load Pretrained BharatSRM-Net v4 Checkpoint
    checkpoint_path = "kaggle_outputs/bharatsrm_v4_pretrained.pth"
    model = BharatSRMNetV4(include_downstream_heads=False)
    
    print(f"Loading pretrained weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[OK] Model successfully initialized in evaluation mode.")

    # 2. Prepare Sample Multi-Spectral Sentinel-2 Tensor (B=4, C=10, H=64, W=64)
    # Scaled to typical Level-2A surface reflectance [0.05, 0.40]
    torch.manual_seed(42)
    lr_input = torch.rand(4, 10, 64, 64, dtype=torch.float32) * 0.35 + 0.05
    validity_mask = torch.ones(4, 1, 64, 64, dtype=torch.float32)
    context_dem = torch.zeros(4, 2, 64, 64, dtype=torch.float32)

    # 3. Super-Resolution Forward Pass
    print("\nRunning super-resolution forward pass...")
    with torch.no_grad():
        out = model(lr_input, validity_mask, context_dem)
        sr_pred = out["sr_image"] # (4, 4, 256, 256)
    print(f"[OK] Generated 4-band RGBN SR Imagery of shape {list(sr_pred.shape)}.")

    # 4. Execute Gate 3 6-Kernel PSF Sensitivity Suite
    evaluator = KernelSensitivityEvaluator(scale_factor=4, num_bands=4, kernel_size=7)
    results = evaluator.evaluate_kernel_sensitivity(sr_pred=sr_pred, lr_observed=lr_input)

    print("\n" + "-" * 60)
    print("SENSOR PSF KERNEL SENSITIVITY BREAKDOWN")
    print("-" * 60)
    for k_name, m in results["per_kernel_metrics"].items():
        print(f"  * {k_name:<24}: L_degrade = {m['L_degrade']:.5f}")

    print("\n" + "=" * 60)
    print("GATE 3 QUANTITATIVE VERIFICATION PROOFS")
    print("=" * 60)
    print(f"  1. Nominal Degradation Loss (Gaussian sigma=1.2) : {results['nominal_degrade_loss']:.5f} (Threshold: < 0.15000)")
    print(f"  2. Mean Degradation Loss across 6 Kernels        : {results['mean_degrade_loss']:.5f}")
    print(f"  3. Standard Deviation across all PSF variations  : {results['std_degrade_loss']:.5f} (Threshold: < 0.05000)")
    print(f"  4. Peak-to-Peak Sensitivity Ratio ((Max-Min)/Nom): {results['sensitivity_ratio']:.4f}")
    
    passed = results["gate3_passed"]
    print(f"\nGATE 3 OFFICIAL VERIFICATION SIGN-OFF: {'PASSED (Within Physical Invariance Limits)' if passed else 'FAILED'}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    run_gate3_evaluation()
