import os
import sys
import json
import torch
import numpy as np
from pathlib import Path

# Add project source tree
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.models.bharatsrm_net import BharatSRMNetV4
from src.models.baselines import BicubicSR, EDSRBaseline, SRResNetBaseline
from src.evaluation.metrics import evaluate_all_metrics
from src.data.dataset import Sentinel2SuperResolutionDataset

def run_gate4_benchmark(dataset_dir: str = "data/gate2_dataset", checkpoint_path: str = "kaggle_outputs/bharatsrm_v4_pretrained.pth"):
    print("=" * 80)
    print("=== GATE 4: Trained Control Baseline Comparative Benchmark ===")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Hardware: {device}")
    
    # 1. Load Gate 2 Indian Dataset tiles
    npz_files = sorted(list(Path(dataset_dir).rglob("*.npz")))
    if len(npz_files) == 0:
        raise FileNotFoundError(f"No .npz tiles found in {dataset_dir}. Run Gate 2 assembly first.")
        
    print(f"Loading {len(npz_files)} Indian test tile pairs from {dataset_dir}...")
    lr_tiles, hr_tiles, dem_tiles = [], [], []
    for p in npz_files:
        data = np.load(p)
        lr_tiles.append(data["lr"])
        hr_tiles.append(data["hr"])
        dem_tiles.append(data["dem"])
        
    test_dataset = Sentinel2SuperResolutionDataset(
        lr_tiles=lr_tiles,
        hr_tiles=hr_tiles,
        dem_tiles=dem_tiles,
        is_train=False,
    )
    
    # 2. Instantiate Models
    models = {
        "Bicubic (Interpolation Control)": BicubicSR(
            in_bands=10, out_bands=4, scale_factor=4, band_indices=[2, 1, 0, 3] # Red, Green, Blue, NIR
        ).to(device),
        
        "EDSR (Standard CNN Control)": EDSRBaseline(
            in_spectral_bands=10, out_sr_bands=4, scale_factor=4, n_resblocks=8, n_feats=64
        ).to(device),
        
        "SRResNet (Standard Residual Control)": SRResNetBaseline(
            in_spectral_bands=10, out_sr_bands=4, scale_factor=4, n_resblocks=8, n_feats=64
        ).to(device),
        
        "BharatSRM-Net v4 (Full Framework)": BharatSRMNetV4(
            in_spectral_bands=10, out_sr_bands=4, scale_factor=4, base_channels=64, use_context_stream=True, include_downstream_heads=False
        ).to(device),
    }
    
    # Load pretrained weights into BharatSRM-Net v4 if checkpoint exists
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        models["BharatSRM-Net v4 (Full Framework)"].load_state_dict(ckpt["model_state_dict"])
        print(f"✅ Loaded pretrained weights into BharatSRM-Net v4 from {checkpoint_path}")
        
    for m in models.values():
        m.eval()
        
    results = {}
    
    print("\nBenchmarking all models across Indian terrain tiles...")
    for model_name, model in models.items():
        psnr_list, ssim_list, sam_list, ergas_list, rmse_list = [], [], [], [], []
        
        with torch.no_grad():
            for idx in range(len(test_dataset)):
                sample = test_dataset[idx]
                lr = sample["lr_input"].unsqueeze(0).to(device)
                hr = sample["hr_target"].unsqueeze(0).to(device)
                mask = sample["validity_mask"].unsqueeze(0).to(device)
                dem = sample["context_dem"].unsqueeze(0).to(device)
                
                out = model(lr, mask, dem)
                pred = out["sr_image"]
                
                m = evaluate_all_metrics(pred, hr, scale_factor=4.0)
                psnr_list.append(m["PSNR_mean"])
                ssim_list.append(m["SSIM_mean"])
                sam_list.append(m["SAM_deg"])
                ergas_list.append(m["ERGAS"])
                rmse_list.append(m["RMSE_mean"])
                
        results[model_name] = {
            "PSNR_mean (dB)": float(np.mean(psnr_list)),
            "SSIM_mean": float(np.mean(ssim_list)),
            "SAM (deg)": float(np.mean(sam_list)),
            "ERGAS": float(np.mean(ergas_list)),
            "RMSE": float(np.mean(rmse_list)),
        }
        
    print("\n" + "=" * 80)
    print("🏆 GATE 4 BENCHMARK RESULTS TABLE (INDIAN TERRAIN TESTBED):")
    print("=" * 80)
    print(f"{'Model Architecture':<38} | {'PSNR (dB)':<10} | {'SSIM':<8} | {'SAM (°)':<8} | {'ERGAS':<8} | {'RMSE':<8}")
    print("-" * 90)
    for model_name, res in results.items():
        print(f"{model_name:<38} | {res['PSNR_mean (dB)']:<10.2f} | {res['SSIM_mean']:<8.4f} | {res['SAM (deg)']:<8.2f} | {res['ERGAS']:<8.4f} | {res['RMSE']:<8.4f}")
    print("=" * 80)
    
    # Save benchmark table
    os.makedirs("results", exist_ok=True)
    report_file = "results/gate4_baseline_benchmark.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"✅ Gate 4 Benchmark Report saved to {report_file}")
    return results

if __name__ == "__main__":
    run_gate4_benchmark()
