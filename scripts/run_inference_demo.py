"""
=============================================================================
BharatSRM-Net v4: Local Interactive Inference Demo & Visual Generator
=============================================================================
Runs full 10m -> 2.5m Super-Resolution inference using the pretrained model
and generates side-by-side visual comparison images (Low-Res vs SR vs Uncertainty).
=============================================================================
"""

import os
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import numpy as np
from PIL import Image

from src.models.bharatsrm_net import BharatSRMNetV4
from src.inference.tiler import TiledInferenceEngine

def run_demo():
    print("=" * 80)
    print("=== BharatSRM-Net v4: Interactive Inference & Visual Demo ===")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference Device: {device}")
    
    # 1. Initialize Model Architecture
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=64,
        use_context_stream=True,
        include_downstream_heads=False,
    ).to(device)
    
    # 2. Load Pretrained Checkpoint
    ckpt_path = "kaggle_outputs/bharatsrm_v4_pretrained.pth"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[OK] Loaded pretrained model checkpoint from {ckpt_path}")
    else:
        print("[WARN] Pretrained checkpoint not found. Using initialization weights.")
        
    model.eval()
    
    # 3. Create a Sample Multi-Spectral Satellite Patch (128 x 128 x 10 bands)
    print("\nGenerating sample 10-band Sentinel-2 Level-2A scene tile (128x128)...")
    np.random.seed(42)
    lr_np = np.zeros((10, 128, 128), dtype=np.float32)
    for b in range(10):
        base_val = 0.15 if b in [0, 1, 2] else (0.35 if b == 3 else 0.20)
        lr_np[b] = np.clip(np.random.normal(base_val, 0.05, (128, 128)), 0.01, 1.0)
    
    lr_np[:, 50:60, :] += 0.2
    lr_np[:, :, 70:80] += 0.2
    lr_np = np.clip(lr_np, 0.0, 1.0)
    
    lr_tensor = torch.from_numpy(lr_np).unsqueeze(0).to(device)
    mask_tensor = torch.ones((1, 1, 128, 128), device=device)
    dem_tensor = torch.zeros((1, 2, 128, 128), device=device)
    
    # 4. Run Seamless 2D Hanning Window Inference
    print("Running Seamless 2D Hanning Window Tiler Inference...")
    t0 = time.time()
    tiler = TiledInferenceEngine(tile_size=64, overlap=16, scale_factor=4, device=str(device))
    with torch.no_grad():
        out = tiler.predict_large_scene(model, lr_tensor, mask_tensor, dem_tensor)
    dt = time.time() - t0
    
    sr_image = out["sr_image"].squeeze(0).cpu().numpy() # (4, 512, 512)
    raw_var = out["variance"].squeeze(0).cpu().numpy()   # (1, 512, 512)
    
    # Apply calibrated temperature scaling (T* = 0.4318)
    cal_var = raw_var * 0.4318
    uncertainty_std = np.sqrt(cal_var).mean(axis=0) # (512, 512)
    
    print(f"[OK] Inference Complete in {dt:.3f} seconds!")
    print(f"  * Input Resolution  : 128 x 128 (10m Sentinel-2 multi-spectral)")
    print(f"  * Output Resolution : 512 x 512 (2.5m nominal RGBN super-resolution - 4x scale)")
    print(f"  * Mean Calibrated Uncertainty (std): {np.mean(uncertainty_std):.4f}")
    
    # 5. Render and Save Visual Images
    os.makedirs("outputs", exist_ok=True)
    
    # A. Low-Res 10m RGB (Bands 2, 1, 0 -> Red, Green, Blue)
    lr_rgb = np.stack([lr_np[2], lr_np[1], lr_np[0]], axis=-1)
    lr_rgb = (np.clip(lr_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    lr_img = Image.fromarray(lr_rgb).resize((512, 512), Image.NEAREST)
    lr_img.save("outputs/demo_low_res_10m.png")
    
    # B. Super-Resolved 2.5m RGB (Bands 0, 1, 2)
    sr_rgb = np.stack([sr_image[0], sr_image[1], sr_image[2]], axis=-1)
    sr_rgb = (np.clip(sr_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    sr_img = Image.fromarray(sr_rgb)
    sr_img.save("outputs/demo_super_resolved_2.5m.png")
    
    # C. Uncertainty Heatmap (Turbo Colormap)
    import matplotlib.cm as cm
    norm_unc = (uncertainty_std - np.min(uncertainty_std)) / (np.max(uncertainty_std) - np.min(uncertainty_std) + 1e-8)
    unc_heatmap = (cm.turbo(norm_unc)[:, :, :3] * 255.0).astype(np.uint8)
    unc_img = Image.fromarray(unc_heatmap)
    unc_img.save("outputs/demo_uncertainty_map.png")
    
    # D. Side-by-Side 3-Panel Composite
    composite = Image.new("RGB", (512 * 3 + 40, 512 + 60), color=(240, 240, 245))
    composite.paste(lr_img, (10, 50))
    composite.paste(sr_img, (512 + 20, 50))
    composite.paste(unc_img, (512 * 2 + 30, 50))
    composite.save("outputs/demo_side_by_side_comparison.png")
    
    print("\n" + "=" * 80)
    print("VISUAL DEMO ARTIFACTS GENERATED:")
    print("  1. Low-Res 10m Input   : outputs/demo_low_res_10m.png")
    print("  2. Super-Resolved 2.5m : outputs/demo_super_resolved_2.5m.png")
    print("  3. Calibrated Variance : outputs/demo_uncertainty_map.png")
    print("  4. 3-Panel Comparison  : outputs/demo_side_by_side_comparison.png")
    print("=" * 80)

if __name__ == "__main__":
    run_demo()
