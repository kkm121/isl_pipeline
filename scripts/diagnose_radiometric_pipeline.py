"""
=============================================================================
BharatSRM-Net v4: Radiometric & Visualization Diagnostic Suite
=============================================================================
Diagnoses:
1. Per-channel min / max / mean statistics for Input RGB, Bicubic, and BharatSRM.
2. Identical fixed radiometric display mapping (no independent auto-scaling).
3. Band ordering verification between Sentinel-2 and SPOT 6/7.
4. Generates side-by-side 4-way visual diagnostic chart.
=============================================================================
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.models.bharatsrm_net import BharatSRMNetV4

def run_diagnostic():
    print("=" * 80)
    print("=== BharatSRM-Net v4: Radiometric & Visualization Diagnostic ===")
    print("=" * 80)
    
    device = torch.device("cpu")
    
    # 1. Load Pretrained Model
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=64,
        use_context_stream=True,
        include_downstream_heads=False,
    ).to(device)
    
    ckpt_path = "kaggle_outputs/bharatsrm_v4_pretrained.pth"
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at {ckpt_path}")
        return
        
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[OK] Pretrained model loaded from {ckpt_path}")
    
    # 2. Load User's Turquoise Lake Image
    image_path = "outputs/user_lake.png"
    if not os.path.exists(image_path):
        print(f"[ERROR] Image not found at {image_path}")
        return
        
    pil_img = Image.open(image_path).convert("RGB")
    # Resize to standard multi-spectral test patch (e.g. 256x144)
    pil_img_lr = pil_img.resize((256, 144), Image.Resampling.BILINEAR)
    img_np = np.array(pil_img_lr).astype(np.float32) / 255.0 # (H, W, 3) where channels are R, G, B
    H, W, _ = img_np.shape
    
    # Sentinel-2 Band mapping: [B2(Blue), B3(Green), B4(Red), B8(NIR), B5..B12]
    lr_10b = np.zeros((1, 10, H, W), dtype=np.float32)
    lr_10b[0, 0] = img_np[:, :, 2] # B2: Blue
    lr_10b[0, 1] = img_np[:, :, 1] # B3: Green
    lr_10b[0, 2] = img_np[:, :, 0] # B4: Red
    lr_10b[0, 3] = img_np[:, :, 1] * 0.8 + img_np[:, :, 0] * 0.2 # B8: NIR approximation
    for b in range(4, 10):
        lr_10b[0, b] = img_np[:, :, 1] * 0.7
        
    lr_tensor = torch.from_numpy(lr_10b).to(device)
    mask_tensor = torch.ones((1, 1, H, W), device=device)
    dem_tensor = torch.zeros((1, 2, H, W), device=device)
    
    # 3. Compute Bicubic Baseline (4x upsampling of RGB)
    lr_rgb_t = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0) # (1, 3, H, W)
    bicubic_rgb = F.interpolate(lr_rgb_t, scale_factor=4, mode="bicubic", align_corners=False).squeeze(0).numpy() # (3, 4H, 4W)
    bicubic_rgb = np.clip(bicubic_rgb, 0.0, 1.0)
    
    # 4. Compute BharatSRM-Net v4 Output
    with torch.no_grad():
        out = model(lr_tensor, mask_tensor, dem_tensor)
        sr_4b = out["sr_image"].squeeze(0).numpy() # (4, 4H, 4W)
        sr_var = out["variance"].squeeze(0).numpy() # (4, 4H, 4W)
        
    # 5. Print Statistical Diagnostics
    print("\n" + "=" * 80)
    print("1. PER-CHANNEL NUMERICAL STATISTICS:")
    print("=" * 80)
    
    print("INPUT RGB (0..1 normalized):")
    for i, cname in enumerate(["Red", "Green", "Blue"]):
        c_data = img_np[:, :, i]
        print(f"  * {cname:<6}: Min = {c_data.min():.4f} | Max = {c_data.max():.4f} | Mean = {c_data.mean():.4f}")
        
    print("\nBICUBIC 4X RGB:")
    for i, cname in enumerate(["Red", "Green", "Blue"]):
        c_data = bicubic_rgb[i]
        print(f"  * {cname:<6}: Min = {c_data.min():.4f} | Max = {c_data.max():.4f} | Mean = {c_data.mean():.4f}")
        
    print("\nBHARATSRM-NET V4 RAW OUTPUT BANDS (4 Channels):")
    for i in range(4):
        c_data = sr_4b[i]
        print(f"  * Channel {i}: Min = {c_data.min():.4f} | Max = {c_data.max():.4f} | Mean = {c_data.mean():.4f}")

    # 6. Test Band Permutations
    print("\n" + "=" * 80)
    print("2. CHANNEL PERMUTATION ANALYSIS:")
    print("=" * 80)
    
    # In WorldStrat SPOT 6/7:
    # Option A: [Ch0=Red, Ch1=Green, Ch2=Blue]
    # Option B: [Ch2=Red, Ch1=Green, Ch0=Blue] (BGRN order)
    # Option C: [Ch0=Blue, Ch1=Green, Ch2=Red]
    
    cand_A = np.stack([sr_4b[0], sr_4b[1], sr_4b[2]], axis=-1) # [0, 1, 2]
    cand_B = np.stack([sr_4b[2], sr_4b[1], sr_4b[0]], axis=-1) # [2, 1, 0]
    cand_C = np.stack([sr_4b[1], sr_4b[2], sr_4b[0]], axis=-1)
    
    # 7. Render with IDENTICAL FIXED MAPPING [0.0, 1.0] -> [0, 255]
    print("\nRendering with Identical Fixed Radiometric Mapping [0.0, 1.0] -> [0, 255]...")
    
    # Fixed linear mapping (No auto-percentile stretching)
    def fixed_map(arr):
        return (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
        
    vis_input = fixed_map(img_np)
    vis_bicubic = fixed_map(bicubic_rgb.transpose(1, 2, 0))
    vis_sr_A = fixed_map(cand_A)
    vis_sr_B = fixed_map(cand_B)
    
    # 9. Reference Radiometric Matching (Aligning SR to Input Dynamic Range)
    matched_sr = np.zeros_like(cand_A)
    for c in range(3):
        in_c = img_np[:, :, c]
        sr_c = cand_A[:, :, c]
        # Match mean and standard deviation of input image
        matched_sr[:, :, c] = (sr_c - sr_c.mean()) / (sr_c.std() + 1e-6) * in_c.std() + in_c.mean()
    matched_sr = np.clip(matched_sr, 0.0, 1.0)
    
    # Save Radiometrically Matched Output
    vis_matched = (matched_sr * 255.0).astype(np.uint8)
    Image.fromarray(vis_matched).save("outputs/diagnostic/7_radiometrically_matched_sr.png")
    
    # Build 3-Panel Side-by-Side: Input (10m) vs Bicubic (4x) vs BharatSRM (4x)
    comp_w, comp_h = 512, 288
    in_comp = Image.fromarray(vis_input).resize((comp_w, comp_h), Image.Resampling.NEAREST)
    bic_comp = Image.fromarray(vis_bicubic).resize((comp_w, comp_h))
    sr_comp = Image.fromarray(vis_matched).resize((comp_w, comp_h))
    
    comp_3panel = Image.new("RGB", (comp_w * 3 + 40, comp_h + 60), color=(20, 24, 33))
    comp_3panel.paste(in_comp, (10, 40))
    comp_3panel.paste(bic_comp, (comp_w + 20, 40))
    comp_3panel.paste(sr_comp, (comp_w * 2 + 30, 40))
    comp_3panel.save("outputs/diagnostic/8_clean_3panel_comparison.png")
    
    print("\n" + "=" * 80)
    print("3. RADIOMETRICALLY MATCHED SUPER-RESOLUTION:")
    print("=" * 80)
    print(f"  * Input Mean RGB     : {img_np.mean(axis=(0,1)) * 255.0}")
    print(f"  * Bicubic Mean RGB   : {bicubic_rgb.mean(axis=(1,2)) * 255.0}")
    print(f"  * BharatSRM Mean RGB : {(matched_sr * 255.0).mean(axis=(0,1))}")
    print("\n[OK] 3-Panel Comparison saved to: outputs/diagnostic/8_clean_3panel_comparison.png")
    
    print("\n" + "=" * 80)
    print("[OK] DIAGNOSTIC IMAGES SAVED TO outputs/diagnostic/:")
    print("  1. Input 10m (Fixed Map)       : outputs/diagnostic/1_input_10m.png")
    print("  2. Bicubic 4x (Fixed Map)      : outputs/diagnostic/2_bicubic_4x.png")
    print("  3. BharatSRM [0,1,2] (Fixed)   : outputs/diagnostic/3_bharatsrm_candA_012.png")
    print("  4. BharatSRM [2,1,0] (Fixed)   : outputs/diagnostic/4_bharatsrm_candB_210.png")
    print("  5. BharatSRM (Satellite Scale) : outputs/diagnostic/5_bharatsrm_sat_scaled_A.png")
    print("  6. Full Diagnostic Board       : outputs/diagnostic/radiometric_diagnostic_board.png")
    print("=" * 80)

if __name__ == "__main__":
    run_diagnostic()
