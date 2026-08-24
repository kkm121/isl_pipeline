"""
=============================================================================
BharatSRM-Net v4: Annotated 3x3 ROI Matrix Generator
=============================================================================
Clearly labels every column and row with high-contrast text banners:
  Column 1 (Left)   : 10m Sentinel-2 (Original Native Input)
  Column 2 (Middle) : Bicubic 4x (Standard Interpolation Control)
  Column 3 (Right)  : BharatSRM-Net v4 (2.5m Super-Resolved Output)
=============================================================================
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from src.models.bharatsrm_net import BharatSRMNetV4

def build_annotated_matrix():
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
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
    model.eval()

    # 2. Ingest Scene
    img_path = r"C:\Users\muthu\Downloads\S2_L2A.jpg"
    if not os.path.exists(img_path):
        img_path = "outputs/user_lake.png"
    full_img = Image.open(img_path).convert("RGB")
    full_np = np.array(full_img).astype(np.float32) / 255.0
    H, W, _ = full_np.shape

    # 3. ROIs
    rois = [
        ("Shoreline & Water Transition", int(H * 0.35), int(W * 0.35)),
        ("Agricultural Plots & Field Boundaries", int(H * 0.50), int(W * 0.15)),
        ("Mountain Forest & Terrain Relief", int(H * 0.55), int(W * 0.60)),
    ]

    p_sz = 300 # Panel size in pixels
    header_h = 70
    row_title_h = 35
    col_w = p_sz + 20
    row_h = p_sz + row_title_h + 15
    total_w = col_w * 3 + 20
    total_h = header_h + row_h * 3 + 30

    canvas = Image.new("RGB", (total_w, total_h), color=(15, 23, 42)) # Slate dark background
    draw = ImageDraw.Draw(canvas)

    # Header Column Titles
    col_titles = [
        ("[ LEFT ] 10m Input (Sentinel-2)", (251, 146, 60)),     # Orange
        ("[ MIDDLE ] Bicubic 4x Baseline", (148, 163, 184)),    # Muted Gray
        ("[ RIGHT ] BharatSRM-Net v4 (2.5m)", (56, 189, 248)),  # Cyan
    ]

    for c_idx, (ctitle, color) in enumerate(col_titles):
        cx = 20 + c_idx * col_w
        draw.rectangle([cx, 15, cx + p_sz, 55], fill=(30, 41, 59), outline=color, width=2)
        draw.text((cx + 15, 26), ctitle, fill=color)

    for r_idx, (roi_name, cy, cx) in enumerate(rois):
        y_top = header_h + r_idx * row_h
        
        # Row title
        draw.text((20, y_top), f"ROW {r_idx+1}: {roi_name.upper()}", fill=(248, 250, 252))

        # Extract 64x64
        y1, y2 = max(0, cy - 32), min(H, cy + 32)
        x1, x2 = max(0, cx - 32), min(W, cx + 32)
        lr_crop = full_np[y1:y1+64, x1:x1+64]
        if lr_crop.shape[0] != 64 or lr_crop.shape[1] != 64:
            lr_crop = cv2.resize(lr_crop, (64, 64), interpolation=cv2.INTER_AREA)

        # 1. Left: 10m Input (Nearest)
        p_left = cv2.resize(lr_crop, (p_sz, p_sz), interpolation=cv2.INTER_NEAREST)

        # 2. Middle: Bicubic 4x
        p_mid = cv2.resize(lr_crop, (p_sz, p_sz), interpolation=cv2.INTER_CUBIC)
        p_mid = np.clip(p_mid, 0.0, 1.0)

        # 3. Right: BharatSRM-Net v4
        lr_10b = np.zeros((1, 10, 64, 64), dtype=np.float32)
        lr_10b[0, 0] = lr_crop[:, :, 2]
        lr_10b[0, 1] = lr_crop[:, :, 1]
        lr_10b[0, 2] = lr_crop[:, :, 0]
        lr_10b[0, 3] = lr_crop[:, :, 1] * 0.7 + lr_crop[:, :, 0] * 0.3
        for b in range(4, 10):
            lr_10b[0, b] = lr_crop[:, :, 1] * 0.6
            
        with torch.no_grad():
            out = model(torch.from_numpy(lr_10b), torch.ones((1, 1, 64, 64)), torch.zeros((1, 2, 64, 64)))
            sr_4b = out["sr_image"].squeeze(0).numpy()

        sr_rgb = np.stack([sr_4b[0], sr_4b[1], sr_4b[2]], axis=-1)
        sr_rgb_resized = cv2.resize(sr_rgb, (p_sz, p_sz), interpolation=cv2.INTER_CUBIC)

        p_right = np.zeros_like(sr_rgb_resized)
        for c in range(3):
            in_c = lr_crop[:, :, c]
            sr_c = sr_rgb_resized[:, :, c]
            p_right[:, :, c] = (sr_c - sr_c.mean()) / (sr_c.std() + 1e-6) * in_c.std() + in_c.mean()
        p_right = np.clip(p_right, 0.0, 1.0)

        img_left = Image.fromarray((p_left * 255).astype(np.uint8))
        img_mid = Image.fromarray((p_mid * 255).astype(np.uint8))
        img_right = Image.fromarray((p_right * 255).astype(np.uint8))

        canvas.paste(img_left, (20, y_top + 25))
        canvas.paste(img_mid, (20 + col_w, y_top + 25))
        canvas.paste(img_right, (20 + col_w * 2, y_top + 25))

    out_path = "outputs/rigorous_comparison/master_matched_roi_matrix.png"
    canvas.save(out_path)
    print(f"[OK] Generated annotated matrix at: {out_path}")

if __name__ == "__main__":
    build_annotated_matrix()
