import os
import glob
import numpy as np
import rasterio
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2
import torch
import torch.nn.functional as F

# Local imports
from src.data.registration import register_image_pair, match_radiometry_histogram

AOI_CATALOG = {
    "indo_gangetic": {"lat": 30.7333, "lon": 76.7794},
    "western_ghats": {"lat": 10.8505, "lon": 76.2711},
    "peri_urban": {"lat": 28.5355, "lon": 77.3910},
    "rajasthan": {"lat": 26.9124, "lon": 75.7873},
}

def resize_multiband(tensor_chw: np.ndarray, target_h: int, target_w: int, mode: str = "bilinear") -> np.ndarray:
    """Safely resizes multi-band satellite rasters (any number of channels) using PyTorch tensor interpolation."""
    t = torch.from_numpy(tensor_chw).unsqueeze(0).float()
    align_corners = False if mode in ["bilinear", "bicubic"] else None
    resized = F.interpolate(t, size=(target_h, target_w), mode=mode, align_corners=align_corners)
    return resized.squeeze(0).numpy().astype(tensor_chw.dtype)

def extract_overlapping_tiles(
    lr_img: np.ndarray, 
    hr_img: np.ndarray, 
    dem_img: np.ndarray | None = None,
    tile_size: int = 256, 
    scale: int = 4
):
    """Tiles co-registered scenes into discrete patch pairs."""
    tiles = []
    # hr_img shape: (C_hr, H*scale, W*scale), lr_img: (C_lr, H, W)
    _, h, w = lr_img.shape
    
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            lr_tile = lr_img[:, y:y+tile_size, x:x+tile_size]
            
            # HR boundaries
            y_hr, x_hr = y * scale, x * scale
            hr_tile_size = tile_size * scale
            hr_tile = hr_img[:, y_hr:y_hr+hr_tile_size, x_hr:x_hr+hr_tile_size]
            
            # DEM tile if available: (2, tile_size, tile_size)
            if dem_img is not None:
                dem_tile = dem_img[:, y:y+tile_size, x:x+tile_size]
            else:
                dem_tile = np.zeros((2, tile_size, tile_size), dtype=np.float32)
            
            # Skip if padded/incomplete
            if lr_tile.shape[1:] != (tile_size, tile_size) or hr_tile.shape[1:] != (hr_tile_size, hr_tile_size):
                continue
                
            # Skip if high nodata/cloud content (simple black/white thresholding)
            if np.mean(lr_tile) < 0.01 or np.mean(hr_tile) < 0.01:
                continue
                
            tiles.append((lr_tile, hr_tile, dem_tile))
    return tiles

def harmonize_cross_sensor_spectra(s2_lr: np.ndarray, bhuvan_hr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Physically harmonizes spectral bands between Sentinel-2 and ISRO Bhuvan (LISS-IV or Cartosat-RGBN).
    - Sentinel-2: B2 (Blue=0), B3 (Green=1), B4 (Red=2), B8 (NIR=3), plus red-edge/SWIR bands (total 10)
    - Bhuvan LISS-IV (3-band): Green (0), Red (1), NIR (2)
    - Bhuvan/WorldStrat 4-band: Red (0), Green (1), Blue (2), NIR (3) or RGBN
    """
    s2_c, _, _ = s2_lr.shape
    hr_c, _, _ = bhuvan_hr.shape

    if hr_c == 3:
        # Bhuvan 3-band (Green, Red, NIR) matched to Sentinel-2 Green (B3=1), Red (B4=2), NIR (B8=3)
        s2_matched = np.copy(s2_lr)
        if s2_c >= 4:
            s2_matched[1] = match_radiometry_histogram(s2_lr[1], bhuvan_hr[0]) # Green
            s2_matched[2] = match_radiometry_histogram(s2_lr[2], bhuvan_hr[1]) # Red
            s2_matched[3] = match_radiometry_histogram(s2_lr[3], bhuvan_hr[2]) # NIR
        return s2_matched, bhuvan_hr
    elif hr_c == 4:
        # 4-band RGBN matching
        s2_matched = np.copy(s2_lr)
        for i in range(min(4, s2_c)):
            s2_matched[i] = match_radiometry_histogram(s2_lr[i], bhuvan_hr[i])
        return s2_matched, bhuvan_hr
    else:
        return s2_lr, bhuvan_hr

def build_gate2_dataset(raw_dir: str, out_dir: str, max_error: float = 1.0):
    """
    Assembles the Gate 2 Indian Dataset.
    Expects raw_dir to have subfolders for each AOI containing Sentinel-2 (LR) and Bhuvan (HR) GeoTIFFs.
    """
    print(f"=== BharatSRM-Net v4: Gate 2 Indian Dataset Assembly ===")
    print(f"Aligning Sentinel-2 & Bhuvan pairs using AKAZE/RANSAC and wavelength-aware harmonization...")
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    total_valid_pairs = 0
    rejected_pairs = 0
    
    for aoi in AOI_CATALOG.keys():
        aoi_raw_dir = Path(raw_dir) / aoi
        if not aoi_raw_dir.exists():
            print(f"[Warning] AOI '{aoi}' not found in {raw_dir}. Skipping.")
            continue
            
        print(f"\nProcessing AOI: {aoi}")
        (out_path / aoi).mkdir(exist_ok=True)
        
        # Discover pairs (assuming naming convention: *LR.tif and *HR.tif with matching prefixes)
        lr_files = sorted(aoi_raw_dir.glob("*_LR.tif"))
        for lr_path in tqdm(lr_files, desc="Registering Pairs"):
            prefix = lr_path.name.replace("_LR.tif", "")
            hr_path = aoi_raw_dir / f"{prefix}_HR.tif"
            dem_path = aoi_raw_dir / f"{prefix}_DEM.tif"
            
            if not hr_path.exists():
                continue
                
            with rasterio.open(lr_path) as src_lr, rasterio.open(hr_path) as src_hr:
                lr_img = src_lr.read().astype(np.float32)  # (C_lr, H, W)
                hr_img = src_hr.read().astype(np.float32)  # (C_hr, H*4, W*4)
                
                # Optional CartoDEM: Elevation (Band 1), Slope (Band 2)
                if dem_path.exists():
                    with rasterio.open(dem_path) as src_dem:
                        dem_img = src_dem.read().astype(np.float32)
                else:
                    dem_img = None
                
                # Safe multi-band scaling to HR dimension for AKAZE feature extraction
                lr_scaled = resize_multiband(lr_img, hr_img.shape[1], hr_img.shape[2], mode="bilinear")
                
                # Run AKAZE + RANSAC registration using Red band for geometric saliency
                try:
                    warped_lr, rms_error, is_valid = register_image_pair(
                        source=lr_scaled, 
                        reference=hr_img, 
                        max_error_pixels=max_error,
                        feature_band_idx_src=2, # S2 Red band
                        feature_band_idx_ref=1 if hr_img.shape[0] == 3 else 0, # Bhuvan Red band
                    )
                except Exception as e:
                    print(f"\n[Error] Registration failed for {prefix}: {e}")
                    rejected_pairs += 1
                    continue
                
                if not is_valid:
                    # RANSAC residual > max_error_pixels (default 1.0)
                    rejected_pairs += 1
                    continue
                    
                # Downsample warped LR back to native LR dimensions
                final_lr = resize_multiband(warped_lr, lr_img.shape[1], lr_img.shape[2], mode="area")
                
                # Spectral radiometry harmonization
                final_lr_matched, hr_final = harmonize_cross_sensor_spectra(final_lr, hr_img)
                
                # Extract 256x256 tiles
                tiles = extract_overlapping_tiles(final_lr_matched, hr_final, dem_img=dem_img, tile_size=256, scale=4)
                
                for idx, (t_lr, t_hr, t_dem) in enumerate(tiles):
                    np.savez_compressed(
                        out_path / aoi / f"{prefix}_tile_{idx}.npz",
                        lr=t_lr,
                        hr=t_hr,
                        dem=t_dem
                    )
                    total_valid_pairs += 1

    print("\n=== Gate 2 Assembly Complete ===")
    print(f"Successfully tiled and registered pairs: {total_valid_pairs}")
    print(f"Rejected pairs (sub-pixel alignment failed): {rejected_pairs}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gate 2: Indian Dataset Assembly")
    parser.add_argument("--raw_dir", type=str, default="data/raw_indian", help="Path to raw Sentinel/Bhuvan AOI folders")
    parser.add_argument("--out_dir", type=str, default="data/gate2_dataset", help="Output path for aligned tiles")
    parser.add_argument("--max_error", type=float, default=1.0, help="Max RANSAC RMS error in pixels")
    args = parser.parse_args()
    
    build_gate2_dataset(args.raw_dir, args.out_dir, args.max_error)
