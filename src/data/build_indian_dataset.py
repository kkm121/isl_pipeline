import os
import glob
import numpy as np
import rasterio
from pathlib import Path
import argparse
from tqdm import tqdm
import cv2

# Local imports
from src.data.registration import register_image_pair, match_radiometry_histogram

AOI_CATALOG = {
    "indo_gangetic": {"lat": 30.7333, "lon": 76.7794},
    "western_ghats": {"lat": 10.8505, "lon": 76.2711},
    "peri_urban": {"lat": 28.5355, "lon": 77.3910},
    "rajasthan": {"lat": 26.9124, "lon": 75.7873},
}

def extract_overlapping_tiles(lr_img: np.ndarray, hr_img: np.ndarray, tile_size: int = 256, scale: int = 4):
    """Tiles co-registered scenes into discrete patch pairs."""
    tiles = []
    # hr_img shape should be (C, H*scale, W*scale), lr_img (C, H, W)
    _, h, w = lr_img.shape
    
    for y in range(0, h - tile_size + 1, tile_size):
        for x in range(0, w - tile_size + 1, tile_size):
            lr_tile = lr_img[:, y:y+tile_size, x:x+tile_size]
            
            # HR boundaries
            y_hr, x_hr = y * scale, x * scale
            hr_tile_size = tile_size * scale
            hr_tile = hr_img[:, y_hr:y_hr+hr_tile_size, x_hr:x_hr+hr_tile_size]
            
            # Skip if padded/incomplete
            if lr_tile.shape[1:] != (tile_size, tile_size) or hr_tile.shape[1:] != (hr_tile_size, hr_tile_size):
                continue
                
            # Skip if high nodata/cloud content (simple black/white thresholding)
            if np.mean(lr_tile) < 0.05 or np.mean(hr_tile) < 0.05:
                continue
                
            tiles.append((lr_tile, hr_tile))
    return tiles

def build_gate2_dataset(raw_dir: str, out_dir: str, max_error: float = 1.0):
    """
    Assembles the Gate 2 Indian Dataset.
    Expects raw_dir to have subfolders for each AOI containing Sentinel-2 (LR) and Bhuvan (HR) GeoTIFFs.
    """
    print(f"=== BharatSRM-Net v4: Gate 2 Dataset Assembly ===")
    print(f"Aligning Sentinel-2 & Bhuvan pairs using AKAZE/RANSAC...")
    
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
            
            if not hr_path.exists():
                continue
                
            with rasterio.open(lr_path) as src_lr, rasterio.open(hr_path) as src_hr:
                lr_img = src_lr.read()  # (C, H, W)
                hr_img = src_hr.read()  # (C, H*4, W*4)
                
                # Resample LR to HR dimensions purely for AKAZE feature matching
                # OpenCV expects (H, W, C) for warping
                lr_chw_scaled = cv2.resize(
                    np.moveaxis(lr_img, 0, -1), 
                    (hr_img.shape[2], hr_img.shape[1]), 
                    interpolation=cv2.INTER_CUBIC
                )
                hr_hwc = np.moveaxis(hr_img, 0, -1)
                
                # Run AKAZE + RANSAC registration
                try:
                    warped_lr, rms_error, is_valid = register_image_pair(
                        source=lr_chw_scaled, 
                        reference=hr_hwc, 
                        max_error_pixels=max_error
                    )
                except Exception as e:
                    print(f"\n[Error] Registration failed for {prefix}: {e}")
                    rejected_pairs += 1
                    continue
                
                if not is_valid:
                    # RANSAC residual > max_error_pixels (default 1.0)
                    rejected_pairs += 1
                    continue
                    
                # Radiometric histogram matching (match warped LR to HR domain for consistency)
                warped_lr_matched = match_radiometry_histogram(warped_lr, hr_hwc)
                
                # Convert back to (C, H, W) and original LR scale
                warped_lr_chw = np.moveaxis(warped_lr_matched, -1, 0)
                final_lr = cv2.resize(
                    warped_lr_matched, 
                    (lr_img.shape[2], lr_img.shape[1]), 
                    interpolation=cv2.INTER_AREA
                )
                final_lr_chw = np.moveaxis(final_lr, -1, 0)
                
                # Extract 256x256 tiles
                tiles = extract_overlapping_tiles(final_lr_chw, hr_img, tile_size=256, scale=4)
                
                for idx, (t_lr, t_hr) in enumerate(tiles):
                    np.savez_compressed(
                        out_path / aoi / f"{prefix}_tile_{idx}.npz",
                        lr=t_lr,
                        hr=t_hr
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
