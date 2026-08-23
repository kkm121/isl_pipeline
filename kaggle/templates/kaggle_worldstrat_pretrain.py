"""
=============================================================================
BharatSRM-Net v4: Kaggle Dual Tesla T4 Pretraining Kernel (WorldStrat RGBN)
=============================================================================
Target: Sentinel-2 L2A 10m -> SPOT 6/7 1.5m Pansharpened RGBN (4x Scale, 2.5m nominal)
Hardware: NVIDIA Tesla T4 x2 (PyTorch 2.x, FP16 AMP)
=============================================================================
"""

import os
import sys
import time
import json
import math
import glob
import numpy as np
from pathlib import Path

# Add project source tree to python search path
sys.path.append('/kaggle/working')
sys.path.append('/kaggle/src')
sys.path.append('/kaggle/input/bharatsrm-v4-source')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

print("=" * 80)
print("=== BharatSRM-Net v4: WorldStrat RGBN Pretraining Kernel (Dual T4) ===")
print("=" * 80)

# 1. Hard CUDA Hardware Check
assert torch.cuda.is_available(), "Tesla T4 GPU required for training"
device = torch.device("cuda")
test_tensor = torch.zeros((1, 1), device=device)
torch.cuda.synchronize()
gpu_name = torch.cuda.get_device_name(0)
gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"✅ GPU ACTIVE: {gpu_name} | VRAM: {gpu_vram:.2f} GB | Count: {torch.cuda.device_count()}")

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. WorldStrat Stream Loader for Pretraining
class WorldStratStreamDataset(Dataset):
    def __init__(self, data_dir: str = "/kaggle/input/worldstrat", num_samples: int = 500, patch_size: int = 64):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.hr_size = patch_size * 4
        self.use_synthetic = True
        self.pairs = []
        
        # Search for Sentinel-2 (LR) and SPOT 6/7 (HR) image pairs
        possible_dirs = [
            data_dir,
            "/kaggle/input/worldstratdataset",
            "/kaggle/input/worldstrat_patched_data",
        ]
        
        for d in possible_dirs:
            if os.path.exists(d):
                # Look for matching scene files
                all_tifs = sorted(glob.glob(os.path.join(d, "**", "*.tif*"), recursive=True))
                lr_files = [f for f in all_tifs if any(k in f.lower() for k in ["_lr", "s2", "sentinel"])]
                hr_files = [f for f in all_tifs if any(k in f.lower() for k in ["_hr", "spot", "rgbn"])]
                
                # Pair based on matching directory or stem
                if len(lr_files) > 0 and len(hr_files) > 0:
                    # Match by common base identifier if possible
                    hr_map = {Path(f).stem.replace("_hr", "").replace("spot_", ""): f for f in hr_files}
                    for lr in lr_files:
                        stem = Path(lr).stem.replace("_lr", "").replace("s2_", "")
                        if stem in hr_map:
                            self.pairs.append((lr, hr_map[stem]))
                    
                    if len(self.pairs) == 0:
                        # Fallback to positional pairing if same count
                        self.pairs = list(zip(lr_files, hr_files))
                        
                    if len(self.pairs) > 0:
                        self.use_synthetic = False
                        print(f"✅ Found {len(self.pairs)} real WorldStrat LR/HR pairs in {d}")
                        break
                        
        if self.use_synthetic:
            print("⚠️ WorldStrat COG files not discovered in Kaggle mounts. Using structured synthetic dataset for Gate 1 pilot.")

    def __len__(self) -> int:
        return len(self.pairs) if not self.use_synthetic else self.num_samples

    def __getitem__(self, idx: int):
        if self.use_synthetic:
            lr = torch.rand(10, self.patch_size, self.patch_size, dtype=torch.float32) * 0.8 + 0.1
            hr = torch.rand(4, self.hr_size, self.hr_size, dtype=torch.float32) * 0.8 + 0.1
            mask = (torch.rand(1, self.patch_size, self.patch_size) > 0.15).float()
            dem = torch.rand(2, self.patch_size, self.patch_size, dtype=torch.float32)
            return {"lr_input": lr, "hr_target": hr, "validity_mask": mask, "context_dem": dem}
        else:
            import rasterio
            lr_path, hr_path = self.pairs[idx % len(self.pairs)]
            with rasterio.open(lr_path) as src_lr:
                raw_lr = src_lr.read().astype(np.float32)
            with rasterio.open(hr_path) as src_hr:
                raw_hr = src_hr.read().astype(np.float32)

            # Reflectance normalization (Sentinel-2 L2A is scaled by 10000)
            if np.nanmax(raw_lr) > 1.0:
                raw_lr = np.clip(raw_lr / 10000.0, 0.0, 1.0)
            if np.nanmax(raw_hr) > 1.0:
                raw_hr = np.clip(raw_hr / 10000.0, 0.0, 1.0)

            # Subset 10 Sentinel-2 bands if full 12/13 band stack is loaded
            # Drop atmospheric bands B1, B9, B10 if 13 bands
            if raw_lr.shape[0] == 13:
                # 10 bands: [B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12]
                band_indices = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]
                raw_lr = raw_lr[band_indices]
            elif raw_lr.shape[0] > 10:
                raw_lr = raw_lr[:10]
            elif raw_lr.shape[0] < 10:
                # Pad to 10 bands if 4-band LR
                pad_bands = np.repeat(raw_lr[:1], 10 - raw_lr.shape[0], axis=0)
                raw_lr = np.concatenate([raw_lr, pad_bands], axis=0)

            # HR SPOT 6/7 is 4 bands (RGBN)
            if raw_hr.shape[0] > 4:
                raw_hr = raw_hr[:4]

            # Crop central patch without resampling distortion
            c_lr, h_lr, w_lr = raw_lr.shape
            c_hr, h_hr, w_hr = raw_hr.shape

            if h_lr >= self.patch_size and w_lr >= self.patch_size:
                y0 = (h_lr - self.patch_size) // 2
                x0 = (w_lr - self.patch_size) // 2
                lr_patch = raw_lr[:, y0:y0+self.patch_size, x0:x0+self.patch_size]
            else:
                lr_t = torch.from_numpy(raw_lr).unsqueeze(0)
                lr_patch = torch.nn.functional.interpolate(lr_t, size=(self.patch_size, self.patch_size), mode='bilinear').squeeze(0).numpy()

            if h_hr >= self.hr_size and w_hr >= self.hr_size:
                y0_hr = (h_hr - self.hr_size) // 2
                x0_hr = (w_hr - self.hr_size) // 2
                hr_patch = raw_hr[:, y0_hr:y0_hr+self.hr_size, x0_hr:x0_hr+self.hr_size]
            else:
                hr_t = torch.from_numpy(raw_hr).unsqueeze(0)
                hr_patch = torch.nn.functional.interpolate(hr_t, size=(self.hr_size, self.hr_size), mode='bilinear').squeeze(0).numpy()

            # Validity mask: invalid where all zero / nodata
            mask = (np.mean(lr_patch, axis=0, keepdims=True) > 0.001).astype(np.float32)
            dem = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)

            return {
                "lr_input": torch.from_numpy(lr_patch).float(),
                "hr_target": torch.from_numpy(hr_patch).float(),
                "validity_mask": torch.from_numpy(mask).float(),
                "context_dem": torch.from_numpy(dem).float(),
            }

# 3. Model & Loss Setup
from src.models.bharatsrm_net import BharatSRMNetV4
from src.training.losses import CompositeBharatSRMLoss
from src.evaluation.metrics import evaluate_all_metrics

model = BharatSRMNetV4(
    in_spectral_bands=10,
    out_sr_bands=4,
    scale_factor=4,
    base_channels=64,
    use_context_stream=True,
    include_downstream_heads=False,
)

model = model.to(device)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = nn.DataParallel(model)

criterion = CompositeBharatSRMLoss(
    lambda_rec=1.0,
    lambda_spec=0.1,
    lambda_degrade=0.5,
    lambda_struct=0.2,
    lambda_conf=0.05,
)

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
EPOCHS = 10
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

train_dataset = WorldStratStreamDataset(num_samples=300, patch_size=64)
val_dataset = WorldStratStreamDataset(num_samples=50, patch_size=64)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

print(f"\nModel initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Starting WorldStrat Pretraining for {EPOCHS} Epochs...")

best_psnr = 0.0
log_file = open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0.0

    for batch in train_loader:
        lr = batch["lr_input"].to(device)
        hr = batch["hr_target"].to(device)
        mask = batch["validity_mask"].to(device)
        dem = batch["context_dem"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(lr, mask, dem)
            losses = criterion(out["sr_image"], hr, lr, out["log_variance"], epoch=epoch)
            loss = losses["loss_total"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item() * len(lr)

    scheduler.step()

    # Validation
    model.eval()
    val_psnr_list = []
    with torch.no_grad():
        for batch in val_loader:
            lr = batch["lr_input"].to(device)
            hr = batch["hr_target"].to(device)
            mask = batch["validity_mask"].to(device)
            dem = batch["context_dem"].to(device)

            with torch.amp.autocast("cuda"):
                out = model(lr, mask, dem)

            m = evaluate_all_metrics(out["sr_image"], hr)
            val_psnr_list.append(m["PSNR_mean"])

    mean_val_psnr = float(np.mean(val_psnr_list))
    log_msg = f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {tot_loss/len(train_dataset):.4f} | Val PSNR: {mean_val_psnr:.2f} dB"
    print(log_msg)
    log_file.write(log_msg + "\n")
    log_file.flush()

    if mean_val_psnr > best_psnr:
        best_psnr = mean_val_psnr
        raw_model = model.module if isinstance(model, nn.DataParallel) else model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
        }
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, "bharatsrm_v4_pretrained.pth"))

log_file.close()
print(f"\nPretraining Complete. Best Val PSNR: {best_psnr:.2f} dB")
