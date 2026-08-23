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

# 2. Inspect Kaggle Mount Tree
print("\n" + "=" * 80)
print("=== Kaggle Input Mount Inspection ===")
print("=" * 80)
if os.path.exists("/kaggle/input"):
    for root, dirs, files in os.walk("/kaggle/input"):
        depth = root.replace("/kaggle/input", "").count(os.sep)
        if depth <= 3:
            print(f"{'  ' * depth}📁 {root} (Dirs: {len(dirs)}, Files: {len(files)})")
            if len(files) > 0 and depth <= 3:
                print(f"{'  ' * (depth + 1)}Sample files: {files[:5]}")
else:
    print("⚠️ /kaggle/input does not exist!")

# 3. Fast One-Pass WorldStrat File Pair Discovery
def discover_worldstrat_pairs(base_dir: str = "/kaggle/input") -> list[tuple[str, str]]:
    """Discovers paired Low-Res (Sentinel-2) and High-Res (SPOT 6/7) imagery in one fast scan."""
    valid_exts = {".tif", ".tiff", ".TIF", ".TIFF", ".png", ".npz", ".npy"}
    all_files = []
    
    for root, _, files in os.walk(base_dir):
        # Ignore our source code dataset
        if "bharatsrm-v4-source" in root:
            continue
        for f in files:
            ext = os.path.splitext(f)[1]
            if ext in valid_exts:
                all_files.append(os.path.join(root, f))
                
    print(f"\n🔍 Discovered total candidate raster files in {base_dir}: {len(all_files)}")
    
    lr_files = []
    hr_files = []
    
    for f in all_files:
        f_lower = f.lower()
        if any(k in f_lower for k in ["/lr/", "_lr", "s2", "sentinel", "l2a", "low_res"]):
            lr_files.append(f)
        elif any(k in f_lower for k in ["/hr/", "_hr", "spot", "rgbn", "high_res", "pan"]):
            hr_files.append(f)
            
    print(f"  -> Low-Res (Sentinel-2) candidates : {len(lr_files)}")
    print(f"  -> High-Res (SPOT 6/7) candidates  : {len(hr_files)}")
    
    pairs = []
    # Match by key identifier
    hr_map = {Path(f).stem.replace("_hr", "").replace("spot_", "").replace("_spot", ""): f for f in hr_files}
    for lr in lr_files:
        lr_key = Path(lr).stem.replace("_lr", "").replace("s2_", "").replace("_s2", "")
        if lr_key in hr_map:
            pairs.append((lr, hr_map[lr_key]))
            
    if len(pairs) == 0 and len(lr_files) > 0 and len(hr_files) > 0:
        # If naming differs, pair matching folders or positional sort
        lr_sorted = sorted(lr_files)
        hr_sorted = sorted(hr_files)
        pairs = list(zip(lr_sorted, hr_sorted))
        print(f"  -> Paired {len(pairs)} scenes positionally by sorted paths.")
        
    return pairs

# 4. Stream Dataset
class WorldStratPairedDataset(Dataset):
    def __init__(self, pairs: list[tuple[str, str]], patch_size: int = 64):
        self.pairs = pairs
        self.patch_size = patch_size
        self.hr_size = patch_size * 4

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        import rasterio
        lr_path, hr_path = self.pairs[idx]
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

        # Center crop patch
        _, h_lr, w_lr = raw_lr.shape
        _, h_hr, w_hr = raw_hr.shape

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

        mask = (np.mean(lr_patch, axis=0, keepdims=True) > 0.001).astype(np.float32)
        dem = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)

        return {
            "lr_input": torch.from_numpy(lr_patch).float(),
            "hr_target": torch.from_numpy(hr_patch).float(),
            "validity_mask": torch.from_numpy(mask).float(),
            "context_dem": torch.from_numpy(dem).float(),
        }

# 5. Discover Dataset & Partition
all_discovered_pairs = discover_worldstrat_pairs()
if len(all_discovered_pairs) == 0:
    raise RuntimeError(
        "GATE 1 CRITICAL FAILURE: Zero real WorldStrat Sentinel-2/SPOT image pairs were discovered in /kaggle/input! "
        "Please attach the WorldStrat dataset (jucor1/worldstrat) to this kernel."
    )

print(f"\n✅ GATE 1 VERIFIED: Discovered {len(all_discovered_pairs)} real WorldStrat LR/HR scene pairs.")

# 80/20 train/val split
split_idx = int(0.8 * len(all_discovered_pairs))
train_pairs = all_discovered_pairs[:split_idx]
val_pairs = all_discovered_pairs[split_idx:]
if len(val_pairs) == 0:
    val_pairs = train_pairs[:min(50, len(train_pairs))]

print(f"Dataset split: {len(train_pairs)} Train pairs | {len(val_pairs)} Validation pairs")

train_dataset = WorldStratPairedDataset(train_pairs, patch_size=64)
val_dataset = WorldStratPairedDataset(val_pairs, patch_size=64)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True, drop_last=True if len(train_dataset) > 16 else False)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, pin_memory=True)

# 6. Model & Loss Setup
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
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
EPOCHS = 10
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

print(f"\nModel initialized: {sum(p.numel() for p in model.parameters()):,} parameters (GroupNorm configured)")
print(f"Starting WorldStrat Pretraining for {EPOCHS} Epochs on real data...")

best_psnr = -float('inf')
log_file = open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0.0
    valid_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        lr = batch["lr_input"].to(device)
        hr = batch["hr_target"].to(device)
        mask = batch["validity_mask"].to(device)
        dem = batch["context_dem"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(lr, mask, dem)
            losses = criterion(out["sr_image"], hr, lr, out["log_variance"], epoch=epoch)
            loss = losses["loss_total"]

        # Assert no NaNs during forward pass
        if torch.isnan(loss) or torch.isinf(loss):
            raise RuntimeError(f"NaN/Inf loss detected at Epoch {epoch}, Batch {batch_idx}! Loss breakdown: {losses}")

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item() * len(lr)
        valid_batches += len(lr)

    scheduler.step()

    # Validation Loop with Strict NaN Detection
    model.eval()
    val_psnr_list = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            lr = batch["lr_input"].to(device)
            hr = batch["hr_target"].to(device)
            mask = batch["validity_mask"].to(device)
            dem = batch["context_dem"].to(device)

            with torch.amp.autocast("cuda"):
                out = model(lr, mask, dem)

            # Strict NaN assertion
            if torch.isnan(out["sr_image"]).any():
                raise RuntimeError(f"NaN detected in model.eval() sr_image output at Epoch {epoch}, Batch {batch_idx}!")

            m = evaluate_all_metrics(out["sr_image"].float(), hr.float())
            psnr_val = m["PSNR_mean"]
            val_psnr_list.append(psnr_val)

    mean_val_psnr = float(np.mean(val_psnr_list))
    avg_train_loss = tot_loss / valid_batches if valid_batches > 0 else 0.0
    log_msg = f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {avg_train_loss:.4f} | Val PSNR: {mean_val_psnr:.2f} dB"
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
