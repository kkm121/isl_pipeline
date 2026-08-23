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
import zipfile
import numpy as np
from pathlib import Path

# Extract latest source code archive if present in dataset mount
for zip_candidate in [
    "/kaggle/input/bharatsrm-v4-source/bharatsrm_src.zip",
    "/kaggle/input/bharatsrm-v4-source/src.zip",
]:
    if os.path.exists(zip_candidate):
        with zipfile.ZipFile(zip_candidate, "r") as z:
            z.extractall("/kaggle/working")
        print(f"✅ Extracted source codebase from {zip_candidate} to /kaggle/working")
        break

# Add project source tree to python search path (prioritizing /kaggle/working)
sys.path.insert(0, '/kaggle/working')
sys.path.insert(1, '/kaggle/src')
sys.path.insert(2, '/kaggle/input/bharatsrm-v4-source')

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

# 2. Strict Scene-ID WorldStrat File Pair Discovery (Zero Positional Fallback)
def discover_worldstrat_pairs() -> list[tuple[str, str]]:
    """Discovers matching Low-Res (Sentinel-2) and High-Res (SPOT 6/7) pairs strictly by matching scene ID."""
    hr_base = "/kaggle/input/worldstrat/hr_dataset"
    lr_base = "/kaggle/input/worldstrat/lr_dataset"
    pairs = []
    
    if not (os.path.exists(hr_base) and os.path.exists(lr_base)):
        raise RuntimeError(f"WorldStrat datasets not found at expected paths: {hr_base} or {lr_base}")

    print("Indexing WorldStrat HR and LR datasets...")
    hr_files = []
    for root, _, files in os.walk(hr_base):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.png', '.npy', '.npz')):
                hr_files.append(os.path.join(root, f))
                
    lr_files = []
    for root, _, files in os.walk(lr_base):
        for f in files:
            if f.lower().endswith(('.tif', '.tiff', '.png', '.npy', '.npz')):
                lr_files.append(os.path.join(root, f))
                
    print(f"✅ Found {len(hr_files)} High-Resolution candidate files and {len(lr_files)} Low-Resolution candidate files.")
    
    # Index HR files strictly by scene ID
    hr_by_scene = {}
    for p in hr_files:
        parts = Path(p).parts
        if "12bit" in parts:
            idx = parts.index("12bit")
            if idx + 1 < len(parts):
                hr_by_scene[parts[idx + 1]] = p
        elif "hr_dataset" in parts:
            idx = parts.index("hr_dataset")
            if idx + 1 < len(parts):
                hr_by_scene[parts[idx + 1]] = p

    # Index LR files strictly by scene ID
    lr_by_scene = {}
    for p in lr_files:
        parts = Path(p).parts
        if "lr_dataset" in parts:
            idx = parts.index("lr_dataset")
            if idx + 1 < len(parts):
                scene = parts[idx + 1]
                if scene not in lr_by_scene:
                    lr_by_scene[scene] = p

    for scene, hr_p in hr_by_scene.items():
        if scene in lr_by_scene:
            pairs.append((lr_by_scene[scene], hr_p))
            
    print(f"✅ Strictly paired {len(pairs)} real WorldStrat Sentinel-2/SPOT 6/7 scenes by matching scene IDs.")
    
    if len(pairs) == 0:
        raise RuntimeError("FATAL: Zero matching scene IDs found between HR and LR datasets. Positional fallback is strictly forbidden.")
        
    return pairs

# 3. Stream Dataset with Strict Modality & Physical Invariance Validation
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
            lr_dtype = src_lr.dtypes[0]
        with rasterio.open(hr_path) as src_hr:
            raw_hr = src_hr.read().astype(np.float32)
            hr_dtype = src_hr.dtypes[0]

        # 1. Strict Modality Validation: Reject non-4-band HR imagery
        if raw_hr.shape[0] != 4:
            # Skip invalid modalities by fetching next valid sample
            next_idx = (idx + 1) % len(self.pairs)
            return self.__getitem__(next_idx)

        # 2. Strict Sentinel-2 LR Band Mapping
        if raw_lr.shape[0] == 13:
            # 13 bands -> 10 bands: [B2, B3, B4, B8, B5, B6, B7, B8A, B11, B12]
            band_indices = [1, 2, 3, 7, 4, 5, 6, 8, 11, 12]
            raw_lr = raw_lr[band_indices]
        elif raw_lr.shape[0] == 12:
            band_indices = [1, 2, 3, 7, 4, 5, 6, 8, 10, 11]
            raw_lr = raw_lr[band_indices]
        elif raw_lr.shape[0] == 10:
            pass  # Already standard 10 bands
        else:
            # Reject invalid LR modality
            next_idx = (idx + 1) % len(self.pairs)
            return self.__getitem__(next_idx)

        # 3. Clean NaNs, Infs, and negative nodata values
        raw_lr = np.nan_to_num(raw_lr, nan=0.0, posinf=1.0, neginf=0.0)
        raw_hr = np.nan_to_num(raw_hr, nan=0.0, posinf=1.0, neginf=0.0)
        raw_lr = np.clip(raw_lr, 0.0, None)
        raw_hr = np.clip(raw_hr, 0.0, None)

        # 4. Explicit Reflectance Normalization based on data type
        if lr_dtype in ['uint16', 'int16', 'uint12']:
            raw_lr = raw_lr / 10000.0
        elif np.max(raw_lr) > 10.0:
            raw_lr = raw_lr / 10000.0

        if hr_dtype in ['uint16', 'int16', 'uint12']:
            hr_max = float(np.max(raw_hr))
            if hr_max > 4095.0:
                raw_hr = raw_hr / 10000.0
            elif hr_max > 1.0:
                raw_hr = raw_hr / 4095.0
        elif np.max(raw_hr) > 10.0:
            raw_hr = raw_hr / 10000.0

        raw_lr = np.clip(raw_lr, 0.0, 1.0)
        raw_hr = np.clip(raw_hr, 0.0, 1.0)

        # 5. Center crop patch
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

        # Clean patch values once more
        lr_patch = np.nan_to_num(lr_patch, nan=0.0, posinf=1.0, neginf=0.0)
        hr_patch = np.nan_to_num(hr_patch, nan=0.0, posinf=1.0, neginf=0.0)

        mask = (np.mean(lr_patch, axis=0, keepdims=True) > 0.001).astype(np.float32)
        mask = np.nan_to_num(mask, nan=0.0)
        dem = np.zeros((2, self.patch_size, self.patch_size), dtype=np.float32)

        return {
            "lr_input": torch.from_numpy(lr_patch).float(),
            "hr_target": torch.from_numpy(hr_patch).float(),
            "validity_mask": torch.from_numpy(mask).float(),
            "context_dem": torch.from_numpy(dem).float(),
        }

# 4. Discover Dataset & Partition
all_discovered_pairs = discover_worldstrat_pairs()
print(f"\n✅ GATE 1 DISCOVERY: Found {len(all_discovered_pairs)} matching WorldStrat scene pairs.")

# 80/20 train/val split
split_idx = int(0.8 * len(all_discovered_pairs))
train_pairs = all_discovered_pairs[:split_idx]
val_pairs = all_discovered_pairs[split_idx:]
if len(val_pairs) == 0:
    val_pairs = train_pairs[:min(50, len(train_pairs))]

print(f"Dataset split: {len(train_pairs)} Train pairs | {len(val_pairs)} Validation pairs")

train_dataset = WorldStratPairedDataset(train_pairs, patch_size=64)
val_dataset = WorldStratPairedDataset(val_pairs, patch_size=64)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True, num_workers=4, drop_last=True if len(train_dataset) > 32 else False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True, num_workers=4)

# 5. Model & Loss Setup
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

criterion = CompositeBharatSRMLoss(
    lambda_rec=1.0,
    lambda_spec=0.1,
    lambda_degrade=0.5,
    lambda_struct=0.2,
    lambda_conf=0.05,
).to(device)

# =========================================================================
# 6. MANDATORY SINGLE-BATCH DIAGNOSTIC IN PURE FP32 (GATE 1 VERIFICATION)
# =========================================================================
print("\n" + "=" * 80)
print("=== GATE 1 SINGLE-BATCH DIAGNOSTIC (PURE FP32) ===")
print("=" * 80)

model.eval()
diag_batch = next(iter(train_loader))
d_lr = diag_batch["lr_input"].to(device)
d_hr = diag_batch["hr_target"].to(device)
d_mask = diag_batch["validity_mask"].to(device)
d_dem = diag_batch["context_dem"].to(device)

print(f"Batch LR shape: {list(d_lr.shape)} | Range: [{d_lr.min().item():.4f}, {d_lr.max().item():.4f}] | Finite: {torch.isfinite(d_lr).all().item()}")
print(f"Batch HR shape: {list(d_hr.shape)} | Range: [{d_hr.min().item():.4f}, {d_hr.max().item():.4f}] | Finite: {torch.isfinite(d_hr).all().item()}")
print(f"Batch Mask shape: {list(d_mask.shape)} | Finite: {torch.isfinite(d_mask).all().item()}")

with torch.no_grad():
    with torch.autocast(device_type="cuda", enabled=False):
        diag_out = model(d_lr.float(), d_mask.float(), d_dem.float())

d_sr = diag_out["sr_image"]
d_logvar = diag_out["log_variance"]
print(f"\nModel Output SR shape: {list(d_sr.shape)} | Range: [{d_sr.min().item():.4f}, {d_sr.max().item():.4f}] | Finite: {torch.isfinite(d_sr).all().item()}")
print(f"Model Output LogVar shape: {list(d_logvar.shape)} | Range: [{d_logvar.min().item():.4f}, {d_logvar.max().item():.4f}] | Finite: {torch.isfinite(d_logvar).all().item()}")

assert torch.isfinite(d_sr).all().item(), "FATAL: Model produced non-finite SR outputs in pure FP32!"
assert torch.isfinite(d_logvar).all().item(), "FATAL: Model produced non-finite log-variance outputs in pure FP32!"

# Compute every individual loss component in FP32
with torch.no_grad():
    l_rec = criterion.charbonnier(d_sr, d_hr).item()
    l_spec = criterion.sam(d_sr, d_hr).item()
    l_degrade = criterion.degrade(d_sr, d_lr).item()
    l_struct = criterion.struct(d_sr, d_hr).item()
    l_conf = criterion.conf(d_sr, d_hr, d_logvar).item()
    comp_loss = criterion(d_sr, d_hr, d_lr, d_logvar, epoch=1)["loss_total"].item()

print(f"\n📊 INDIVIDUAL LOSS COMPONENT BREAKDOWN (PURE FP32):")
print(f"  • L_rec (Charbonnier)          : {l_rec:.5f} (Finite: {math.isfinite(l_rec)})")
print(f"  • L_spec (Spectral Angle Mapper): {l_spec:.5f} (Finite: {math.isfinite(l_spec)})")
print(f"  • L_degrade (MTF Consistency)  : {l_degrade:.5f} (Finite: {math.isfinite(l_degrade)})")
print(f"  • L_struct (SSIM + Sobel Edge) : {l_struct:.5f} (Finite: {math.isfinite(l_struct)})")
print(f"  • L_conf (Heteroscedastic NLL) : {l_conf:.5f} (Finite: {math.isfinite(l_conf)})")
print(f"  • TOTAL COMPOSITE LOSS         : {comp_loss:.5f} (Finite: {math.isfinite(comp_loss)})")

assert all(math.isfinite(v) for v in [l_rec, l_spec, l_degrade, l_struct, l_conf, comp_loss]), "FATAL: Non-finite loss component detected in FP32!"
print("\n✅ GATE 1 MATHEMATICAL & DATASET DIAGNOSTIC PASSED IN PURE FP32!")
print("=" * 80 + "\n")

# Wrap in DataParallel if dual GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = nn.DataParallel(model)

optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
EPOCHS = 10
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = torch.amp.GradScaler("cuda")

print(f"Starting WorldStrat Pretraining for {EPOCHS} Epochs on real data...")

best_psnr = -float('inf')
log_file = open(os.path.join(OUTPUT_DIR, "training_log.txt"), "w")

for epoch in range(1, EPOCHS + 1):
    model.train()
    tot_loss = 0.0
    tot_rec, tot_spec, tot_degrade, tot_struct, tot_conf = 0.0, 0.0, 0.0, 0.0, 0.0
    valid_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        lr = batch["lr_input"].to(device)
        hr = batch["hr_target"].to(device)
        mask = batch["validity_mask"].to(device)
        dem = batch["context_dem"].to(device)

        lr = torch.nan_to_num(lr, nan=0.0, posinf=1.0, neginf=0.0)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=1.0, neginf=0.0)
        mask = torch.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=1.0)
        dem = torch.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            out = model(lr, mask, dem)
            losses = criterion(out["sr_image"], hr, lr, out["log_variance"], epoch=epoch)
            loss = losses["loss_total"]

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️ Non-finite loss at Epoch {epoch}, Batch {batch_idx}. Skipping.")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        bs = len(lr)
        tot_loss += loss.item() * bs
        tot_rec += losses["loss_rec"].item() * bs
        tot_spec += losses["loss_spec"].item() * bs
        tot_degrade += losses["loss_degrade"].item() * bs
        tot_struct += losses["loss_struct"].item() * bs
        tot_conf += losses["loss_conf"].item() * bs
        valid_batches += bs

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

            lr = torch.nan_to_num(lr, nan=0.0, posinf=1.0, neginf=0.0)
            hr = torch.nan_to_num(hr, nan=0.0, posinf=1.0, neginf=0.0)
            mask = torch.nan_to_num(mask, nan=1.0, posinf=1.0, neginf=1.0)
            dem = torch.nan_to_num(dem, nan=0.0, posinf=0.0, neginf=0.0)

            with torch.amp.autocast("cuda"):
                out = model(lr, mask, dem)

            if torch.isnan(out["sr_image"]).any():
                raise RuntimeError(f"NaN detected in model.eval() sr_image output at Epoch {epoch}, Batch {batch_idx}!")

            m = evaluate_all_metrics(out["sr_image"].float(), hr.float())
            psnr_val = m["PSNR_mean"]
            val_psnr_list.append(psnr_val)

    mean_val_psnr = float(np.mean(val_psnr_list))
    n = max(1, valid_batches)
    avg_loss = tot_loss / n
    log_msg = (
        f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {avg_loss:.4f} "
        f"[Rec: {tot_rec/n:.4f}, Spec: {tot_spec/n:.4f}, Deg: {tot_degrade/n:.4f}, Struct: {tot_struct/n:.4f}, Conf: {tot_conf/n:.4f}] "
        f"| Val PSNR: {mean_val_psnr:.2f} dB"
    )
    print(log_msg)
    log_file.write(log_msg + "\n")
    log_file.flush()

    if mean_val_psnr > best_psnr:
        best_psnr = mean_val_psnr
        raw_m = model.module if isinstance(model, nn.DataParallel) else model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": raw_m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
        }
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, "bharatsrm_v4_pretrained.pth"))

log_file.close()
print(f"\nPretraining Complete. Best Val PSNR: {best_psnr:.2f} dB")
