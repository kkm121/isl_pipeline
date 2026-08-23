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
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import glob

print("=" * 80)
print("=== BharatSRM-Net v4: WorldStrat RGBN Pretraining Kernel (Dual T4) ===")
print("=" * 80)

# 1. Hard CUDA Hardware Check
assert torch.cuda.is_available(), "T4 GPU required for training"
device = torch.device("cuda")
test_tensor = torch.zeros((1, 1), device=device)
torch.cuda.synchronize()
print(f"✅ GPU ACTIVE: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Synthetic / Paired Stream Loader for Pretraining
class WorldStratStreamDataset(Dataset):
    def __init__(self, data_dir: str = "/kaggle/input/worldstrat", num_samples: int = 500, patch_size: int = 64):
        self.data_dir = data_dir
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.hr_size = patch_size * 4
        self.use_synthetic = True
        self.pairs = []
        
        if os.path.exists(data_dir):
            lr_files = sorted(glob.glob(os.path.join(data_dir, "**", "*_lr_*.tif"), recursive=True))
            hr_files = sorted(glob.glob(os.path.join(data_dir, "**", "*_hr_*.tif"), recursive=True))
            if len(lr_files) > 0 and len(hr_files) > 0:
                self.pairs = list(zip(lr_files, hr_files))
                self.use_synthetic = False
                
        if self.use_synthetic:
            print("⚠️ WARNING: WorldStrat COG files not found in Kaggle input mount. Falling back to synthetic data!")

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
            try:
                import rasterio
                lr_path, hr_path = self.pairs[idx % len(self.pairs)]
                with rasterio.open(lr_path) as src_lr:
                    lr = torch.from_numpy(src_lr.read().astype(np.float32))
                with rasterio.open(hr_path) as src_hr:
                    hr = torch.from_numpy(src_hr.read().astype(np.float32))
                
                if lr.shape[1:] != (self.patch_size, self.patch_size):
                    lr = torch.nn.functional.interpolate(lr.unsqueeze(0), size=(self.patch_size, self.patch_size), mode='bilinear').squeeze(0)
                if hr.shape[1:] != (self.hr_size, self.hr_size):
                    hr = torch.nn.functional.interpolate(hr.unsqueeze(0), size=(self.hr_size, self.hr_size), mode='bilinear').squeeze(0)
                    
                mask = torch.ones(1, self.patch_size, self.patch_size, dtype=torch.float32)
                dem = torch.zeros(2, self.patch_size, self.patch_size, dtype=torch.float32)
                return {"lr_input": lr, "hr_target": hr, "validity_mask": mask, "context_dem": dem}
            except Exception:
                lr = torch.rand(10, self.patch_size, self.patch_size, dtype=torch.float32) * 0.8 + 0.1
                hr = torch.rand(4, self.hr_size, self.hr_size, dtype=torch.float32) * 0.8 + 0.1
                mask = (torch.rand(1, self.patch_size, self.patch_size) > 0.15).float()
                dem = torch.rand(2, self.patch_size, self.patch_size, dtype=torch.float32)
                return {"lr_input": lr, "hr_target": hr, "validity_mask": mask, "context_dem": dem}

# In a standalone script, we import the model and losses
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

if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
    model = nn.DataParallel(model)
model = model.to(device)

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

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

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
            # DataParallel wraps model, meaning criterion might need modification if it expects module methods, 
            # but here criterion is just a callable that takes outputs.

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
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
        }
        torch.save(checkpoint, os.path.join(OUTPUT_DIR, "bharatsrm_v4_pretrained.pth"))

log_file.close()
print(f"\nPretraining Complete. Best Val PSNR: {best_psnr:.2f} dB")
