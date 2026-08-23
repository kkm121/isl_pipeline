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
    def __init__(self, num_samples: int = 500, patch_size: int = 64):
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.hr_size = patch_size * 4

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        # 10-band LR Sentinel-2 input: (10, 64, 64)
        lr = torch.rand(10, self.patch_size, self.patch_size, dtype=torch.float32) * 0.8 + 0.1
        # 4-band HR SPOT target: (4, 256, 256)
        hr = torch.rand(4, self.hr_size, self.hr_size, dtype=torch.float32) * 0.8 + 0.1
        # Validity mask: (1, 64, 64)
        mask = (torch.rand(1, self.patch_size, self.patch_size) > 0.15).float()
        # DEM context: (2, 64, 64)
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
).to(device)

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

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

print(f"\nModel initialized: {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"Starting WorldStrat Pretraining for {EPOCHS} Epochs...")

best_psnr = 0.0
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
    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Loss: {tot_loss/len(train_dataset):.4f} | Val PSNR: {mean_val_psnr:.2f} dB")

    if mean_val_psnr > best_psnr:
        best_psnr = mean_val_psnr
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "bharatsrm_v4_pretrained.pth"))

print(f"\nPretraining Complete. Best Val PSNR: {best_psnr:.2f} dB")
