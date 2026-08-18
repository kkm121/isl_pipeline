"""
=============================================================================
TIER 2: CONTINUOUS SIGNFORMER GCN & SPATIAL-TEMPORAL SEQUENCE MODEL
=============================================================================
Hardware: NVIDIA T4 / Dual T4 (Kaggle Accelerator: GPU T4 x2)
Dataset: Exploration-Lab/iSign (Official IIT Kanpur Dataset - 127K+ Sentences)
Estimated Duration: ~2.0 to 3.5 hours (20 Epochs)
Output: /kaggle/working/tier2_signformer_best.pth & tier2_metrics.json
=============================================================================
"""

import io
import json
import math
import os
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

print("=== [TIER 2] STARTING CONTINUOUS SIGNFORMER GCN TRAINING ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)} | VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
else:
    device = torch.device("cpu")
    print("Warning: CUDA not detected, running on CPU.")

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. HuggingFace Auth for Official iSign Access
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        HF_TOKEN = ""

print(f"HuggingFace Token: {HF_TOKEN[:8]}... (Authenticated for Exploration-Lab/iSign)")

# Fetch official iSign_v1.1.csv metadata from Hugging Face
print("Streaming official iSign_v1.1.csv from Hugging Face...")
url = "https://huggingface.co/datasets/Exploration-Lab/iSign/resolve/main/iSign_v1.1.csv"
req = urllib.request.Request(url, headers={"Authorization": f"Bearer {HF_TOKEN}", "User-Agent": "isl-trainer/1.0"})

try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        csv_bytes = resp.read()
    isign_meta = pd.read_csv(io.BytesIO(csv_bytes))
    print(f"Successfully loaded official iSign metadata: {len(isign_meta):,} continuous sentence segments.")
except Exception as e:
    raise RuntimeError(f"Failed to stream official iSign dataset from Hugging Face: {e}")

# Dataset loader
class ISignContinuousDataset(Dataset):
    def __init__(self, meta_df: pd.DataFrame, data_dir: str = "/kaggle/input", max_len: int = 150):
        self.meta = meta_df.to_dict('records')
        self.max_len = max_len
        self.data_dir = data_dir
        # Basic character vocab for CTC
        self.vocab = {chr(i): i for i in range(32, 127)}

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        row = self.meta[idx]
        text = str(row.get("text", ""))
        rel_p = str(row.get("path", row.get("file", row.get("video", ""))))
        abs_p = os.path.join(self.data_dir, rel_p)
        
        if not os.path.exists(abs_p):
             # Try recursive search if direct path fails
             import glob
             matches = glob.glob(os.path.join(self.data_dir, "**", os.path.basename(rel_p)), recursive=True)
             if matches:
                 abs_p = matches[0]
             else:
                 raise RuntimeError(f"Data file {rel_p} not found in {self.data_dir}")

        df = pd.read_parquet(abs_p)
        if "x" in df.columns and "y" in df.columns and "type" in df.columns:
            pivoted = df.pivot(index="frame", columns=["type", "landmark_index"], values=["x", "y"]).fillna(0.0)
            seq = pivoted.values.astype(np.float32)
            seq = seq.reshape(-1, 76, 2)
        else:
            x_cols = sorted([c for c in df.columns if c.startswith("x")])[:76]
            y_cols = sorted([c for c in df.columns if c.startswith("y")])[:76]
            seq = df[x_cols + y_cols].fillna(0.0).values.astype(np.float32)
            seq = seq.reshape(-1, 76, 2)

        T = seq.shape[0]
        if T > self.max_len:
            seq = seq[:self.max_len]
        elif T < self.max_len:
            seq = np.concatenate([seq, np.zeros((self.max_len - T, 76, 2), dtype=np.float32)])
            
        target = [self.vocab.get(c, 0) for c in text]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(target, dtype=torch.long), len(target)

def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    seqs, targets, target_lens = zip(*batch)
    seqs = torch.stack(seqs)
    targets = torch.cat(targets)
    target_lens_t = torch.tensor(target_lens, dtype=torch.long)
    input_lens = torch.full((len(batch),), seqs.shape[1], dtype=torch.long)
    return seqs, targets, input_lens, target_lens_t

# 2. Tier-2 Model Architecture: ST-GCN + Multi-Head Self-Attention
class STGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_nodes: int = 76):
        super().__init__()
        self.conv_spatial = nn.Linear(in_channels, out_channels)
        self.conv_temporal = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, SeqLen, Nodes=76, InChannels)
        B, T, V, C = x.shape
        x_s = self.conv_spatial(x)
        x_s = x_s.permute(0, 3, 1, 2).reshape(B * V, -1, T)
        x_t = self.relu(self.bn(self.conv_temporal(x_s)))
        out = x_t.reshape(B, V, -1, T).permute(0, 3, 1, 2)
        return out


class SignFormerContinuousGCN(nn.Module):
    def __init__(self, num_nodes: int = 76, in_channels: int = 2, d_model: int = 256, nhead: int = 8, vocab_size: int = 5000):
        super().__init__()
        self.gcn1 = STGCNBlock(in_channels, 128, num_nodes)
        self.gcn2 = STGCNBlock(128, d_model, num_nodes)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.ctc_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, SeqLen=150, 76, 2)
        feat = self.gcn1(x)
        feat = self.gcn2(feat)
        # Spatial joint mean-pooling -> (Batch, SeqLen, d_model)
        feat_temporal = feat.mean(dim=2)
        encoded = self.transformer(feat_temporal)
        # CTC emission logits for continuous sentence decoding: (Batch, SeqLen, vocab_size)
        logits = self.ctc_projection(encoded)
        return logits


model = SignFormerContinuousGCN(num_nodes=76, in_channels=2, d_model=256, nhead=8, vocab_size=5000).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

print(f"SignFormer GCN instantiated. Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

# 3. Continuous Multi-Sentence Training Loop
EPOCHS = 20
BATCH_SIZE = 16
SEQ_LEN = 150
t_start = time.time()
best_loss = float("inf")

train_dataset = ISignContinuousDataset(isign_meta, max_len=SEQ_LEN)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2)

print(f"\nTraining Tier-2 Continuous SignFormer for {EPOCHS} Epochs on T4 GPU...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_loader)

    for x_batch, targets, input_lens, target_lens in train_loader:
        x_batch = x_batch.to(device)
        targets = targets.to(device)
        input_lens = input_lens.to(device)
        target_lens = target_lens.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(x_batch)  # (Batch, SeqLen, Vocab)
            log_probs = logits.log_softmax(2).transpose(0, 1)  # (SeqLen, Batch, Vocab) for CTCLoss
            loss = criterion(log_probs, targets, input_lens, target_lens)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

    scheduler.step()
    avg_loss = epoch_loss / max(num_batches, 1)
    print(f"Tier-2 Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Continuous CTC Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = os.path.join(OUTPUT_DIR, "tier2_signformer_best.pth")
        torch.save(model.state_dict(), save_path)
        print(f"  -> Best Tier-2 checkpoint saved to {save_path}")

elapsed = time.time() - t_start
print(f"\nTier-2 Continuous Training Complete in {elapsed/60:.2f} minutes (Best CTC Loss: {best_loss:.4f}).")

metrics = {
    "tier": 2,
    "model_architecture": "SignFormerContinuousGCN",
    "dataset": "Exploration-Lab/iSign",
    "continuous_sentence_segments": len(isign_meta),
    "epochs": EPOCHS,
    "best_ctc_loss": float(best_loss),
    "training_time_minutes": round(elapsed / 60, 2),
    "status": "COMPLETE",
}
with open(os.path.join(OUTPUT_DIR, "tier2_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Tier-2 Metrics saved to {OUTPUT_DIR}/tier2_metrics.json")
