"""
=============================================================================
TIER 1: SOTA ISOLATED SIGN SPATIAL-TEMPORAL TRANSFORMER (263 ISL CLASSES)
=============================================================================
Hardware: NVIDIA Tesla T4 (Kaggle Accelerator: GPU T4 x2)
Dataset: swaptr/indian-sign-language-mediapipe-holistic-landmarks
Architecture: Squeezeformer / SE-Conv1D + Transformer Multi-Head Attention Hybrid
Feature Engineering: 328 Dimensions (Coordinates 152 + Velocities 152 + Distances 24)
Regularization: MixUp (alpha=0.2) + DropLandmark (10%) + Affine Augmentation + Label Smoothing
Output: /kaggle/working/tier1_include_best.pth & tier1_metrics.json
=============================================================================
"""

import glob
import json
import math
import os
import random
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

print("=" * 80)
print("=== [TIER 1 SOTA] ISL SPATIAL-TEMPORAL TRANSFORMER (263 CLASSES) ===")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"✅ GPU ACTIVE: {device_name} | VRAM: {vram_gb:.2f} GB")
else:
    device = torch.device("cpu")
    print("⚠️ WARNING: Running on CPU.")

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===========================================================================
# 1. Dataset Discovery & Universal Parquet Parsing
# ===========================================================================
print("\n[Step 1] Locating Dataset & Discovering Canonical Landmarks...")

search_roots = [
    "/kaggle/input",
    "/kaggle/input/datasets",
    "/kaggle/input/datasets/swaptr/indian-sign-language-mediapipe-holistic-landmarks",
    "/root/.cache/kagglehub/datasets",
]

try:
    import kagglehub
    kh_path = kagglehub.dataset_download("swaptr/indian-sign-language-mediapipe-holistic-landmarks")
    if kh_path and os.path.exists(kh_path):
        search_roots.insert(0, kh_path)
        print(f"✅ kagglehub root: {kh_path}")
except Exception as e:
    print(f"kagglehub notice: {e}")

all_parquet_files: List[str] = []
for s_root in search_roots:
    if os.path.exists(s_root):
        for root, _, files in os.walk(s_root):
            for f in files:
                if f.endswith(".parquet"):
                    all_parquet_files.append(os.path.join(root, f))

all_parquet_files = sorted(list(set(all_parquet_files)))
print(f"Discovered {len(all_parquet_files):,} Parquet Sequence Files.")

if not all_parquet_files:
    raise FileNotFoundError("No Parquet files found. Please attach 'swaptr/indian-sign-language-mediapipe-holistic-landmarks'.")

# Discover Canonical Landmarks
def discover_canonical_landmarks(files: List[str], probe_n: int = 30, target: int = 76) -> List[Tuple[str, int]]:
    seen = set()
    for fp in files[:probe_n]:
        try:
            df = pd.read_parquet(fp, columns=["type", "landmark_index"])
            seen.update(zip(df["type"], df["landmark_index"]))
        except Exception:
            continue
    canonical = sorted(list(seen))[:target]
    return canonical

CANONICAL_LANDMARKS = discover_canonical_landmarks(all_parquet_files, probe_n=30, target=76)
print(f"Canonical landmark schema locked to {len(CANONICAL_LANDMARKS)} landmarks.")

# Build Samples Registry
samples_registry: List[Tuple[str, str]] = []
for p_file in all_parquet_files:
    parent_dir = os.path.basename(os.path.dirname(p_file))
    if parent_dir.lower() in ["train_landmarks", "1", "versions", "input", "landmarks", "keypoints"]:
        c_name = os.path.basename(p_file).split("_")[0]
    else:
        c_name = parent_dir
    samples_registry.append((p_file, c_name))

unique_classes = sorted(list(set(s[1] for s in samples_registry)))
num_classes = len(unique_classes)
class_to_idx = {c: i for i, c in enumerate(unique_classes)}
print(f"Total Unique Classes: {num_classes}")

labels_arr = np.array([class_to_idx[s[1]] for s in samples_registry])
idxs = np.arange(len(samples_registry))

train_idx, rest_idx = train_test_split(idxs, test_size=0.30, random_state=SEED, stratify=labels_arr)
val_idx, test_idx = train_test_split(rest_idx, test_size=0.50, random_state=SEED, stratify=labels_arr[rest_idx])

train_samples = [samples_registry[i] for i in train_idx]
val_samples = [samples_registry[i] for i in val_idx]
test_samples = [samples_registry[i] for i in test_idx]

print(f"Stratified Splits -> Train: {len(train_samples):,}, Val: {len(val_samples):,}, Test: {len(test_samples):,}")


# ===========================================================================
# 2. Rich 328-Dim Feature Extractor & Augmentation Dataset
# ===========================================================================
class SOTAISLDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, str]],
        class_to_idx: Dict[str, int],
        canonical_landmarks: List[Tuple[str, int]],
        max_len: int = 150,
        is_train: bool = True,
        cache: bool = True,
    ):
        self.samples = samples
        self.class_to_idx = class_to_idx
        self.canonical_landmarks = canonical_landmarks
        self.max_len = max_len
        self.is_train = is_train
        self.cache_enabled = cache
        self._cache: Dict[int, Tuple[np.ndarray, int]] = {}

        # 24 Key Anatomical Distance Pairs
        self.distance_pairs = [
            (0, 1), (0, 2), (0, 3), (0, 4),  # Nose to face/shoulders
            (10, 20), (11, 21), (12, 22), (13, 23),  # Wrist to fingertips (LH)
            (20, 24), (20, 28), (20, 32), (20, 36),  # Thumb to other fingertips (LH)
            (40, 50), (41, 51), (42, 52), (43, 53),  # Wrist to fingertips (RH)
            (50, 54), (50, 58), (50, 62), (50, 66),  # Thumb to other fingertips (RH)
            (20, 50), (24, 54), (0, 20), (0, 50),   # Inter-hand & hand-to-nose
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def _parse_parquet(self, path: str) -> np.ndarray:
        df = pd.read_parquet(path).dropna(subset=["frame", "x", "y"])
        frames = sorted(df["frame"].unique())
        frame_to_row = {f: i for i, f in enumerate(frames)}
        T = len(frames)
        L = len(self.canonical_landmarks)
        lm_to_idx = {lm: i for i, lm in enumerate(self.canonical_landmarks)}

        seq = np.zeros((max(T, 1), L, 2), dtype=np.float32)
        grouped = df.groupby(["frame", "type", "landmark_index"], as_index=False)[["x", "y"]].mean()

        for row in grouped.itertuples(index=False):
            key = (row.type, row.landmark_index)
            if key in lm_to_idx and row.frame in frame_to_row:
                r, c = frame_to_row[row.frame], lm_to_idx[key]
                seq[r, c, 0] = row.x
                seq[r, c, 1] = row.y

        # Per-frame Torso/Centroid Normalization
        for t in range(seq.shape[0]):
            mask = (seq[t] != 0).any(axis=1)
            if mask.any():
                center = seq[t][mask].mean(axis=0)
                scale = seq[t][mask].std() + 1e-6
                seq[t][mask] = (seq[t][mask] - center) / scale

        return seq  # (T, 76, 2)

    def _augment(self, seq: np.ndarray) -> np.ndarray:
        # seq: (T, 76, 2)
        T, V, C = seq.shape

        # 1. Random Spatial Rotation (-15° to +15°)
        angle = random.uniform(-15, 15) * math.pi / 180.0
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        seq = np.dot(seq, rot_mat.T)

        # 2. Random Scale & Jitter
        scale = random.uniform(0.88, 1.12)
        seq = seq * scale + np.array([random.uniform(-0.04, 0.04), random.uniform(-0.04, 0.04)])

        # 3. DropLandmark (10% masking)
        mask = np.random.rand(V) > 0.10
        seq[:, ~mask, :] = 0.0

        # 4. Temporal Resampling / Speed Warping
        speed = random.uniform(0.85, 1.15)
        new_T = max(10, int(T * speed))
        indices = np.linspace(0, T - 1, new_T).astype(int)
        seq = seq[indices]

        return seq

    def _extract_328d_features(self, seq: np.ndarray) -> np.ndarray:
        # seq: (T, 76, 2)
        T, V, C = seq.shape

        # 1. Base Coordinates (152 dim)
        base_coords = seq.reshape(T, -1)

        # 2. First-Order Velocities (152 dim)
        velocities = np.zeros_like(base_coords)
        velocities[1:] = base_coords[1:] - base_coords[:-1]

        # 3. Key Euclidean Distance Pairs (24 dim)
        distances = np.zeros((T, len(self.distance_pairs)), dtype=np.float32)
        for i, (i1, i2) in enumerate(self.distance_pairs):
            idx1 = min(i1, V - 1)
            idx2 = min(i2, V - 1)
            distances[:, i] = np.linalg.norm(seq[:, idx1, :] - seq[:, idx2, :], axis=-1)

        # Concat: 152 + 152 + 24 = 328 dimensions
        features_328 = np.concatenate([base_coords, velocities, distances], axis=-1)

        # Standardize Length to max_len
        curr_T = features_328.shape[0]
        if curr_T > self.max_len:
            start = (curr_T - self.max_len) // 2
            features_328 = features_328[start : start + self.max_len]
        elif curr_T < self.max_len:
            features_328 = np.vstack([features_328, np.zeros((self.max_len - curr_T, 328), dtype=np.float32)])

        return features_328

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_enabled and idx in self._cache:
            raw_seq, lbl = self._cache[idx]
        else:
            p_path, c_name = self.samples[idx]
            lbl = self.class_to_idx.get(c_name, 0)
            raw_seq = self._parse_parquet(p_path)
            if self.cache_enabled:
                self._cache[idx] = (raw_seq, lbl)

        seq = raw_seq.copy()
        if self.is_train:
            seq = self._augment(seq)

        features = self._extract_328d_features(seq)
        return torch.tensor(features, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


# MixUp Data Augmentation Function
def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion: nn.Module, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ===========================================================================
# 3. SOTA Squeezeformer / SE-Conv1D + Transformer Hybrid Architecture
# ===========================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class SE1DConvBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 4, dropout: float = 0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.GELU(),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        res = x
        out = self.drop(self.act(self.norm(self.conv1(x))))
        out = self.conv2(out)
        w = self.se(out)
        out = out * w
        return out + res


class SOTASignTransformer(nn.Module):
    def __init__(self, in_features: int = 328, num_classes: int = 263, d_model: int = 256, nhead: int = 4, num_layers: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.pos_enc = PositionalEncoding(d_model, max_len=160)

        # 2x SE-Conv1D Blocks (Local temporal motion bursts)
        self.se_block1 = SE1DConvBlock(d_model)
        self.se_block2 = SE1DConvBlock(d_model)

        # 3x Transformer Encoder Layers (Global sign semantics)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=0.15,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-Pooling Head (MeanPool + MaxPool -> 512)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, 512),
            nn.GELU(),
            nn.Dropout(0.35),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=150, 328)
        feat = self.in_proj(x)
        feat = self.pos_enc(feat)

        # Pass through SE-Conv blocks: (B, T, C) -> (B, C, T)
        feat_c = feat.transpose(1, 2)
        feat_c = self.se_block1(feat_c)
        feat_c = self.se_block2(feat_c)
        feat = feat_c.transpose(1, 2)

        # Pass through Transformer
        encoded = self.transformer(feat)  # (B, T, d_model)

        # Multi-Pooling
        mean_pool = encoded.mean(dim=1)
        max_pool, _ = encoded.max(dim=1)
        multi_pooled = torch.cat([mean_pool, max_pool], dim=-1)  # (B, d_model * 2)

        return self.classifier(multi_pooled)


model = SOTASignTransformer(in_features=328, num_classes=num_classes, d_model=256, nhead=4, num_layers=3).to(device)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)

EPOCHS = 35
scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

train_ds = SOTAISLDataset(train_samples, class_to_idx, CANONICAL_LANDMARKS, is_train=True)
val_ds = SOTAISLDataset(val_samples, class_to_idx, CANONICAL_LANDMARKS, is_train=False)
test_ds = SOTAISLDataset(test_samples, class_to_idx, CANONICAL_LANDMARKS, is_train=False)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

print(f"\nModel Initialized: SOTASignTransformer (Parameters: {sum(p.numel() for p in model.parameters()):,})")


# ===========================================================================
# 4. Multi-Epoch Training Loop with MixUp
# ===========================================================================
best_val_acc = 0.0
t_start = time.time()

print(f"\n[Step 2] Training SOTA Model for {EPOCHS} Epochs on {device}...")

for epoch in range(EPOCHS):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        # Apply MixUp in training
        mixed_xb, y_a, y_b, lam = mixup_data(xb, yb, alpha=0.2)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(mixed_xb)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item() * len(yb)
        preds = torch.argmax(out, dim=1)
        correct += (preds == yb).sum().item()
        total += len(yb)

    scheduler.step()

    # Validation Pass
    model.eval()
    v_corr, v_tot = 0, 0
    with torch.no_grad():
        for xv, yv in val_loader:
            xv, yv = xv.to(device), yv.to(device)
            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                vout = model(xv)
            v_corr += (torch.argmax(vout, dim=1) == yv).sum().item()
            v_tot += len(yv)

    val_acc = v_corr / max(v_tot, 1)
    train_acc = correct / max(total, 1)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Loss: {tot_loss/max(total,1):.4f} | "
          f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(OUTPUT_DIR, "tier1_include_best.pth")
        torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx, "num_classes": num_classes}, save_path)
        print(f"  -> Best SOTA checkpoint saved to {save_path} (Val Acc: {best_val_acc*100:.2f}%)")

elapsed = time.time() - t_start
print(f"\n✅ Training Complete in {elapsed/60:.2f} minutes (Best Val Acc: {best_val_acc*100:.2f}%).")


# ===========================================================================
# 5. Held-Out Test Set Evaluation (Top-1 & Top-5 Accuracy)
# ===========================================================================
print("\n[Step 3] Evaluating Best Checkpoint on Held-Out Test Set...")
ckpt = torch.load(os.path.join(OUTPUT_DIR, "tier1_include_best.pth"), map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

top1_correct = 0
top5_correct = 0
total_tested = 0

with torch.no_grad():
    for xt, yt in test_loader:
        xt, yt = xt.to(device), yt.to(device)
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            logits = model(xt)
        
        # Top-1
        top1_preds = torch.argmax(logits, dim=1)
        top1_correct += (top1_preds == yt).sum().item()

        # Top-5
        _, top5_preds = torch.topk(logits, k=min(5, num_classes), dim=1)
        top5_correct += sum([yt[i] in top5_preds[i] for i in range(len(yt))])
        total_tested += len(yt)

final_top1 = top1_correct / max(total_tested, 1)
final_top5 = top5_correct / max(total_tested, 1)

print("=" * 80)
print(f"🏆 FINAL TEST SET EVALUATION RESULTS ({total_tested:,} test samples):")
print(f"   • Top-1 Accuracy: {final_top1*100:.2f}%")
print(f"   • Top-5 Accuracy: {final_top5*100:.2f}%")
print("=" * 80)

# Save Final Metrics
metrics = {
    "tier": 1,
    "model_architecture": "SOTASignTransformer (Squeezeformer/SE-Conv1D + Multi-Head Attention)",
    "classes": num_classes,
    "feature_dimensions": 328,
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "test_samples": len(test_samples),
    "epochs": EPOCHS,
    "best_val_accuracy": float(best_val_acc),
    "top1_test_accuracy": float(final_top1),
    "top5_test_accuracy": float(final_top5),
    "training_time_minutes": round(elapsed / 60, 2),
    "status": "COMPLETE",
}

with open(os.path.join(OUTPUT_DIR, "tier1_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Tier-1 SOTA Metrics saved to {OUTPUT_DIR}/tier1_metrics.json")
print("Execution Finished Successfully.")
