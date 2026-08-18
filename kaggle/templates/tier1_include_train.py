"""
=============================================================================
TIER 1: ISOLATED SIGN SPATIAL-TEMPORAL FEATURE EXTRACTOR (263 ISL CLASSES)
=============================================================================
Hardware: NVIDIA Tesla T4 (Kaggle Accelerator: GPU T4 x2)
Dataset: swaptr/indian-sign-language-mediapipe-holistic-landmarks
Auto-Discovery: Universal recursive scanner (handles both CSV & Folder formats)
Output: /kaggle/working/tier1_include_best.pth & tier1_metrics.json
=============================================================================
"""

import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

print("=" * 80)
print("=== [TIER 1] ISL SPATIAL-TEMPORAL FEATURE EXTRACTOR (263 CLASSES) ===")
print("=" * 80)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
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
# 1. Universal Dataset Discovery (Recursive Scanner)
# ===========================================================================
print("\n[Step 1] Scanning for ISL Parquet & Metadata Files...")

search_roots = [
    "/kaggle/input",
    "/kaggle/input/datasets",
    "/kaggle/input/datasets/swaptr/indian-sign-language-mediapipe-holistic-landmarks",
    "/root/.cache/kagglehub/datasets",
]

# If kagglehub was called or needed
try:
    import kagglehub
    kh_path = kagglehub.dataset_download("swaptr/indian-sign-language-mediapipe-holistic-landmarks")
    if kh_path and os.path.exists(kh_path):
        search_roots.insert(0, kh_path)
        print(f"✅ kagglehub root: {kh_path}")
except Exception as e:
    print(f"kagglehub notice: {e}")

label_map_file = None
for s_root in search_roots:
    if os.path.exists(s_root):
        for root, _, files in os.walk(s_root):
            if "label_map.json" in files:
                label_map_file = os.path.join(root, "label_map.json")
                break
        if label_map_file: break

class_to_idx = {}
if label_map_file:
    print(f"✅ Found label_map.json at: {label_map_file}")
    with open(label_map_file, "r") as f:
        class_to_idx = json.load(f)
else:
    print("⚠️ WARNING: label_map.json not found! Classes will be inferred from paths.")

all_parquet_files = []
all_csv_files = []

for s_root in search_roots:
    if os.path.exists(s_root):
        for root, _, files in os.walk(s_root):
            for f in files:
                full_p = os.path.join(root, f)
                if f.endswith(".parquet"):
                    all_parquet_files.append(full_p)
                elif f.endswith(".csv") and ("train" in f.lower() or "meta" in f.lower() or "label" in f.lower()):
                    all_csv_files.append(full_p)

# De-duplicate
all_parquet_files = sorted(list(set(all_parquet_files)))
all_csv_files = sorted(list(set(all_csv_files)))

print(f"Discovered {len(all_parquet_files):,} Parquet Sequence Files.")
print(f"Discovered {len(all_csv_files)} Metadata CSV Files.")

# Build Sample Registry: list of (parquet_path, class_name, signer_id)
samples_registry: List[Tuple[str, str, int]] = []

if all_csv_files:
    # Try parsing CSV metadata
    csv_loaded = False
    for c_file in all_csv_files:
        try:
            df_meta = pd.read_csv(c_file)
            signer_c = next((c for c in df_meta.columns if any(k in c.lower() for k in ["participant", "signer"])), None)
            sign_c = next((c for c in df_meta.columns if any(k in c.lower() for k in ["sign", "label", "gloss"])), None)
            path_c = next((c for c in df_meta.columns if any(k in c.lower() for k in ["path", "file", "video", "sequence"])), None)

            if sign_c and path_c:
                base_d = os.path.dirname(c_file)
                for _, row in df_meta.iterrows():
                    r_p = str(row[path_c])
                    s_name = str(row[sign_c])
                    sg_id = int(row[signer_c]) if (signer_c and str(row[signer_c]).isdigit()) else 0

                    abs_cand = os.path.join(base_d, r_p)
                    if os.path.exists(abs_cand):
                        samples_registry.append((abs_cand, s_name, sg_id))
                if len(samples_registry) > 0:
                    print(f"✅ Loaded {len(samples_registry):,} samples from {os.path.basename(c_file)}")
                    csv_loaded = True
                    break
        except Exception:
            pass

if not samples_registry and all_parquet_files:
    # Universal Folder-based Classification: <class_folder>/<file>.parquet
    print("Indexing Parquet files by parsing paths for true class labels...")
    for idx, p_file in enumerate(all_parquet_files):
        c_name = None
        norm_p = p_file.replace("\\", "/")
        fname = os.path.basename(p_file)
        
        for cls_name in class_to_idx.keys():
            if f"/{cls_name}/" in norm_p or fname.startswith(f"{cls_name}_") or fname == f"{cls_name}.parquet":
                c_name = cls_name
                break
                
        if c_name is None:
            # Fallback
            parent_dir = os.path.basename(os.path.dirname(p_file))
            if parent_dir.lower() in ["train_landmarks", "1", "versions", "input", "landmarks"]:
                c_name = os.path.basename(p_file).split("_")[0]
            else:
                c_name = parent_dir
        
        # Estimate signer ID from filename if present (e.g. signer3_hello_01.parquet -> 3)
        signer_id = 0
        if "signer" in fname.lower():
            try:
                part = fname.lower().split("signer")[1]
                digits = "".join([ch for ch in part[:3] if ch.isdigit()])
                if digits:
                    signer_id = int(digits)
            except Exception:
                signer_id = idx % 15
        else:
            signer_id = idx % 15

        samples_registry.append((p_file, c_name, signer_id))

if not samples_registry:
    raise FileNotFoundError(
        "No ISL parquet sequence files found after scanning /kaggle/input and kagglehub. "
        "Please ensure 'swaptr/indian-sign-language-mediapipe-holistic-landmarks' is attached."
    )

# Build Vocabulary & Signer Disjoint Splits
if not class_to_idx:
    unique_classes = sorted(list(set(s[1] for s in samples_registry)))
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
num_classes = len(class_to_idx)

unique_signers = sorted(list(set(s[2] for s in samples_registry)))
n_s = len(unique_signers)

if n_s >= 3:
    n_tr = max(1, int(n_s * 0.70))
    n_va = max(1, int(n_s * 0.15))
    train_signers = set(unique_signers[:n_tr])
    val_signers = set(unique_signers[n_tr : n_tr + n_va])
    test_signers = set(unique_signers[n_tr + n_va :])

    train_samples = [s for s in samples_registry if s[2] in train_signers]
    val_samples = [s for s in samples_registry if s[2] in val_signers]
    test_samples = [s for s in samples_registry if s[2] in test_signers]
else:
    # Index-based split
    np.random.seed(42)
    shuffled = np.random.permutation(len(samples_registry)).tolist()
    n_tr = int(0.70 * len(samples_registry))
    n_va = int(0.15 * len(samples_registry))
    train_samples = [samples_registry[i] for i in shuffled[:n_tr]]
    val_samples = [samples_registry[i] for i in shuffled[n_tr : n_tr + n_va]]
    test_samples = [samples_registry[i] for i in shuffled[n_tr + n_va :]]

print(f"Total Unique Classes: {num_classes}")
print(f"Disjoint Dataset Splits -> Train: {len(train_samples):,}, Val: {len(val_samples):,}, Test: {len(test_samples):,}")


# ===========================================================================
# 2. Sequence Dataset Loader
# ===========================================================================
class ISLSequenceParquetDataset(Dataset):
    def __init__(self, sample_list: List[Tuple[str, str, int]], class_to_idx: Dict[str, int], max_len: int = 150):
        self.samples = sample_list
        self.class_to_idx = class_to_idx
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        p_path, c_name, _ = self.samples[idx]
        lbl = self.class_to_idx.get(c_name, 0)

        df = pd.read_parquet(p_path)
        
        # 152 dimensions (76 keypoints * 2)
        face_idxs = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267]
        pose_idxs = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        lh_idxs = list(range(21))
        rh_idxs = list(range(21))
        
        req_types = (['face'] * 23) + (['pose'] * 11) + (['left_hand'] * 21) + (['right_hand'] * 21)
        req_idxs = face_idxs + pose_idxs + lh_idxs + rh_idxs

        if "x" in df.columns and "y" in df.columns and "type" in df.columns:
            try:
                pivoted = df.pivot(index="frame", columns=["type", "landmark_index"], values=["x", "y"])
                frames = len(pivoted)
                seq_x = np.zeros((frames, 76), dtype=np.float32)
                seq_y = np.zeros((frames, 76), dtype=np.float32)
                
                for i, (t, l_idx) in enumerate(zip(req_types, req_idxs)):
                    if ('x', t, l_idx) in pivoted.columns:
                        seq_x[:, i] = pivoted[('x', t, l_idx)].values
                    if ('y', t, l_idx) in pivoted.columns:
                        seq_y[:, i] = pivoted[('y', t, l_idx)].values
                
                seq = np.hstack([seq_x, seq_y])
            except Exception:
                x_v = df["x"].fillna(0.0).values.reshape(-1, 1)
                y_v = df["y"].fillna(0.0).values.reshape(-1, 1)
                seq = np.hstack([x_v, y_v]).astype(np.float32)
        else:
            x_cols = sorted([c for c in df.columns if c.startswith("x")])[:76]
            y_cols = sorted([c for c in df.columns if c.startswith("y")])[:76]
            seq = df[x_cols + y_cols].fillna(0.0).values.astype(np.float32)

        T, F = seq.shape
        target_F = 152

        if F < target_F:
            seq = np.hstack([seq, np.zeros((T, target_F - F), dtype=np.float32)])
        elif F > target_F:
            seq = seq[:, :target_F]
            
        # Normalize landmarks relative to mid-shoulder distance
        # Indices in our 76-keypoint list:
        # face (0-22), pose (23-33), left (34-54), right (55-75)
        # Shoulders: pose 11 -> index 23, pose 12 -> index 24
        l_shoulder_idx = 23
        r_shoulder_idx = 24
        
        x_l, y_l = seq[:, l_shoulder_idx], seq[:, 76 + l_shoulder_idx]
        x_r, y_r = seq[:, r_shoulder_idx], seq[:, 76 + r_shoulder_idx]
        
        mid_x = (x_l + x_r) / 2.0
        mid_y = (y_l + y_r) / 2.0
        
        dist = np.sqrt((x_l - x_r)**2 + (y_l - y_r)**2)
        dist = np.where(dist < 1e-5, 1.0, dist)
        
        for i in range(76):
            seq[:, i] = (seq[:, i] - mid_x) / dist
            seq[:, 76 + i] = (seq[:, 76 + i] - mid_y) / dist

        if T > self.max_len:
            start = (T - self.max_len) // 2
            seq = seq[start : start + self.max_len]
        elif T < self.max_len:
            seq = np.vstack([seq, np.zeros((self.max_len - T, target_F), dtype=np.float32)])

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(lbl, dtype=torch.long)


# ===========================================================================
# 3. Model Architecture
# ===========================================================================
class ResBlock1D(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = self.bn2(self.conv2(x))
        return self.relu(x + res)

class Tier1TemporalCNN(nn.Module):
    def __init__(self, in_features: int = 152, num_classes: int = 263, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_features, 256, 3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )
        self.layer1 = ResBlock1D(256, dropout)
        self.layer2 = ResBlock1D(256, dropout)
        self.layer3 = ResBlock1D(256, dropout)
        self.layer4 = ResBlock1D(256, dropout)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T=150, F=152) -> transpose to (B, 152, T)
        x = x.transpose(1, 2)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


model = Tier1TemporalCNN(152, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

train_loader = DataLoader(ISLSequenceParquetDataset(train_samples, class_to_idx), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(ISLSequenceParquetDataset(val_samples, class_to_idx), batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(ISLSequenceParquetDataset(test_samples, class_to_idx), batch_size=32, shuffle=False, num_workers=2)

# ===========================================================================
# 4. Multi-Epoch Training Loop
# ===========================================================================
EPOCHS = 20
best_val_acc = 0.0
t_start = time.time()

print(f"\n[Step 2] Training Tier-1 for {EPOCHS} Epochs on {device}...")
for epoch in range(EPOCHS):
    model.train()
    tot_loss, correct, total = 0.0, 0, 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
            out = model(xb)
            loss = criterion(out, yb)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tot_loss += loss.item() * len(yb)
        correct += (torch.argmax(out, dim=1) == yb).sum().item()
        total += len(yb)

    scheduler.step()

    # Validation
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
    print(f"Epoch [{epoch+1:02d}/{EPOCHS:02d}] | Loss: {tot_loss/max(total,1):.4f} | Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%")

    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        save_path = os.path.join(OUTPUT_DIR, "tier1_include_best.pth")
        torch.save({"model_state": model.state_dict(), "class_to_idx": class_to_idx, "num_classes": num_classes}, save_path)
        print(f"  -> Best Tier-1 checkpoint saved to {save_path}")

elapsed = time.time() - t_start
print(f"\n✅ Tier-1 Training Complete in {elapsed/60:.2f} minutes (Best Val Acc: {best_val_acc*100:.2f}%).")

# Save Metrics
metrics = {
    "tier": 1,
    "model_architecture": "Tier1TemporalCNN",
    "classes": num_classes,
    "train_samples": len(train_samples),
    "val_samples": len(val_samples),
    "test_samples": len(test_samples),
    "epochs": EPOCHS,
    "best_val_accuracy": float(best_val_acc),
    "training_time_minutes": round(elapsed / 60, 2),
    "status": "COMPLETE",
}

with open(os.path.join(OUTPUT_DIR, "tier1_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"Tier-1 Metrics saved to {OUTPUT_DIR}/tier1_metrics.json")
print("Execution Complete.")
