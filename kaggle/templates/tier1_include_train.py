"""
=============================================================================
TIER 1: ISOLATED SIGN SPATIAL-TEMPORAL FEATURE EXTRACTOR (263 ISL CLASSES)
=============================================================================
Hardware: NVIDIA T4 / Dual T4 (Kaggle Accelerator: GPU T4 x2)
Dataset: swaptr/indian-sign-language-mediapipe-holistic-landmarks (Kaggle Input)
Estimated Duration: ~25 to 45 minutes (15-25 Epochs)
Output: /kaggle/working/tier1_include_best.pth & tier1_metrics.json
=============================================================================
"""

import glob
import json
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset

print("=== [TIER 1] STARTING ISL FEATURE EXTRACTOR TRAINING ===")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Warning: CUDA not detected, using CPU.")

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Ingest Dataset
DATA_DIR = "/kaggle/input/indian-sign-language-mediapipe-holistic-landmarks"
if not os.path.exists(DATA_DIR):
    candidates = glob.glob("/kaggle/input/**/train.csv", recursive=True)
    if candidates:
        DATA_DIR = os.path.dirname(candidates[0])
    else:
        raise FileNotFoundError(f"Dataset not found at {DATA_DIR}. Please attach 'swaptr/indian-sign-language-mediapipe-holistic-landmarks'.")

csv_path = os.path.join(DATA_DIR, "train.csv")
metadata = pd.read_csv(csv_path)
print(f"Loaded train.csv ({len(metadata)} samples).")

signer_col = next((c for c in metadata.columns if any(k in c.lower() for k in ["participant", "signer"])), metadata.columns[1])
sign_col = next((c for c in metadata.columns if any(k in c.lower() for k in ["sign", "label", "gloss"])), metadata.columns[-1])
path_col = next((c for c in metadata.columns if any(k in c.lower() for k in ["path", "file", "video", "sequence"])), metadata.columns[0])

unique_signers = sorted(metadata[signer_col].unique())
n_s = len(unique_signers)
train_signers = set(unique_signers[: max(1, int(n_s * 0.70))])
val_signers = set(unique_signers[max(1, int(n_s * 0.70)) : max(1, int(n_s * 0.85))])
test_signers = set(unique_signers[max(1, int(n_s * 0.85)) :])

train_meta = metadata[metadata[signer_col].isin(train_signers)]
val_meta = metadata[metadata[signer_col].isin(val_signers)]
test_meta = metadata[metadata[signer_col].isin(test_signers)]

classes = sorted(metadata[sign_col].unique())
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}
print(f"Classes: {num_classes} | Signer Splits: Train={len(train_meta)}, Val={len(val_meta)}, Test={len(test_meta)}")


class GISLRDataset(Dataset):
    def __init__(self, meta_df, data_dir, class_to_idx, max_len=150):
        self.samples = []
        self.max_len = max_len
        self.class_to_idx = class_to_idx

        for _, row in meta_df.iterrows():
            rel_p = str(row[path_col])
            lbl = self.class_to_idx.get(str(row[sign_col]), 0)
            abs_p = os.path.join(data_dir, rel_p)
            if not os.path.exists(abs_p):
                matches = glob.glob(os.path.join(data_dir, "**", os.path.basename(rel_p)), recursive=True)
                if matches:
                    abs_p = matches[0]
                else:
                    continue
            self.samples.append((abs_p, lbl))
        print(f"Loaded {len(self.samples)} valid sequence parquet files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        df = pd.read_parquet(path)
        if "x" in df.columns and "y" in df.columns and "type" in df.columns:
            try:
                pivoted = df.pivot(index="frame", columns=["type", "landmark_index"], values=["x", "y"]).fillna(0.0)
                seq = pivoted.values.astype(np.float32)
            except Exception:
                x_v = df["x"].fillna(0.0).values.reshape(-1, 1)
                y_v = df["y"].fillna(0.0).values.reshape(-1, 1)
                seq = np.hstack([x_v, y_v]).astype(np.float32)
        else:
            x_cols = sorted([c for c in df.columns if c.startswith("x")])[:76]
            y_cols = sorted([c for c in df.columns if c.startswith("y")])[:76]
            seq = df[x_cols + y_cols].fillna(0.0).values.astype(np.float32)

        T, F = seq.shape
        if F < 152:
            seq = np.hstack([seq, np.zeros((T, 152 - F), dtype=np.float32)])
        elif F > 152:
            seq = seq[:, :152]

        if T > self.max_len:
            start = (T - self.max_len) // 2
            seq = seq[start : start + self.max_len]
        elif T < self.max_len:
            seq = np.vstack([seq, np.zeros((self.max_len - T, 152), dtype=np.float32)])

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class Tier1TemporalCNN(nn.Module):
    def __init__(self, in_features=152, num_classes=263, dropout=0.25):
        super().__init__()
        self.conv1 = nn.Conv1d(in_features, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(128, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.conv3 = nn.Conv1d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        return self.fc(x)


model = Tier1TemporalCNN(152, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

train_loader = DataLoader(GISLRDataset(train_meta, DATA_DIR, class_to_idx), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(GISLRDataset(val_meta, DATA_DIR, class_to_idx), batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(GISLRDataset(test_meta, DATA_DIR, class_to_idx), batch_size=32, shuffle=False, num_workers=2)

EPOCHS = 20
best_val_acc = 0.0
t_start = time.time()

print(f"\nTraining Tier-1 for {EPOCHS} Epochs on T4 GPU...")
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
        print(f"  -> Best Tier-1 model saved to {save_path}")

elapsed = time.time() - t_start
print(f"\nTier-1 Training Complete in {elapsed/60:.2f} minutes (Best Val Acc: {best_val_acc*100:.2f}%).")

# Save Tier-1 metrics
metrics = {
    "tier": 1,
    "model_architecture": "Tier1TemporalCNN",
    "classes": num_classes,
    "epochs": EPOCHS,
    "best_val_accuracy": float(best_val_acc),
    "training_time_minutes": round(elapsed / 60, 2),
    "status": "COMPLETE",
}
with open(os.path.join(OUTPUT_DIR, "tier1_metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Tier-1 Metrics saved to {OUTPUT_DIR}/tier1_metrics.json")
