import os
import glob
import json
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
import pyarrow.parquet as pq
import time

# ==========================================
# SECTION 1 - Setup & GPU Verification
# ==========================================
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}')
assert torch.cuda.is_available(), 'T4 GPU required'

# ==========================================
# SECTION 2 - Data Loading
# ==========================================
DATA_DIR = '/kaggle/input/indian-sign-language-mediapipe-holistic-landmarks/'
if not os.path.exists(DATA_DIR):
    raise FileNotFoundError('Dataset not mounted. Add swaptr/indian-sign-language-mediapipe-holistic-landmarks as dataset source.')

parquet_files = glob.glob(os.path.join(DATA_DIR, '**', '*.parquet'), recursive=True)
if not parquet_files:
    raise FileNotFoundError(f'No parquet files found in {DATA_DIR}')

# Inspect actual column names
sample_df = pd.read_parquet(parquet_files[0])
print("Sample parquet columns discovered:", sample_df.columns.tolist())

csv_path = os.path.join(DATA_DIR, 'train.csv')
if not os.path.exists(csv_path):
    raise FileNotFoundError('train.csv metadata file not found.')

metadata = pd.read_csv(csv_path)
print("Metadata columns:", metadata.columns.tolist())

# Apply signer-disjoint split
train_signers = set(range(1, 11))
val_signers = set([11, 12])
test_signers = set([13, 14, 15])

train_meta = metadata[metadata['signer_id'].isin(train_signers)]
val_meta = metadata[metadata['signer_id'].isin(val_signers)]
test_meta = metadata[metadata['signer_id'].isin(test_signers)]

classes = sorted(metadata['sign'].unique())
num_classes = len(classes)
class_to_idx = {c: i for i, c in enumerate(classes)}

print(f"Total videos: {len(metadata)}")
print(f"Num classes: {num_classes}")
print(f"Class distribution top 5:\n{metadata['sign'].value_counts().head()}")

class ISLSequenceDataset(Dataset):
    def __init__(self, meta_df, data_dir, class_to_idx, max_len=150):
        self.meta_df = meta_df.reset_index(drop=True)
        self.data_dir = data_dir
        self.class_to_idx = class_to_idx
        self.max_len = max_len
        
        self.sequences = []
        self.labels = []
        
        print(f"Loading {len(self.meta_df)} sequences...")
        for idx, row in self.meta_df.iterrows():
            vid = row['video_id']
            label = self.class_to_idx[row['sign']]
            # Assuming files are named <video_id>.parquet or located in class folders
            # Try direct file first
            vid_path = os.path.join(self.data_dir, f"{vid}.parquet")
            if not os.path.exists(vid_path):
                # Search
                search = glob.glob(os.path.join(self.data_dir, '**', f"{vid}.parquet"), recursive=True)
                if not search:
                    continue
                vid_path = search[0]
                
            df = pd.read_parquet(vid_path)
            # Find 2D landmark coordinates based on actual columns (x and y)
            # ISL mediapipe dataset usually has frame, type, landmark_index, x, y, z
            if 'x' in df.columns and 'y' in df.columns:
                # Pivot frame to get temporal sequence of flat (x,y)
                df_xy = df.pivot(index='frame', columns=['type', 'landmark_index'], values=['x', 'y']).fillna(0)
                seq = df_xy.values
            else:
                # Fallback to direct x_ / y_ columns if flattened
                x_cols = sorted([c for c in df.columns if c.startswith('x_')])
                y_cols = sorted([c for c in df.columns if c.startswith('y_')])
                seq = df[x_cols + y_cols].fillna(0).values
                
            # Truncate / Pad
            T, F = seq.shape
            if T > self.max_len:
                seq = seq[:self.max_len]
            elif T < self.max_len:
                pad = np.zeros((self.max_len - T, F))
                seq = np.vstack([seq, pad])
                
            self.sequences.append(seq)
            self.labels.append(label)
            
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        if len(self.sequences) == 0:
            raise RuntimeError("No sequences could be loaded. Check data paths and columns.")
            
        self.in_features = self.sequences.shape[2]
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])

train_ds = ISLSequenceDataset(train_meta, DATA_DIR, class_to_idx)
val_ds = ISLSequenceDataset(val_meta, DATA_DIR, class_to_idx)
test_ds = ISLSequenceDataset(test_meta, DATA_DIR, class_to_idx)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

# ==========================================
# SECTION 3 - Model
# ==========================================
class Tier1TemporalCNN(nn.Module):
    def __init__(self, in_features, num_classes):
        super(Tier1TemporalCNN, self).__init__()
        # Input: (batch, T, in_features) -> Conv1d needs (batch, in_features, T)
        self.conv1 = nn.Conv1d(in_features, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.pool2 = nn.MaxPool1d(2)
        
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(512)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (B, T, F) -> permute to (B, F, T)
        x = x.permute(0, 2, 1)
        
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.relu(self.bn3(self.conv3(x)))
        
        x = self.global_pool(x).squeeze(-1)
        x = self.fc(x)
        return x

in_features = train_ds.in_features
model = Tier1TemporalCNN(in_features, num_classes).cuda()

# ==========================================
# SECTION 4 - Training
# ==========================================
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler('cuda')

epochs = 15
best_val_acc = 0.0
best_model_path = '/kaggle/working/tier1_include_best.pth'

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = outputs.max(1)
        train_correct += preds.eq(targets).sum().item()
        train_total += inputs.size(0)
        
    scheduler.step()
    
    train_acc = train_correct / train_total
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            _, preds = outputs.max(1)
            val_correct += preds.eq(targets).sum().item()
            val_total += inputs.size(0)
            
    val_acc = val_correct / val_total
    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/train_total:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
    
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)
        print(f" -> Saved new best model (Val Acc: {best_val_acc:.4f})")

# ==========================================
# SECTION 5 - Honest Evaluation
# ==========================================
print("\n--- Testing on Held-out Signers ---")
model.load_state_dict(torch.load(best_model_path))
model.eval()

test_correct = 0
test_total = 0
top5_correct = 0
all_targets = []
all_preds = []

with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.cuda(), targets.cuda()
        outputs = model(inputs)
        
        # Top 1
        _, preds = outputs.max(1)
        test_correct += preds.eq(targets).sum().item()
        
        # Top 5
        _, top5_preds = outputs.topk(5, dim=1)
        top5_correct += torch.any(top5_preds == targets.unsqueeze(1), dim=1).sum().item()
        
        test_total += inputs.size(0)
        
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

top1_acc = test_correct / test_total
top5_acc = top5_correct / test_total

print(f"Test Top-1 Accuracy: {top1_acc:.4f}")
print(f"Test Top-5 Accuracy: {top5_acc:.4f}")

cm = confusion_matrix(all_targets, all_preds)
print("Confusion Matrix:")
print(cm)

# ==========================================
# SECTION 6 - Save Real Metrics
# ==========================================
results = {
    'dataset': 'swaptr/indian-sign-language-mediapipe-holistic-landmarks',
    'gpu': torch.cuda.get_device_name(0),
    'num_classes': num_classes,
    'num_train_videos': len(train_meta),
    'num_test_videos': len(test_meta),
    'top1_test_accuracy': float(top1_acc),
    'top5_test_accuracy': float(top5_acc),
    'training_epochs': epochs,
    'zero_synthetic_data': True,
    'signer_disjoint_split': True
}

# Ensure working directory exists (useful if testing locally, but Kaggle has it)
os.makedirs('/kaggle/working', exist_ok=True)
with open('/kaggle/working/real_include_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print('Saved to /kaggle/working/real_include_results.json')
