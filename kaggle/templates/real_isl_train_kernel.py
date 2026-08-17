import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Dataset class for real ISL dataset
class RealISLDataset(Dataset):
    def __init__(self, data_dir):
        # Ingest real landmark files (76-keypoint arrays)
        # Expected structure: data_dir/signer_id/class_id/sample_id.npy
        self.data_dir = data_dir
        self.samples = []
        self.labels = []
        self.signers = []
        
        if os.path.exists(data_dir):
            for signer_dir in os.listdir(data_dir):
                signer_path = os.path.join(data_dir, signer_dir)
                if not os.path.isdir(signer_path):
                    continue
                signer_id = int(signer_dir.replace('signer_', ''))
                
                for class_dir in os.listdir(signer_path):
                    class_path = os.path.join(signer_path, class_dir)
                    if not os.path.isdir(class_path):
                        continue
                    class_id = int(class_dir.replace('class_', ''))
                    
                    for sample_file in os.listdir(class_path):
                        if sample_file.endswith('.npy'):
                            self.samples.append(os.path.join(class_path, sample_file))
                            self.labels.append(class_id)
                            self.signers.append(signer_id)
        else:
            print(f"Warning: {data_dir} not found. Generating dummy metadata for template testing.")
            self.samples = ["dummy_path.npy"] * 1000
            self.labels = np.random.randint(0, 263, 1000).tolist()
            self.signers = np.random.randint(1, 16, 1000).tolist()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.samples[idx]
        if os.path.exists(path):
            feature = np.load(path)
            feature = torch.tensor(feature, dtype=torch.float32)
        else:
            # Dummy fallback for testing: 76 keypoints * T frames (e.g. 50)
            feature = torch.randn(76, 50) 
            
        label = self.labels[idx]
        signer = self.signers[idx]
        return feature, label, signer

# 2. Model definitions
class Tier1Model(nn.Module):
    def __init__(self, num_classes=263):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(76, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class Tier2Model(nn.Module):
    def __init__(self, num_classes=263):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(76, 256, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    model.train()
    total_loss = 0
    for features, labels, _ in dataloader:
        features, labels = features.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast():
            outputs = model(features)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(dataloader) if len(dataloader) > 0 else 0

def evaluate(model, dataloader, device, num_classes=263):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for features, labels, _ in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    if not all_labels:
        return 0.0, [], []
        
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds, labels=range(num_classes)).tolist()
    
    # Calculate per-class accuracy
    cm_array = np.array(cm)
    per_class_acc = (cm_array.diagonal() / np.maximum(cm_array.sum(axis=1), 1)).tolist()
    
    return acc, cm, per_class_acc

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset_path = '/kaggle/input/real-isl-dataset'
    dataset = RealISLDataset(dataset_path)
    
    # 3. Strict Signer-disjoint splitting
    # Signers 1-10: Train, 11-12: Val, 13-15: Test
    train_idx = [i for i, s in enumerate(dataset.signers) if 1 <= s <= 10]
    val_idx = [i for i, s in enumerate(dataset.signers) if 11 <= s <= 12]
    test_idx = [i for i, s in enumerate(dataset.signers) if 13 <= s <= 15]
    
    train_loader = DataLoader(torch.utils.data.Subset(dataset, train_idx), batch_size=32, shuffle=True)
    val_loader = DataLoader(torch.utils.data.Subset(dataset, val_idx), batch_size=32)
    test_loader = DataLoader(torch.utils.data.Subset(dataset, test_idx), batch_size=32)
    
    num_classes = 263
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    
    epochs = 15
    
    # Train Tier 1
    print("Training Tier 1...")
    tier1_model = Tier1Model(num_classes).to(device)
    t1_opt = optim.Adam(tier1_model.parameters(), lr=1e-3)
    
    best_t1_acc = 0.0
    for epoch in range(epochs):
        train_loss = train_epoch(tier1_model, train_loader, criterion, t1_opt, scaler, device)
        val_acc, _, _ = evaluate(tier1_model, val_loader, device, num_classes)
        print(f"Tier 1 - Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
        if val_acc >= best_t1_acc:
            best_t1_acc = val_acc
            torch.save(tier1_model.state_dict(), 'tier1_real_isl_best.pth')
            
    # Train Tier 2
    print("Training Tier 2...")
    tier2_model = Tier2Model(num_classes).to(device)
    t2_opt = optim.Adam(tier2_model.parameters(), lr=1e-3)
    
    best_t2_acc = 0.0
    for epoch in range(epochs):
        train_loss = train_epoch(tier2_model, train_loader, criterion, t2_opt, scaler, device)
        val_acc, _, _ = evaluate(tier2_model, val_loader, device, num_classes)
        print(f"Tier 2 - Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f} - Val Acc: {val_acc:.4f}")
        if val_acc >= best_t2_acc:
            best_t2_acc = val_acc
            torch.save(tier2_model.state_dict(), 'tier2_real_isl_best.pth')
            
    # Final Evaluation & Metrics Export
    print("Exporting final metrics...")
    if os.path.exists('tier1_real_isl_best.pth'):
        tier1_model.load_state_dict(torch.load('tier1_real_isl_best.pth'))
    t1_test_acc, t1_cm, t1_pca = evaluate(tier1_model, test_loader, device, num_classes)
    
    if os.path.exists('tier2_real_isl_best.pth'):
        tier2_model.load_state_dict(torch.load('tier2_real_isl_best.pth'))
    t2_test_acc, t2_cm, t2_pca = evaluate(tier2_model, test_loader, device, num_classes)
    
    metrics = {
        "tier1": {
            "test_accuracy": t1_test_acc,
            "per_class_accuracy": t1_pca,
            "confusion_matrix": t1_cm
        },
        "tier2": {
            "test_accuracy": t2_test_acc,
            "per_class_accuracy": t2_pca,
            "confusion_matrix": t2_cm
        }
    }
    
    with open('real_isl_metrics.json', 'w') as f:
        json.dump(metrics, f)
    print("Done. Metrics saved to real_isl_metrics.json")

if __name__ == '__main__':
    main()
