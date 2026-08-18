# DEPRECATED: This kernel relies on SyntheticISLDataset which is no longer supported.
# Use real_isl_train_kernel.py instead.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
from torch.utils.data import Dataset, DataLoader

# 1. Define Tier1TemporalCNN
class Tier1TemporalCNN(nn.Module):
    def __init__(self, num_classes=200):
        super(Tier1TemporalCNN, self).__init__()
        # A simple temporal CNN for sequence classification
        self.conv1 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x shape: (B, C, T)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x

# 2. Synthetic Data Generation
class SyntheticISLDataset(Dataset):
    def __init__(self, num_classes=200, num_signers=10, samples_per_signer=2):
        self.data = []
        self.labels = []
        self.signers = []
        # Generate data: 200 classes * 10 signers * 2 samples = 4000 total samples
        for c in range(num_classes):
            for s in range(num_signers):
                for _ in range(samples_per_signer):
                    # Generate random feature sequence (e.g., 128 dim, 50 frames)
                    feature = torch.randn(128, 50)
                    self.data.append(feature)
                    self.labels.append(c)
                    self.signers.append(s)
                    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.signers[idx]

def resolve_safe_device() -> torch.device:
    if torch.cuda.is_available():
        try:
            cap = torch.cuda.get_device_capability()
            if cap[0] >= 7:
                return torch.device("cuda")
            else:
                print(f"CUDA device compute capability {cap} is sm_{cap[0]}{cap[1]} (< sm_70). Falling back to multi-core CPU.")
                return torch.device("cpu")
        except Exception as e:
            print(f"CUDA check failed: {e}. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


def main():
    device = resolve_safe_device()
    print(f"Using device: {device}")
    
    # Initialize dataset
    dataset = SyntheticISLDataset()
    print(f"Generated {len(dataset)} synthetic samples.")
    
    # 3. Signer-disjoint split
    # Train: signers 0-6 (7 signers)
    # Val: signers 7-8 (2 signers)
    # Test: signer 9 (1 signer)
    train_indices = [i for i, s in enumerate(dataset.signers) if s < 7]
    val_indices = [i for i, s in enumerate(dataset.signers) if s >= 7 and s < 9]
    test_indices = [i for i, s in enumerate(dataset.signers) if s >= 9]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # 4. Initialize model, optimizer, and scaler for Mixed Precision (FP16)
    model = Tier1TemporalCNN(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    epochs = 15
    best_val_acc = 0.0
    history = []
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        # 5. Training with Mixed Precision (FP16)
        for batch_data, batch_labels, _ in train_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(batch_data)
                loss = criterion(outputs, batch_labels)
            
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_data, batch_labels, _ in val_loader:
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                outputs = model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
        val_acc = correct / total if total > 0 else 0.0
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_acc": val_acc
        }
        history.append(metrics)
        
        # 6. Save best checkpoint
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "tier1_best.pth")
            print(f"-> Saved new best model (Val Acc: {best_val_acc:.4f})")
            
    # 7. Final test evaluation
    print("Evaluating on test set (unseen signer)...")
    if os.path.exists("tier1_best.pth"):
        model.load_state_dict(torch.load("tier1_best.pth"))
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_data, batch_labels, _ in test_loader:
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    test_acc = correct / total if total > 0 else 0.0
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # 8. Output JSON metrics summary
    summary = {
        "best_val_acc": best_val_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "history": history
    }
    
    with open("metrics_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    print("Saved metrics_summary.json")

if __name__ == "__main__":
    main()
