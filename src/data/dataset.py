import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any
import json
import logging

from src.data.preprocessing import normalize_landmarks, pad_sequence, flatten_landmarks, augment_landmarks
from src.models.config import DataConfig

logger = logging.getLogger(__name__)

class ISLDataset(Dataset):
    def __init__(self, sequences: Any, labels: Any, config: Optional[DataConfig] = None, augment: bool = False):
        self.sequences = np.array(sequences)
        self.labels = np.array(labels)
        self.config = config or DataConfig()
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        if self.augment:
            seq = augment_landmarks(seq)
            
        seq = normalize_landmarks(seq)
        seq = pad_sequence(seq, self.config.sequence_length)
        seq = flatten_landmarks(seq)
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class ISLDataModule:
    def __init__(self, config: DataConfig):
        self.config = config
        self.sequences = None
        self.labels = None

    def load_from_directory(self, data_dir: str):
        # directory structure: data_dir/{class_name}/*.npy
        seqs = []
        lbls = []
        class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        for i, class_dir in enumerate(class_dirs):
            for file_path in class_dir.glob('*.npy'):
                seq = np.load(file_path)
                seqs.append(seq)
                lbls.append(i)
        
        max_t = max(len(s) for s in seqs) if seqs else self.config.sequence_length
        padded_seqs = [pad_sequence(s, max_t) for s in seqs]
        self.sequences = np.array(padded_seqs) if padded_seqs else np.zeros((0, max_t, 21, 3))
        self.labels = np.array(lbls)

    def load_from_json(self, json_path: str):
        with open(json_path, 'r') as f:
            manifest = json.load(f)
        seqs = []
        lbls = []
        for item in manifest:
            seq = np.load(item['path'])
            seqs.append(seq)
            lbls.append(item['label'])
        
        max_t = max(len(s) for s in seqs) if seqs else self.config.sequence_length
        padded_seqs = [pad_sequence(s, max_t) for s in seqs]
        self.sequences = np.array(padded_seqs) if padded_seqs else np.zeros((0, max_t, 21, 3))
        self.labels = np.array(lbls)

    def split(self) -> Tuple[ISLDataset, ISLDataset, ISLDataset]:
        np.random.seed(self.config.random_seed)
        n = len(self.labels)
        indices = np.random.permutation(n)
        
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)
        
        train_idx = indices[:train_end]
        val_idx = indices[train_end:val_end]
        test_idx = indices[val_end:]
        
        train_ds = ISLDataset(self.sequences[train_idx], self.labels[train_idx], self.config, augment=True)
        val_ds = ISLDataset(self.sequences[val_idx], self.labels[val_idx], self.config, augment=False)
        test_ds = ISLDataset(self.sequences[test_idx], self.labels[test_idx], self.config, augment=False)
        
        return train_ds, val_ds, test_ds

    def get_dataloaders(self, batch_size: int = 32, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
        train_ds, val_ds, test_ds = self.split()
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def train_dataloader(self, batch_size: int = 32) -> DataLoader:
        train_ds, _, _ = self.split()
        return DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    def val_dataloader(self, batch_size: int = 32) -> DataLoader:
        _, val_ds, _ = self.split()
        return DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    def test_dataloader(self, batch_size: int = 32) -> DataLoader:
        _, _, test_ds = self.split()
        return DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    def create_synthetic(self, n_samples: int = 100, num_samples: Optional[int] = None) -> Tuple[ISLDataset, ISLDataset, ISLDataset]:
        total_samples = num_samples if num_samples is not None else n_samples
        self.sequences = np.random.randn(total_samples, self.config.sequence_length, 21, 3)
        self.labels = np.random.randint(0, self.config.num_classes, total_samples)
        train_ds, val_ds, test_ds = self.split()
        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds
        return train_ds, val_ds, test_ds
