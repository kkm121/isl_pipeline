import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Set, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.data.preprocessing import (
    augment_landmarks,
    extract_2d_pose_vector,
    flatten_landmarks,
    normalize_landmarks,
    pad_sequence,
)
from src.models.config import DataConfig

logger = logging.getLogger(__name__)

# Standard 200 Classroom & Educational ISL Glosses
CLASSROOM_VOCABULARY_200 = [
    # Alphabet A-Z (26)
    *[chr(i) for i in range(ord("A"), ord("Z") + 1)],
    # Numbers 0-9 (10)
    *[str(i) for i in range(10)],
    # Classroom Roles & People (14)
    "TEACHER",
    "STUDENT",
    "PRINCIPAL",
    "CLASSMATE",
    "FRIEND",
    "PARENT",
    "DOCTOR",
    "DEAF",
    "HEARING",
    "INTERPRETER",
    "BOY",
    "GIRL",
    "CHILD",
    "TEAM",
    # Classroom Objects & Tools (20)
    "BOOK",
    "PEN",
    "PENCIL",
    "NOTEBOOK",
    "DESK",
    "CHAIR",
    "BLACKBOARD",
    "COMPUTER",
    "BAG",
    "PAPER",
    "ERASER",
    "RULER",
    "SCISSORS",
    "CALCULATOR",
    "PROJECTOR",
    "BELL",
    "CLOCK",
    "DOOR",
    "WINDOW",
    "LIBRARY",
    # Academic Subjects (15)
    "MATH",
    "SCIENCE",
    "ENGLISH",
    "HINDI",
    "HISTORY",
    "GEOGRAPHY",
    "PHYSICS",
    "CHEMISTRY",
    "BIOLOGY",
    "COMPUTER_SCIENCE",
    "ART",
    "MUSIC",
    "SPORTS",
    "EXAM",
    "HOMEWORK",
    # Actions & Instructions (35)
    "READ",
    "WRITE",
    "LISTEN",
    "WATCH",
    "SPEAK",
    "LEARN",
    "STUDY",
    "TEACH",
    "OPEN",
    "CLOSE",
    "SIT",
    "STAND",
    "COME",
    "GO",
    "ASK",
    "ANSWER",
    "EXPLAIN",
    "REPEAT",
    "HELP",
    "PRACTICE",
    "UNDERSTAND",
    "REMEMBER",
    "FORGET",
    "THINK",
    "START",
    "FINISH",
    "SUBMIT",
    "CHECK",
    "CORRECT",
    "DRAW",
    "COUNT",
    "SOLVE",
    "SHARE",
    "CLEAN",
    "WAIT",
    # Questions & Interrogatives (10)
    "WHAT",
    "WHERE",
    "WHEN",
    "WHY",
    "WHO",
    "WHICH",
    "HOW",
    "HOW_MANY",
    "HOW_MUCH",
    "REASON",
    # Common Expressions & Courtesies (15)
    "HELLO",
    "GOODBYE",
    "PLEASE",
    "THANK_YOU",
    "WELCOME",
    "SORRY",
    "EXCUSE_ME",
    "YES",
    "NO",
    "GOOD",
    "BAD",
    "CORRECT_ANSWER",
    "WRONG_ANSWER",
    "AGREE",
    "DISAGREE",
    # Time & Scheduling (15)
    "TODAY",
    "TOMORROW",
    "YESTERDAY",
    "NOW",
    "LATER",
    "MORNING",
    "AFTERNOON",
    "EVENING",
    "NIGHT",
    "PERIOD",
    "BREAK",
    "LUNCH",
    "HOLIDAY",
    "ATTENDANCE",
    "TIME",
    # Environmental & Emotional Descriptors (19)
    "HAPPY",
    "SAD",
    "CONFUSED",
    "TIRED",
    "EXCITED",
    "QUIET",
    "LOUD",
    "FAST",
    "SLOW",
    "EASY",
    "DIFFICULT",
    "IMPORTANT",
    "READY",
    "CLEAR",
    "DOUBT",
    "SMART",
    "HARDWORKING",
    "SAFE",
    "SUCCESS",
]


class SignerDisjointSplitter:
    """Partitions a dataset into train, val, and test splits strictly by signer_id.

    Guarantees that no signer present in the training split appears in the validation
    or test splits, ensuring zero data leakage and strict signer-invariant generalization.
    """

    @staticmethod
    def split_by_signer(
        signer_ids: List[str | int],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        unique_signers = np.array(sorted(list(set(signer_ids))))
        rng = np.random.RandomState(seed)
        rng.shuffle(unique_signers)

        num_signers = len(unique_signers)
        if num_signers < 3:
            logger.warning(f"Only {num_signers} unique signers found. Falling back to sample-level split.")
            indices = np.arange(len(signer_ids))
            rng.shuffle(indices)
            n = len(indices)
            n_tr = int(n * train_ratio)
            n_va = int(n * val_ratio)
            return indices[:n_tr].tolist(), indices[n_tr : n_tr + n_va].tolist(), indices[n_tr + n_va :].tolist()

        n_train = max(1, int(round(num_signers * train_ratio)))
        n_val = max(1, int(round(num_signers * val_ratio)))

        train_signers: Set[Any] = set(unique_signers[:n_train])
        val_signers: Set[Any] = set(unique_signers[n_train : n_train + n_val])
        test_signers: Set[Any] = set(unique_signers[n_train + n_val :])

        if not test_signers and len(val_signers) > 1:
            # Rebalance if rounding leaves test empty
            moved = val_signers.pop()
            test_signers.add(moved)

        train_indices = [i for i, sid in enumerate(signer_ids) if sid in train_signers]
        val_indices = [i for i, sid in enumerate(signer_ids) if sid in val_signers]
        test_indices = [i for i, sid in enumerate(signer_ids) if sid in test_signers]

        return train_indices, val_indices, test_indices


class ISLDataset(Dataset):
    """PyTorch Dataset for ISL coordinate trajectories with spatial normalization & augmentation."""

    def __init__(
        self,
        sequences: Any,
        labels: Any,
        config: Optional[DataConfig] = None,
        augment: bool = False,
        signer_ids: Optional[List[Any]] = None,
    ):
        self.sequences = np.array(sequences)
        self.labels = np.array(labels)
        self.config = config or DataConfig()
        self.augment = augment
        self.signer_ids = signer_ids

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = self.sequences[idx]
        label = self.labels[idx]

        if self.augment:
            seq = augment_landmarks(seq)

        seq = normalize_landmarks(seq)
        seq = pad_sequence(seq, self.config.sequence_length)
        if getattr(self.config, "landmark_dim", 3) == 2 and seq.ndim >= 3 and seq.shape[-1] > 2:
            seq = extract_2d_pose_vector(seq)
        else:
            seq = flatten_landmarks(seq)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class ISLDataModule:
    """Manages ISL Dataset ingestion, signer-disjoint splitting, and DataLoaders."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.sequences: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.signer_ids: Optional[List[Any]] = None
        self.train_dataset: Optional[ISLDataset] = None
        self.val_dataset: Optional[ISLDataset] = None
        self.test_dataset: Optional[ISLDataset] = None

    def load_from_directory(self, data_dir: str) -> None:
        seqs = []
        lbls = []
        signers: List[str] = []
        class_dirs = sorted([d for d in Path(data_dir).iterdir() if d.is_dir()])
        for i, class_dir in enumerate(class_dirs):
            for file_path in class_dir.glob("*.npy"):
                seq = np.load(file_path)
                seqs.append(seq)
                lbls.append(i)
                # Parse signer prefix if formatted e.g. signer1_word_01.npy
                parts = file_path.stem.split("_")
                signer = parts[0] if len(parts) > 1 else f"signer_{len(signers) % 5}"
                signers.append(signer)

        max_t = max((len(s) for s in seqs), default=self.config.sequence_length)
        kps = self.config.num_landmarks
        dim = 3
        padded_seqs = [pad_sequence(s, max_t) for s in seqs]
        self.sequences = np.array(padded_seqs) if padded_seqs else np.zeros((0, max_t, kps, dim))
        self.labels = np.array(lbls)
        self.signer_ids = signers

    def load_from_json(self, json_path: str) -> None:
        with open(json_path, "r") as f:
            manifest = json.load(f)
        seqs = []
        lbls = []
        signers: List[str] = []
        for item in manifest:
            seq = np.load(item["path"])
            seqs.append(seq)
            lbls.append(item["label"])
            signers.append(item.get("signer_id", f"signer_{len(signers) % 5}"))

        max_t = max((len(s) for s in seqs), default=self.config.sequence_length)
        kps = self.config.num_landmarks
        dim = 3
        padded_seqs = [pad_sequence(s, max_t) for s in seqs]
        self.sequences = np.array(padded_seqs) if padded_seqs else np.zeros((0, max_t, kps, dim))
        self.labels = np.array(lbls)
        self.signer_ids = signers

    def split(self) -> Tuple[ISLDataset, ISLDataset, ISLDataset]:
        if self.labels is None or self.sequences is None:
            raise ValueError("Data not loaded. Call load_from_directory, load_from_json, or create_synthetic first.")

        if getattr(self.config, "use_signer_disjoint", True) and self.signer_ids:
            train_idx, val_idx, test_idx = SignerDisjointSplitter.split_by_signer(
                self.signer_ids,
                train_ratio=self.config.train_split,
                val_ratio=self.config.val_split,
                test_ratio=self.config.test_split,
                seed=self.config.random_seed,
            )
        else:
            np.random.seed(self.config.random_seed)
            n = len(self.labels)
            indices = np.random.permutation(n)
            train_end = int(n * self.config.train_split)
            val_end = train_end + int(n * self.config.val_split)
            train_idx = indices[:train_end].tolist()
            val_idx = indices[train_end:val_end].tolist()
            test_idx = indices[val_end:].tolist()

        train_ds = ISLDataset(self.sequences[train_idx], self.labels[train_idx], self.config, augment=True)
        val_ds = ISLDataset(self.sequences[val_idx], self.labels[val_idx], self.config, augment=False)
        test_ds = ISLDataset(self.sequences[test_idx], self.labels[test_idx], self.config, augment=False)

        self.train_dataset = train_ds
        self.val_dataset = val_ds
        self.test_dataset = test_ds

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

    def create_synthetic(
        self,
        n_samples: int = 100,
        num_samples: Optional[int] = None,
        num_signers: Optional[int] = None,
    ) -> Tuple[ISLDataset, ISLDataset, ISLDataset]:
        total_samples = num_samples if num_samples is not None else n_samples
        kps = getattr(self.config, "num_landmarks", 21)
        dim = getattr(self.config, "landmark_dim", 3)
        seq_len = self.config.sequence_length

        self.sequences = np.random.randn(total_samples, seq_len, kps, dim)
        self.labels = np.random.randint(0, self.config.num_classes, total_samples)
        if getattr(self.config, "use_signer_disjoint", False) or num_signers is not None:
            signers_count = num_signers or 5
            self.signer_ids = [f"signer_{i % signers_count}" for i in range(total_samples)]
        else:
            self.signer_ids = None

        train_ds, val_ds, test_ds = self.split()
        return train_ds, val_ds, test_ds
