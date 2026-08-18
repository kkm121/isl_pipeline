"""
Real ISL/INCLUDE Parquet Dataset Loader.

Handles BOTH formats:
  1. GISLR tall format  (swaptr/indian-sign-language-mediapipe-holistic-landmarks on Kaggle)
       columns: frame, row_id, type, landmark_index, x, y, z
  2. Official iSign poses (Exploration-Lab/iSign pose-format .pose files)

The researcher confirmed the Kaggle dataset uses the GISLR tall format, not the wide
x_0, y_0 flat format. This module handles both.
"""

from __future__ import annotations

import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# 76-keypoint extraction spec (GISLR tall format)
# ---------------------------------------------------------------------------
# We extract:  21 left-hand + 21 right-hand + 11 upper-body pose + 23 face NMMs = 76 kp
POSE_UPPER_BODY_IDX: List[int] = [0, 11, 12, 13, 14, 15, 16, 23, 24, 1, 4]  # 11 pts
FACE_NMM_IDX: List[int] = [
    70, 63, 105, 66, 107, 336, 296, 334, 293, 300,  # eyebrows (10)
    33, 133, 362, 263,  # eye corners (4)
    0, 13, 14, 17, 37, 267, 78, 308, 82,  # mouth (9)
]  # 23 pts

# Total: 21 + 21 + 11 + 23 = 76
TOTAL_KP = 21 + 21 + len(POSE_UPPER_BODY_IDX) + len(FACE_NMM_IDX)
assert TOTAL_KP == 76, f"Expected 76 keypoints, got {TOTAL_KP}"


def _extract_76kp_from_gislr_frame(
    frame_df: pd.DataFrame,
) -> np.ndarray:
    """Extract (76, 3) array from one frame's rows in GISLR tall format.

    Args:
        frame_df: DataFrame with columns [type, landmark_index, x, y, z]
                  for a single frame.
    Returns:
        np.ndarray of shape (76, 3) — zeros where landmarks are missing/NaN.
    """
    kp = np.zeros((76, 3), dtype=np.float32)
    ptr = 0

    for ltype, indices in [
        ("left_hand", range(21)),
        ("right_hand", range(21)),
    ]:
        subset = frame_df[frame_df["type"] == ltype]
        idx_map: Dict[int, Tuple[float, float, float]] = {}
        for _, row in subset.iterrows():
            li = int(row["landmark_index"])
            idx_map[li] = (
                float(row["x"]) if not pd.isna(row["x"]) else 0.0,
                float(row["y"]) if not pd.isna(row["y"]) else 0.0,
                float(row["z"]) if not pd.isna(row["z"]) else 0.0,
            )
        for li in indices:
            if li in idx_map:
                kp[ptr] = idx_map[li]
            ptr += 1

    pose_subset = frame_df[frame_df["type"] == "pose"]
    pose_map: Dict[int, Tuple[float, float, float]] = {}
    for _, row in pose_subset.iterrows():
        li = int(row["landmark_index"])
        pose_map[li] = (
            float(row["x"]) if not pd.isna(row["x"]) else 0.0,
            float(row["y"]) if not pd.isna(row["y"]) else 0.0,
            float(row["z"]) if not pd.isna(row["z"]) else 0.0,
        )
    for li in POSE_UPPER_BODY_IDX:
        if li in pose_map:
            kp[ptr] = pose_map[li]
        ptr += 1

    face_subset = frame_df[frame_df["type"] == "face"]
    face_map: Dict[int, Tuple[float, float, float]] = {}
    for _, row in face_subset.iterrows():
        li = int(row["landmark_index"])
        face_map[li] = (
            float(row["x"]) if not pd.isna(row["x"]) else 0.0,
            float(row["y"]) if not pd.isna(row["y"]) else 0.0,
            float(row["z"]) if not pd.isna(row["z"]) else 0.0,
        )
    for li in FACE_NMM_IDX:
        if li in face_map:
            kp[ptr] = face_map[li]
        ptr += 1

    return kp


def load_gislr_parquet_to_sequence(
    parquet_path: str,
    max_len: int = 150,
) -> np.ndarray:
    """Load one GISLR-format parquet file and return (max_len, 152) float32 array.

    Extracts 76 keypoints × 2D (x, y) = 152 features per frame.
    Center-crops long sequences, zero-pads short ones.

    Args:
        parquet_path: Path to a GISLR-format per-video parquet file.
        max_len: Target sequence length after padding/cropping.
    Returns:
        np.ndarray shape (max_len, 152).
    Raises:
        FileNotFoundError: if parquet_path does not exist.
        ValueError: if file has unexpected schema (not GISLR tall format).
    """
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Landmark parquet not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)

    required = {"frame", "type", "landmark_index", "x", "y"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"Expected GISLR tall columns {required}, got {set(df.columns)}. "
            f"File: {parquet_path}"
        )

    frames_sorted = sorted(df["frame"].unique())
    seq_frames: List[np.ndarray] = []

    for f in frames_sorted:
        frame_df = df[df["frame"] == f]
        kp = _extract_76kp_from_gislr_frame(frame_df)
        xy = kp[:, :2].flatten()  # (76, 2) → (152,)
        seq_frames.append(xy)

    if not seq_frames:
        return np.zeros((max_len, 152), dtype=np.float32)

    seq = np.stack(seq_frames, axis=0).astype(np.float32)  # (T, 152)
    return adaptive_pad_or_truncate(seq, max_len=max_len)


class IncludeParquetDataset(Dataset):
    """Loads the swaptr/indian-sign-language-mediapipe-holistic-landmarks dataset.

    Expects the Kaggle directory structure:
        data_root/
            train.csv        — columns: path, participant_id, sequence_id, sign
            label_map.json   — {"sign_name": class_id, ...}
            train_landmark_files/  — individual .parquet files per sequence

    Returns (seq: Tensor[max_len, 152], label: int, signer_id: int).
    """

    def __init__(self, data_root: str, max_len: int = 150) -> None:
        import json

        if not os.path.isdir(data_root):
            raise FileNotFoundError(
                f"Dataset root not found: {data_root}. "
                "Mount swaptr/indian-sign-language-mediapipe-holistic-landmarks as input."
            )

        csv_path = os.path.join(data_root, "train.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"train.csv not found at {csv_path}. Dataset structure unexpected."
            )

        label_map_path = os.path.join(data_root, "label_map.json")
        if not os.path.exists(label_map_path):
            raise FileNotFoundError(f"label_map.json not found at {label_map_path}.")

        with open(label_map_path, encoding="utf-8") as f:
            label_map: Dict[str, int] = json.load(f)

        meta = pd.read_csv(csv_path)
        self.max_len = max_len
        self.sequences: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.signers: List[int] = []

        for _, row in meta.iterrows():
            rel_path = str(row.get("path", ""))
            sign = str(row.get("sign", ""))
            signer_id = int(row.get("participant_id", 0))
            label_id = label_map.get(sign, -1)

            if label_id < 0:
                continue

            abs_path = os.path.join(data_root, rel_path)
            try:
                seq_np = load_gislr_parquet_to_sequence(abs_path, max_len=max_len)
            except (FileNotFoundError, ValueError, Exception):
                continue

            self.sequences.append(torch.tensor(seq_np, dtype=torch.float32))
            self.labels.append(label_id)
            self.signers.append(signer_id)

        if not self.sequences:
            raise RuntimeError(
                f"No valid sequences loaded from {data_root}. "
                "Verify the dataset is correctly mounted and parquet files exist."
            )

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        """Returns (sequence [max_len, 152], label, signer_id)."""
        return self.sequences[idx], self.labels[idx], self.signers[idx]


class SignerDisjointSplitter:
    """Splits dataset indices by signer_id — no signer overlap between splits."""

    def __init__(self, dataset: IncludeParquetDataset) -> None:
        self.signers = np.array(dataset.signers)

    def split(
        self,
        train_signers: List[int],
        val_signers: List[int],
        test_signers: List[int],
    ) -> Tuple[List[int], List[int], List[int]]:
        train_idx: List[int] = np.where(np.isin(self.signers, train_signers))[0].tolist()
        val_idx: List[int] = np.where(np.isin(self.signers, val_signers))[0].tolist()
        test_idx: List[int] = np.where(np.isin(self.signers, test_signers))[0].tolist()
        return train_idx, val_idx, test_idx


def adaptive_pad_or_truncate(seq: np.ndarray, max_len: int = 150) -> np.ndarray:
    """Center-crops long sequences; zero-pads short sequences.

    Args:
        seq: np.ndarray of shape (T, feat_dim).
        max_len: Target sequence length.
    Returns:
        np.ndarray of shape (max_len, feat_dim).
    """
    t, feat_dim = seq.shape
    if t > max_len:
        start = (t - max_len) // 2
        return seq[start : start + max_len, :]
    if t < max_len:
        pad_front = (max_len - t) // 2
        pad_back = max_len - t - pad_front
        return np.pad(seq, ((pad_front, pad_back), (0, 0)), mode="constant", constant_values=0)
    return seq


def collate_variable_length(
    batch: List[Tuple[torch.Tensor, int, int]],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """DataLoader collate_fn — pads variable-length sequences within a batch.

    Returns:
        padded_seqs: (B, max_T, feat_dim)
        lengths: (B,) — original sequence length of each item
        labels: (B,)
        signers: (B,)
    """
    seqs, labels, signers = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    padded_seqs = pad_sequence(list(seqs), batch_first=True, padding_value=0.0)
    labels_t = torch.tensor(list(labels), dtype=torch.long)
    signers_t = torch.tensor(list(signers), dtype=torch.long)
    return padded_seqs, lengths, labels_t, signers_t
