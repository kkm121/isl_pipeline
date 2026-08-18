"""Unit tests for the INCLUDE dataset loader, preprocessing, and batch collation.

Tests cover:
- Sequence padding and truncation (adaptive_pad_or_truncate)
- Signer-disjoint dataset splitting (SignerDisjointSplitter)
- Variable-length batch collation (collate_variable_length)
- 76-keypoint mid-shoulder spatial normalization (normalize_landmarks)
- GISLR tall-format parquet 76kp extraction (_extract_76kp_from_gislr_frame)
"""

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.include_dataset import (
    FACE_NMM_IDX,
    POSE_UPPER_BODY_IDX,
    SignerDisjointSplitter,
    _extract_76kp_from_gislr_frame,
    adaptive_pad_or_truncate,
    collate_variable_length,
)
from src.data.preprocessing import normalize_landmarks


@pytest.fixture
def dummy_short_sequence() -> np.ndarray:
    """Fixture returning a 2D sequence shorter than the target length: shape (30, 152)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((30, 152)).astype(np.float32)


@pytest.fixture
def dummy_long_sequence() -> np.ndarray:
    """Fixture returning a 2D sequence longer than the target length: shape (300, 152)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((300, 152)).astype(np.float32)


@pytest.fixture
def dummy_exact_sequence() -> np.ndarray:
    """Fixture returning a 2D sequence matching exact target length: shape (150, 152)."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((150, 152)).astype(np.float32)


@pytest.fixture
def dummy_landmarks_76kp() -> np.ndarray:
    """Fixture returning unnormalized 76-keypoint holistic landmarks: shape (1, 76, 3)."""
    rng = np.random.default_rng(42)
    return rng.uniform(-10.0, 10.0, size=(1, 76, 3)).astype(np.float32)


@pytest.fixture
def mock_gislr_frame_df() -> pd.DataFrame:
    """Fixture returning a mock single-frame GISLR tall-format DataFrame.

    Columns: type, landmark_index, x, y, z
    Covers left_hand (21), right_hand (21), pose (subset), face (subset).
    """
    rows = []
    rng = np.random.default_rng(0)
    # Left hand: 21 landmarks
    for li in range(21):
        rows.append({"frame": 0, "type": "left_hand", "landmark_index": li,
                     "x": rng.random(), "y": rng.random(), "z": rng.random()})
    # Right hand: 21 landmarks
    for li in range(21):
        rows.append({"frame": 0, "type": "right_hand", "landmark_index": li,
                     "x": rng.random(), "y": rng.random(), "z": rng.random()})
    # Upper body pose: POSE_UPPER_BODY_IDX landmarks
    for li in POSE_UPPER_BODY_IDX:
        rows.append({"frame": 0, "type": "pose", "landmark_index": li,
                     "x": rng.random(), "y": rng.random(), "z": rng.random()})
    # Face NMMs: FACE_NMM_IDX landmarks
    for li in FACE_NMM_IDX:
        rows.append({"frame": 0, "type": "face", "landmark_index": li,
                     "x": rng.random(), "y": rng.random(), "z": rng.random()})
    return pd.DataFrame(rows)


def test_adaptive_pad_short_sequence(dummy_short_sequence: np.ndarray) -> None:
    """Test that a (30, 152) array gets centered and zero-padded to (150, 152)."""
    target_len = 150
    padded = adaptive_pad_or_truncate(dummy_short_sequence, max_len=target_len)

    assert padded.shape == (target_len, 152)

    pad_len = target_len - 30
    pad_front = pad_len // 2  # 60

    # Front padding should be zeros
    np.testing.assert_allclose(padded[:pad_front], 0.0)
    # Middle section should match the original sequence
    np.testing.assert_allclose(padded[pad_front : pad_front + 30], dummy_short_sequence)
    # Back padding should be zeros
    np.testing.assert_allclose(padded[pad_front + 30 :], 0.0)


def test_adaptive_pad_long_sequence(dummy_long_sequence: np.ndarray) -> None:
    """Test that a (300, 152) array gets center-cropped to (150, 152)."""
    target_len = 150
    cropped = adaptive_pad_or_truncate(dummy_long_sequence, max_len=target_len)

    assert cropped.shape == (target_len, 152)

    start = (300 - target_len) // 2  # 75
    expected = dummy_long_sequence[start : start + target_len, :]
    np.testing.assert_allclose(cropped, expected)


def test_adaptive_pad_exact_sequence(dummy_exact_sequence: np.ndarray) -> None:
    """Test that a (150, 152) array is returned unchanged."""
    target_len = 150
    result = adaptive_pad_or_truncate(dummy_exact_sequence, max_len=target_len)

    assert result.shape == (target_len, 152)
    np.testing.assert_allclose(result, dummy_exact_sequence)


def test_signer_disjoint_no_overlap() -> None:
    """Verify train/val/test signer sets have zero intersection."""
    signer_ids = list(range(1, 16)) * 4  # 60 samples across 15 signers
    mock_dataset = MagicMock()
    mock_dataset.signers = signer_ids

    splitter = SignerDisjointSplitter(mock_dataset)

    train_signers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    val_signers = [11, 12, 13]
    test_signers = [14, 15]

    train_idx, val_idx, test_idx = splitter.split(train_signers, val_signers, test_signers)

    assert len(train_idx) > 0
    assert len(val_idx) > 0
    assert len(test_idx) > 0

    set_train = set(train_idx)
    set_val = set(val_idx)
    set_test = set(test_idx)
    assert len(set_train.intersection(set_val)) == 0
    assert len(set_train.intersection(set_test)) == 0
    assert len(set_val.intersection(set_test)) == 0

    # Verify signer-level zero intersection
    train_signer_set = {signer_ids[i] for i in train_idx}
    val_signer_set = {signer_ids[i] for i in val_idx}
    test_signer_set = {signer_ids[i] for i in test_idx}
    assert train_signer_set.isdisjoint(val_signer_set)
    assert train_signer_set.isdisjoint(test_signer_set)
    assert val_signer_set.isdisjoint(test_signer_set)


def test_collate_variable_length() -> None:
    """Verify collate_variable_length pads a batch of different-length sequences."""
    t1 = torch.randn(20, 152)
    t2 = torch.randn(45, 152)
    t3 = torch.randn(10, 152)

    batch = [(t1, 0, 101), (t2, 1, 102), (t3, 2, 103)]
    padded_seqs, lengths, labels, signers = collate_variable_length(batch)

    assert padded_seqs.shape == (3, 45, 152)
    assert torch.equal(lengths, torch.tensor([20, 45, 10], dtype=torch.long))
    assert torch.equal(labels, torch.tensor([0, 1, 2], dtype=torch.long))
    assert torch.equal(signers, torch.tensor([101, 102, 103], dtype=torch.long))

    assert torch.allclose(padded_seqs[0, :20], t1)
    assert torch.allclose(padded_seqs[1, :45], t2)
    assert torch.allclose(padded_seqs[2, :10], t3)
    assert torch.all(padded_seqs[0, 20:] == 0.0)
    assert torch.all(padded_seqs[2, 10:] == 0.0)


def test_normalize_landmarks_76kp(dummy_landmarks_76kp: np.ndarray) -> None:
    """Verify normalize_landmarks centers a (1, 76, 3) array on mid-shoulder."""
    normalized = normalize_landmarks(dummy_landmarks_76kp)
    assert normalized.shape == (1, 76, 3)
    # After normalization, mid-shoulder centroid should be at origin
    mid_shoulder_norm = (normalized[:, 43, :] + normalized[:, 44, :]) / 2.0
    np.testing.assert_allclose(mid_shoulder_norm, 0.0, atol=1e-5)


def test_extract_76kp_returns_correct_shape(mock_gislr_frame_df: pd.DataFrame) -> None:
    """Verify _extract_76kp_from_gislr_frame returns shape (76, 3) from GISLR tall format."""
    kp_array = _extract_76kp_from_gislr_frame(mock_gislr_frame_df)

    assert isinstance(kp_array, np.ndarray)
    assert kp_array.shape == (76, 3)
    assert kp_array.dtype == np.float32
    # All hand landmarks should be non-zero (we provided values for all 21+21)
    left_hand_block = kp_array[:21]
    right_hand_block = kp_array[21:42]
    assert not np.all(left_hand_block == 0.0), "Left hand values should not all be zero"
    assert not np.all(right_hand_block == 0.0), "Right hand values should not all be zero"
