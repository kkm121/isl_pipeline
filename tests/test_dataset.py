import pytest
import numpy as np
import torch

from src.data.preprocessing import normalize_landmarks, pad_sequence, flatten_landmarks, augment_landmarks
from src.data.dataset import ISLDataset, ISLDataModule
from src.models.config import DataConfig

def test_normalize_landmarks():
    np.random.seed(42)
    landmarks = np.random.rand(10, 21, 3)
    normalized = normalize_landmarks(landmarks)
    assert normalized.shape == (10, 21, 3)
    # wrist is usually at index 0, check if centered around it
    np.testing.assert_allclose(normalized[:, 0, :], 0, atol=1e-6)

def test_normalize_landmarks_zero_input():
    landmarks = np.zeros((10, 21, 3))
    normalized = normalize_landmarks(landmarks)
    assert normalized.shape == (10, 21, 3)
    np.testing.assert_allclose(normalized, 0, atol=1e-6)

def test_pad_sequence_shorter():
    sequence = np.random.rand(10, 21, 3)
    padded = pad_sequence(sequence, target_length=20)
    assert padded.shape == (20, 21, 3)
    np.testing.assert_allclose(padded[:10], sequence)
    np.testing.assert_allclose(padded[10:], 0)

def test_pad_sequence_longer():
    sequence = np.random.rand(30, 21, 3)
    padded = pad_sequence(sequence, target_length=20)
    assert padded.shape == (20, 21, 3)
    np.testing.assert_allclose(padded, sequence[:20])

def test_pad_sequence_exact():
    sequence = np.random.rand(20, 21, 3)
    padded = pad_sequence(sequence, target_length=20)
    assert padded.shape == (20, 21, 3)
    np.testing.assert_allclose(padded, sequence)

def test_flatten_landmarks():
    sequence = np.random.rand(10, 21, 3)
    flattened = flatten_landmarks(sequence)
    assert flattened.shape == (10, 63)
    np.testing.assert_allclose(flattened[0, :3], sequence[0, 0, :])

def test_augment_landmarks_shape():
    np.random.seed(42)
    sequence = np.random.rand(10, 21, 3)
    augmented = augment_landmarks(sequence, p=1.0)
    assert augmented.shape == sequence.shape

def test_augment_landmarks_different():
    np.random.seed(42)
    sequence = np.random.rand(10, 21, 3)
    augmented = augment_landmarks(sequence, p=1.0)
    assert not np.allclose(sequence, augmented)

def test_isl_dataset_creation():
    np.random.seed(42)
    data = [np.random.rand(10, 21, 3) for _ in range(5)]
    labels = [0, 1, 2, 3, 4]
    dataset = ISLDataset(data, labels, augment=False)
    assert len(dataset) == 5

def test_isl_dataset_getitem():
    np.random.seed(42)
    data = [np.random.rand(10, 21, 3) for _ in range(5)]
    labels = [0, 1, 2, 3, 4]
    dataset = ISLDataset(data, labels, augment=False)
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.dtype == torch.float32
    assert y.dtype == torch.long

def test_isl_datamodule_synthetic():
    config = DataConfig(data_dir=".", num_landmarks=21, landmark_dim=3, sequence_length=15, num_classes=5, train_split=0.6, val_split=0.2, test_split=0.2)
    dm = ISLDataModule(config)
    dm.create_synthetic(num_samples=100)
    assert len(dm.train_dataset) == 60
    assert len(dm.val_dataset) == 20
    assert len(dm.test_dataset) == 20

def test_isl_datamodule_split_ratios():
    config = DataConfig(data_dir=".", num_landmarks=21, landmark_dim=3, sequence_length=15, num_classes=5, train_split=0.8, val_split=0.1, test_split=0.1)
    dm = ISLDataModule(config)
    dm.create_synthetic(num_samples=100)
    assert len(dm.train_dataset) == 80
    assert len(dm.val_dataset) == 10
    assert len(dm.test_dataset) == 10
