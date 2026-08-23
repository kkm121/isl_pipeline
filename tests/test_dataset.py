"""
Unit Tests for Dataset and Preprocessing Pipeline
"""

import numpy as np
import pytest
from unittest.mock import patch

from src.data.dataset import Sentinel2SuperResolutionDataset


def test_radiometric_normalize_range():
    dataset = Sentinel2SuperResolutionDataset(lr_tiles=[])
    
    # Create an image with extreme values
    img = np.random.normal(loc=100, scale=50, size=(10, 32, 32)).astype(np.float32)
    
    norm_img = dataset._radiometric_normalize(img)
    
    # Check shape
    assert norm_img.shape == img.shape
    
    # Output must be in [0, 1]
    assert np.all(norm_img >= 0.0)
    assert np.all(norm_img <= 1.0)


@patch("src.data.dataset.random.random")
@patch("src.data.dataset.random.choice")
def test_augmentation_only_train(mock_choice, mock_random):
    # Set random to always trigger augmentations if called
    mock_random.return_value = 0.9 
    mock_choice.return_value = 1 # 90 degree rot
    
    lr = np.zeros((10, 16, 16), dtype=np.float32)
    lr[0, 0, 0] = 1.0 # Marker to track flips/rots
    
    dataset = Sentinel2SuperResolutionDataset(
        lr_tiles=[lr],
        is_train=False,
        augment=True, # Even if augment=True, is_train=False should prevent it
    )
    
    sample = dataset[0]
    out_lr = sample["lr_input"].numpy()
    
    # Should be exactly the original image (marker at 0,0) after normalization
    # Normalization of mostly zeros with one 1.0 -> 1st percentile=0, 99th percentile=0
    # Wait, the normalization logic uses percentile. For array with one 1.0, 99th percentile might be 0.
    # If p99 == p1, it clips to [0, 1]. So marker remains 1.0.
    assert out_lr[0, 0, 0] == 1.0


def test_augmentation_preserves_shape():
    lr = np.random.rand(10, 16, 16).astype(np.float32)
    hr = np.random.rand(4, 64, 64).astype(np.float32)
    mask = np.random.rand(1, 16, 16).astype(np.float32)
    dem = np.random.rand(2, 16, 16).astype(np.float32)
    
    dataset = Sentinel2SuperResolutionDataset(
        lr_tiles=[lr],
        hr_tiles=[hr],
        cloud_masks=[mask],
        dem_tiles=[dem],
        is_train=True,
        augment=True,
    )
    
    # Force some augmentations randomly
    sample = dataset[0]
    
    assert sample["lr_input"].shape == (10, 16, 16)
    assert sample["hr_target"].shape == (4, 64, 64)
    assert sample["validity_mask"].shape == (1, 16, 16)
    assert sample["context_dem"].shape == (2, 16, 16)


def test_missing_hr_returns_no_key():
    lr = np.random.rand(10, 16, 16).astype(np.float32)
    
    dataset = Sentinel2SuperResolutionDataset(
        lr_tiles=[lr],
        hr_tiles=None,
    )
    
    sample = dataset[0]
    
    assert "lr_input" in sample
    assert "hr_target" not in sample
