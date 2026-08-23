"""
=============================================================================
BharatSRM-Net v4: Dataset & Preprocessing Pipeline
=============================================================================
Spectral Bands:
  - 10 Bands Sentinel-2 L2A Input:
    Native 10m: B2 (Blue), B3 (Green), B4 (Red), B8 (NIR)
    Resampled 20m: B5, B6, B7 (Vegetation Red Edge), B8A (Narrow NIR), B11, B12 (SWIR)
  - Target: 4 Bands HR RGBN at 2.5m nominal resolution.
  - Validity Mask: S2cloudless + QA60 (1=valid, 0=cloud/shadow).
=============================================================================
"""

import random

import numpy as np
import torch
from torch.utils.data import Dataset


class Sentinel2SuperResolutionDataset(Dataset):
    """
    Multispectral Sentinel-2 to High-Resolution RGBN Dataset with Cloud Masking
    and Geographically Disjoint Splitting.
    """

    def __init__(
        self,
        lr_tiles: list[np.ndarray],
        hr_tiles: list[np.ndarray] | None = None,
        cloud_masks: list[np.ndarray] | None = None,
        dem_tiles: list[np.ndarray] | None = None,
        is_train: bool = True,
        augment: bool = True,
    ):
        self.lr_tiles = lr_tiles
        self.hr_tiles = hr_tiles
        self.cloud_masks = cloud_masks
        self.dem_tiles = dem_tiles
        self.is_train = is_train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.lr_tiles)

    @staticmethod
    def _radiometric_normalize(img: np.ndarray) -> np.ndarray:
        """Applies 1st/99th percentile band-wise normalization to [0.0, 1.0]."""
        # img: (C, H, W)
        norm_img = np.zeros_like(img, dtype=np.float32)
        for c in range(img.shape[0]):
            band = img[c]
            p1, p99 = np.percentile(band, 1), np.percentile(band, 99)
            if p99 > p1:
                clipped = np.clip(band, p1, p99)
                norm_img[c] = (clipped - p1) / (p99 - p1 + 1e-8)
            else:
                norm_img[c] = np.clip(band, 0.0, 1.0)
        return norm_img

    def _apply_augmentations(
        self,
        lr: np.ndarray,
        hr: np.ndarray | None,
        mask: np.ndarray,
        dem: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None]:
        # Random horizontal flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
            if hr is not None:
                hr = np.flip(hr, axis=-1).copy()
            if dem is not None:
                dem = np.flip(dem, axis=-1).copy()

        # Random vertical flip
        if random.random() > 0.5:
            lr = np.flip(lr, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()
            if hr is not None:
                hr = np.flip(hr, axis=-2).copy()
            if dem is not None:
                dem = np.flip(dem, axis=-2).copy()

        # Random 90-degree rotation
        rot_k = random.choice([0, 1, 2, 3])
        if rot_k > 0:
            lr = np.rot90(lr, k=rot_k, axes=(-2, -1)).copy()
            mask = np.rot90(mask, k=rot_k, axes=(-2, -1)).copy()
            if hr is not None:
                hr = np.rot90(hr, k=rot_k, axes=(-2, -1)).copy()
            if dem is not None:
                dem = np.rot90(dem, k=rot_k, axes=(-2, -1)).copy()

        return lr, hr, mask, dem

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        lr = self.lr_tiles[idx].astype(np.float32)  # (10, H, W)
        lr = self._radiometric_normalize(lr)

        # Cloud validity mask: (1, H, W) where 1=clear, 0=cloud/shadow
        if self.cloud_masks is not None:
            mask = self.cloud_masks[idx].astype(np.float32)
            if mask.ndim == 2:
                mask = mask[np.newaxis, ...]
        else:
            mask = np.ones((1, lr.shape[1], lr.shape[2]), dtype=np.float32)

        # HR Target (if available): (4, 4H, 4W)
        if self.hr_tiles is not None:
            hr = self.hr_tiles[idx].astype(np.float32)
            hr = self._radiometric_normalize(hr)
        else:
            hr = None

        # CartoDEM: (2, H, W) [Elevation, Slope]
        if self.dem_tiles is not None:
            dem = self.dem_tiles[idx].astype(np.float32)
        else:
            dem = np.zeros((2, lr.shape[1], lr.shape[2]), dtype=np.float32)

        if self.is_train and self.augment:
            lr, hr, mask, dem = self._apply_augmentations(lr, hr, mask, dem)

        sample = {
            "lr_input": torch.from_numpy(lr),
            "validity_mask": torch.from_numpy(mask),
            "context_dem": torch.from_numpy(dem),
        }
        if hr is not None:
            sample["hr_target"] = torch.from_numpy(hr)

        return sample
