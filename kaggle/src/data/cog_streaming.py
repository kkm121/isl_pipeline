"""
=============================================================================
BharatSRM-Net v4: Cloud-Optimized GeoTIFF (COG) Windowed Streaming Pipeline
=============================================================================
Section 5.6.1 Specification:
  - Scene-level streaming via windowed reads for large-scale satellite rasters
    (e.g., 10,980 x 10,980 Sentinel-2 tiles).
  - Eliminates out-of-memory overhead by reading tile bounding windows (e.g. 256x256)
    on-the-fly directly from Cloud-Optimized GeoTIFFs (COGs).
  - Dual Backend:
    1. Rasterio Windowed Read Backend (when rasterio is available).
    2. High-Performance NumPy Memmap/Array Fallback Backend (when rasterio is absent).
=============================================================================
"""

import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

# Try importing rasterio conditionally
try:
    import rasterio
    from rasterio.windows import Window

    HAS_RASTERIO = True
except ImportError:
    rasterio = None  # type: ignore
    Window = None  # type: ignore
    HAS_RASTERIO = False


@dataclass
class WindowSpec:
    """Bounding window coordinates for spatial slicing."""

    row_off: int
    col_off: int
    height: int
    width: int


def generate_tile_grid(
    scene_height: int,
    scene_width: int,
    tile_size: int = 256,
    overlap: int = 32,
    pad_boundary: bool = True,
) -> list[WindowSpec]:
    """
    Generates a regular grid of overlapping window specifications covering a scene.

    Args:
        scene_height: Height of full scene raster in pixels.
        scene_width: Width of full scene raster in pixels.
        tile_size: Spatial height and width of each tile.
        overlap: Overlap in pixels between adjacent tiles.
        pad_boundary: If True, adjust boundary offsets so tiles stay within bounds
                      or have exact tile_size dimensions.

    Returns:
        List of WindowSpec objects.
    """
    stride = tile_size - overlap
    if stride <= 0:
        raise ValueError(
            f"Overlap ({overlap}) must be strictly less than tile_size ({tile_size})"
        )

    windows: list[WindowSpec] = []

    row_offsets = list(range(0, max(1, scene_height - overlap), stride))
    col_offsets = list(range(0, max(1, scene_width - overlap), stride))

    for r in row_offsets:
        # Check boundary
        if pad_boundary and (r + tile_size > scene_height):
            actual_r = max(0, scene_height - tile_size)
            h = min(tile_size, scene_height)
        else:
            actual_r = r
            h = min(tile_size, scene_height - actual_r)

        for c in col_offsets:
            if pad_boundary and (c + tile_size > scene_width):
                actual_c = max(0, scene_width - tile_size)
                w = min(tile_size, scene_width)
            else:
                actual_c = c
                w = min(tile_size, scene_width - actual_c)

            win = WindowSpec(
                row_off=int(actual_r),
                col_off=int(actual_c),
                height=int(h),
                width=int(w),
            )
            # Avoid duplicate windows if clamped
            if not any(
                w.row_off == win.row_off
                and w.col_off == win.col_off
                and w.height == win.height
                and w.width == win.width
                for w in windows
            ):
                windows.append(win)

    return windows


class COGWindowReader:
    """
    Reads windowed spatial slices from Cloud-Optimized GeoTIFFs or NumPy arrays.
    Seamlessly falls back to pure NumPy when rasterio is unavailable.
    """

    def __init__(
        self,
        source: str | np.ndarray,
        num_bands: int | None = None,
        band_indices: Sequence[int] | None = None,
    ):
        """
        Args:
            source: Path to GeoTIFF file or in-memory NumPy array (C, H, W).
            num_bands: Expected number of channels (optional).
            band_indices: 1-indexed band list for rasterio or 0-indexed for numpy.
        """
        self.source = source
        self.num_bands = num_bands
        self.band_indices = band_indices
        self.is_array = isinstance(source, np.ndarray)

        if not self.is_array:
            self.file_path = str(source)
            self.use_rasterio = (
                HAS_RASTERIO
                and os.path.exists(self.file_path)
                and not self.file_path.endswith(".npy")
            )
        else:
            self.file_path = None
            self.use_rasterio = False

    def read_window(
        self,
        window: WindowSpec,
        normalize: bool = False,
    ) -> np.ndarray:
        """
        Reads a window slice of shape (C, height, width).

        Args:
            window: WindowSpec defining the bounding box.
            normalize: If True, applies percentile / radiometric normalization to [0, 1].

        Returns:
            NumPy array of shape (C, height, width) with dtype float32.
        """
        if self.use_rasterio and rasterio is not None:
            data = self._read_rasterio_window(window)
        else:
            data = self._read_numpy_window(window)

        if normalize:
            data = self._normalize_array(data)

        return data.astype(np.float32)

    def _read_rasterio_window(self, window: WindowSpec) -> np.ndarray:
        """Reads window via rasterio with boundless edge padding."""
        with rasterio.open(self.file_path) as src:  # type: ignore
            rst_window = Window(
                col_off=window.col_off,
                row_off=window.row_off,
                width=window.width,
                height=window.height,
            )
            indexes = (
                self.band_indices
                if self.band_indices is not None
                else list(range(1, src.count + 1))
            )
            data = src.read(
                indexes=indexes,
                window=rst_window,
                boundless=True,
                fill_value=0.0,
            )
            return data

    def _read_numpy_window(self, window: WindowSpec) -> np.ndarray:
        """Reads window from NumPy array with boundless edge padding."""
        if self.is_array:
            arr = self.source  # type: ignore
        elif self.file_path and self.file_path.endswith(".npy"):
            arr = np.load(self.file_path, mmap_mode="r")
        else:
            raise FileNotFoundError(
                f"Cannot read raster without rasterio from non-numpy file: {self.file_path}"
            )

        # Ensure (C, H, W)
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]

        C, H, W = arr.shape
        r0, c0 = window.row_off, window.col_off
        h, w = window.height, window.width

        # Boundless window slicing
        out = np.zeros((C, h, w), dtype=arr.dtype)

        src_r0 = max(0, r0)
        src_r1 = min(H, r0 + h)
        src_c0 = max(0, c0)
        src_c1 = min(W, c0 + w)

        dst_r0 = max(0, -r0)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, -c0)
        dst_c1 = dst_c0 + (src_c1 - src_c0)

        if src_r1 > src_r0 and src_c1 > src_c0:
            if self.band_indices is not None:
                # 0-indexed or 1-indexed conversion
                idx = [
                    i if i < C else (i - 1)
                    for i in self.band_indices
                    if (0 <= i < C or 0 <= (i - 1) < C)
                ]
                out[:, dst_r0:dst_r1, dst_c0:dst_c1] = arr[
                    idx, src_r0:src_r1, src_c0:src_c1
                ]
            else:
                out[:, dst_r0:dst_r1, dst_c0:dst_c1] = arr[
                    :, src_r0:src_r1, src_c0:src_c1
                ]

        return out

    @staticmethod
    def _normalize_array(img: np.ndarray) -> np.ndarray:
        """Applies 1st/99th percentile normalization per band to [0.0, 1.0]."""
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


class COGStreamer:
    """
    Multi-source synchronized streamer reading LR, HR, Mask, and DEM windows simultaneously.
    """

    def __init__(
        self,
        lr_source: str | np.ndarray,
        hr_source: str | np.ndarray | None = None,
        mask_source: str | np.ndarray | None = None,
        dem_source: str | np.ndarray | None = None,
        scale_factor: int = 4,
    ):
        self.scale_factor = scale_factor
        self.lr_reader = COGWindowReader(lr_source)
        self.hr_reader = (
            COGWindowReader(hr_source) if hr_source is not None else None
        )
        self.mask_reader = (
            COGWindowReader(mask_source) if mask_source is not None else None
        )
        self.dem_reader = (
            COGWindowReader(dem_source) if dem_source is not None else None
        )

    def read_synchronized_window(
        self,
        lr_window: WindowSpec,
        normalize: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Reads aligned windows across all available input sources.

        Args:
            lr_window: WindowSpec defined on the 10m LR grid.
            normalize: Whether to normalize radiometric bands to [0, 1].

        Returns:
            Dict containing 'lr', 'hr' (optional), 'mask', 'dem'.
        """
        lr_data = self.lr_reader.read_window(lr_window, normalize=normalize)

        # 1. Cloud Mask
        if self.mask_reader is not None:
            mask_data = self.mask_reader.read_window(lr_window, normalize=False)
            mask_data = np.clip(mask_data, 0.0, 1.0)
        else:
            mask_data = np.ones((1, lr_window.height, lr_window.width), dtype=np.float32)

        # 2. CartoDEM (Elevation & Slope)
        if self.dem_reader is not None:
            dem_data = self.dem_reader.read_window(lr_window, normalize=False)
        else:
            dem_data = np.zeros(
                (2, lr_window.height, lr_window.width), dtype=np.float32
            )

        # 3. High-Resolution Reference (s=4 scaled window)
        hr_data: np.ndarray | None = None
        if self.hr_reader is not None:
            s = self.scale_factor
            hr_window = WindowSpec(
                row_off=lr_window.row_off * s,
                col_off=lr_window.col_off * s,
                height=lr_window.height * s,
                width=lr_window.width * s,
            )
            hr_data = self.hr_reader.read_window(hr_window, normalize=normalize)

        result = {
            "lr": lr_data,
            "mask": mask_data,
            "dem": dem_data,
        }
        if hr_data is not None:
            result["hr"] = hr_data

        return result


class COGStreamingDataset(Dataset):
    """
    PyTorch Dataset enabling zero-memory-exhaustion streaming over Cloud-Optimized GeoTIFFs.
    """

    def __init__(
        self,
        streamer: COGStreamer,
        windows: list[WindowSpec],
        is_train: bool = True,
        augment: bool = False,
    ):
        self.streamer = streamer
        self.windows = windows
        self.is_train = is_train
        self.augment = augment

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        win = self.windows[idx]
        data = self.streamer.read_synchronized_window(win, normalize=True)

        lr = data["lr"]
        mask = data["mask"]
        dem = data["dem"]
        hr = data.get("hr", None)

        if self.augment and self.is_train:
            lr, hr, mask, dem = self._apply_augmentations(lr, hr, mask, dem)

        out_item: dict[str, torch.Tensor] = {
            "lr": torch.from_numpy(lr),
            "mask": torch.from_numpy(mask),
            "dem": torch.from_numpy(dem),
            "window_coords": torch.tensor(
                [win.row_off, win.col_off, win.height, win.width],
                dtype=torch.long,
            ),
        }
        if hr is not None:
            out_item["hr"] = torch.from_numpy(hr)

        return out_item

    @staticmethod
    def _apply_augmentations(
        lr: np.ndarray,
        hr: np.ndarray | None,
        mask: np.ndarray,
        dem: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
        """Applies random spatial flips and rotations consistently across all streams."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            lr = np.flip(lr, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
            dem = np.flip(dem, axis=-1).copy()
            if hr is not None:
                hr = np.flip(hr, axis=-1).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            lr = np.flip(lr, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()
            dem = np.flip(dem, axis=-2).copy()
            if hr is not None:
                hr = np.flip(hr, axis=-2).copy()

        return lr, hr, mask, dem
