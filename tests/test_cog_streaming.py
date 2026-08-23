"""
Unit and Integration Tests for COG Streaming Pipeline & Windowed Readers
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.cog_streaming import (
    COGStreamer,
    COGStreamingDataset,
    COGWindowReader,
    WindowSpec,
    generate_tile_grid,
)


def test_generate_tile_grid():
    # 512x512 scene with 256x256 tiles and 32px overlap -> stride = 224
    windows = generate_tile_grid(
        scene_height=512, scene_width=512, tile_size=256, overlap=32
    )

    assert len(windows) > 0
    for w in windows:
        assert isinstance(w, WindowSpec)
        assert w.height <= 256
        assert w.width <= 256
        assert w.row_off >= 0
        assert w.col_off >= 0


def test_cog_window_reader_numpy_array():
    dummy_s2 = np.random.rand(10, 512, 512).astype(np.float32)
    reader = COGWindowReader(dummy_s2)

    # 1. Interior window
    win_interior = WindowSpec(row_off=64, col_off=64, height=128, width=128)
    tile = reader.read_window(win_interior, normalize=True)
    assert tile.shape == (10, 128, 128)
    assert np.all(tile >= 0.0) and np.all(tile <= 1.0)

    # 2. Boundless edge window extending beyond scene boundary
    win_edge = WindowSpec(row_off=450, col_off=450, height=128, width=128)
    tile_edge = reader.read_window(win_edge, normalize=False)
    assert tile_edge.shape == (10, 128, 128)


def test_cog_streamer_synchronized_reads():
    lr_arr = np.random.rand(10, 128, 128).astype(np.float32)
    hr_arr = np.random.rand(4, 512, 512).astype(np.float32)
    mask_arr = np.ones((1, 128, 128), dtype=np.float32)
    dem_arr = np.random.rand(2, 128, 128).astype(np.float32)

    streamer = COGStreamer(
        lr_source=lr_arr,
        hr_source=hr_arr,
        mask_source=mask_arr,
        dem_source=dem_arr,
        scale_factor=4,
    )

    lr_win = WindowSpec(row_off=16, col_off=16, height=64, width=64)
    data = streamer.read_synchronized_window(lr_win, normalize=True)

    assert "lr" in data and data["lr"].shape == (10, 64, 64)
    assert "mask" in data and data["mask"].shape == (1, 64, 64)
    assert "dem" in data and data["dem"].shape == (2, 64, 64)
    assert "hr" in data and data["hr"].shape == (4, 256, 256)  # 4x scale


def test_cog_streaming_dataset_and_dataloader():
    lr_arr = np.random.rand(10, 256, 256).astype(np.float32)
    hr_arr = np.random.rand(4, 1024, 1024).astype(np.float32)
    mask_arr = np.ones((1, 256, 256), dtype=np.float32)
    dem_arr = np.random.rand(2, 256, 256).astype(np.float32)

    streamer = COGStreamer(
        lr_source=lr_arr,
        hr_source=hr_arr,
        mask_source=mask_arr,
        dem_source=dem_arr,
        scale_factor=4,
    )

    windows = generate_tile_grid(
        scene_height=256, scene_width=256, tile_size=64, overlap=16
    )
    dataset = COGStreamingDataset(
        streamer=streamer,
        windows=windows,
        is_train=True,
        augment=True,
    )

    assert len(dataset) == len(windows)
    item = dataset[0]

    assert isinstance(item["lr"], torch.Tensor) and item["lr"].shape == (10, 64, 64)
    assert isinstance(item["hr"], torch.Tensor) and item["hr"].shape == (4, 256, 256)
    assert isinstance(item["mask"], torch.Tensor) and item["mask"].shape == (1, 64, 64)
    assert isinstance(item["dem"], torch.Tensor) and item["dem"].shape == (2, 64, 64)

    # Test PyTorch DataLoader integration with batching
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))

    assert batch["lr"].shape == (2, 10, 64, 64)
    assert batch["hr"].shape == (2, 4, 256, 256)
    assert batch["mask"].shape == (2, 1, 64, 64)
    assert batch["dem"].shape == (2, 2, 64, 64)
