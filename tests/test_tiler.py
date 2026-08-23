"""
Unit Tests for Overlapping Tiled Inference Engine
"""

import pytest
import torch

from src.inference.tiler import TiledInferenceEngine, create_2d_hanning_window
from src.models.bharatsrm_net import BharatSRMNetV4


def test_2d_hanning_window():
    win = create_2d_hanning_window(64, 64)
    assert win.shape == (64, 64)
    # Center should be near 1.0
    assert win[32, 32] == pytest.approx(1.0, abs=0.05)
    # Edges should be small
    assert win[0, 0] < 0.1


def test_tiled_inference_large_scene():
    engine = TiledInferenceEngine(tile_size=32, overlap=8, scale_factor=4, device="cpu")
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=16,
        context_channels=8,
        include_downstream_heads=False,
    )
    model.eval()

    scene_lr = torch.rand(1, 10, 64, 64)
    out = engine.predict_large_scene(model, scene_lr)

    # 4x scale factor: 64x64 -> 256x256
    assert out["sr_image"].shape == (1, 4, 256, 256)
    assert out["variance"].shape == (1, 4, 256, 256)
    assert not torch.isnan(out["sr_image"]).any()
