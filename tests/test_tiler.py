"""
Unit Tests for Overlapping Tiled Inference Engine
"""

import pytest
import torch

from src.inference.tiler import TiledInferenceEngine, create_2d_hanning_window
from src.models.bharatsrm_net import BharatSRMNetV4


def test_hanning_window_center_exact():
    win = create_2d_hanning_window(64, 64)
    win_odd = create_2d_hanning_window(65, 65)
    assert win_odd[32, 32] == pytest.approx(1.0, abs=1e-2)
    assert win.max() == pytest.approx(1.0, abs=1e-2)

def test_hanning_window_corners_zero():
    win = create_2d_hanning_window(64, 64)
    assert win[0, 0] == pytest.approx(0.0, abs=1e-2)
    assert win[0, -1] == pytest.approx(0.0, abs=1e-2)
    assert win[-1, 0] == pytest.approx(0.0, abs=1e-2)
    assert win[-1, -1] == pytest.approx(0.0, abs=1e-2)


def test_constant_image_preserved():
    # Pass a constant image, ensure output is constant.
    # Note: BharatSRMNetV4 might not output a constant image even if input is constant,
    # so we mock the model to simply return the input (or upscaled input).
    
    class MockModel(torch.nn.Module):
        def __init__(self, scale_factor=4):
            super().__init__()
            self.scale_factor = scale_factor
            
        def forward(self, lr, *args, **kwargs):
            # Just interpolate to scale factor
            B, C, H, W = lr.shape
            sr = torch.nn.functional.interpolate(lr, scale_factor=self.scale_factor, mode='nearest')
            return {"sr_image": sr, "variance": sr.clone()}
            
    engine = TiledInferenceEngine(tile_size=32, overlap=8, scale_factor=4, device="cpu")
    model = MockModel(scale_factor=4)
    
    constant_val = 0.5
    scene_lr = torch.full((1, 4, 64, 64), constant_val)
    out = engine.predict_large_scene(model, scene_lr)
    
    sr_out = out["sr_image"]
    var_out = out["variance"]
    
    # Exclude edges since they don't have overlapping tiles to sum to 1.0
    # In tiled inference with hanning window, edges might drop off if not padded correctly.
    # But for a strictly normalizing engine, the center should be exactly constant.
    sr_center = sr_out[..., 32:-32, 32:-32]
    var_center = var_out[..., 32:-32, 32:-32]
    
    assert torch.allclose(sr_center, torch.tensor(constant_val), atol=1e-4)
    assert torch.allclose(var_center, torch.tensor(constant_val), atol=1e-4)
