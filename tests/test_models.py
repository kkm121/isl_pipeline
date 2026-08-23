"""
Unit and Integration Tests for BharatSRMNetV4 Architecture
"""

import torch
import pytest
import warnings

from src.models.bharatsrm_net import BharatSRMNetV4


def test_bharatsrm_net_forward():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        context_channels=16,
        use_context_stream=True,
        include_downstream_heads=True,
    )
    model.eval()

    lr = torch.randn(2, 10, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    dem = torch.randn(2, 2, 16, 16)

    with torch.no_grad():
        out = model(lr, mask, dem)

    assert out["sr_image"].shape == (2, 4, 64, 64)
    assert out["log_variance"].shape == (2, 4, 64, 64)
    assert out["variance"].shape == (2, 4, 64, 64)


def test_dem_warning_when_missing():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        use_context_stream=True,
    )
    model.eval()

    lr = torch.randn(2, 10, 16, 16)
    mask = torch.ones(2, 1, 16, 16)

    with pytest.warns(UserWarning):
        out = model(lr, mask, None)


def test_output_range_bounded():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        use_context_stream=False,
    )
    model.eval()

    lr = torch.randn(2, 10, 16, 16) * 100 # Large inputs
    
    with torch.no_grad():
        out = model(lr)
        
    sr_img = out["sr_image"]
    # Check bounds (assuming sigmoid or clipping applied at end)
    assert torch.all(sr_img >= 0.0) and torch.all(sr_img <= 1.0)


def test_uncertainty_clamping():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        use_context_stream=False,
    )
    model.eval()

    lr = torch.randn(2, 10, 16, 16) * 1000 # extreme values to push variance out
    
    with torch.no_grad():
        out = model(lr)
        
    log_var = out["log_variance"]
    assert torch.all(log_var >= -8.0) and torch.all(log_var <= 5.0)


def test_gradient_flow_all_params():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        use_context_stream=False,
        include_downstream_heads=True,
    )
    model.train()

    lr = torch.randn(2, 10, 16, 16)
    
    # zero gradients
    model.zero_grad()
    
    out = model(lr)
    loss = out["sr_image"].mean() + out["log_variance"].mean()
    if "features_hr" in out:
        loss += model.predict_downstream_road(out["features_hr"], out["sr_image"]).mean()
        loss += model.predict_downstream_lulc(out["features_hr"], out["sr_image"]).mean()
        
        # also call change head to ensure its parameters get gradients
        loss += model.predict_downstream_change(
            out["sr_image"], out["sr_image"], 
            out["features_hr"], out["features_hr"], 
            out["log_variance"], out["log_variance"]
        ).mean()
        
    loss.backward()

    # Check all params have non-zero grad
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} has no grad"
        assert not torch.all(param.grad == 0), f"Parameter {name} has zero grad"
