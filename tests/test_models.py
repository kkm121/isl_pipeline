"""
Unit and Integration Tests for BharatSRMNetV4 Architecture
"""

import torch

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

    # 4x resolution upscale: (16, 16) -> (64, 64)
    assert out["sr_image"].shape == (2, 4, 64, 64)
    assert out["log_variance"].shape == (2, 4, 64, 64)
    assert out["variance"].shape == (2, 4, 64, 64)
    assert torch.all(out["sr_image"] >= 0.0) and torch.all(out["sr_image"] <= 1.0)
    assert torch.all(out["variance"] > 0.0)


def test_downstream_heads_forward():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        context_channels=16,
        include_downstream_heads=True,
    )
    model.eval()

    lr = torch.randn(1, 10, 16, 16)
    with torch.no_grad():
        out = model(lr)
        feat_hr = out["features_hr"]
        sr_img = out["sr_image"]

        # Road head: (B, 1, 64, 64)
        roads = model.predict_downstream_road(feat_hr, sr_img)
        assert roads.shape == (1, 1, 64, 64)
        assert torch.all(roads >= 0.0) and torch.all(roads <= 1.0)

        # LULC head: (B, 5, 64, 64)
        lulc = model.predict_downstream_lulc(feat_hr, sr_img)
        assert lulc.shape == (1, 5, 64, 64)


def test_backward_gradient_flow():
    model = BharatSRMNetV4(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
    )
    model.train()

    lr = torch.randn(1, 10, 16, 16, requires_grad=True)
    mask = torch.ones(1, 1, 16, 16)
    out = model(lr, mask)

    loss = out["sr_image"].sum() + out["log_variance"].sum()
    loss.backward()

    assert lr.grad is not None
    assert not torch.isnan(lr.grad).any()
