"""
Unit and Integration Tests for Baseline and Ablation Models
"""

import torch

from src.models.baselines import (
    A0_BaseNet,
    A1_ContextNet,
    BicubicSR,
    EDSRBaseline,
    SRResNetBaseline,
)


def test_bicubic_sr_10band():
    model = BicubicSR(in_bands=10, out_bands=4, scale_factor=4)
    x = torch.rand(2, 10, 16, 16)
    out = model(x)

    assert "sr_image" in out
    sr = out["sr_image"]
    assert sr.shape == (2, 4, 64, 64)
    assert torch.all(sr >= 0.0) and torch.all(sr <= 1.0)


def test_bicubic_sr_custom_indices():
    model = BicubicSR(
        in_bands=10, out_bands=4, scale_factor=4, band_indices=[3, 2, 1, 7]
    )
    x = torch.rand(1, 10, 8, 8)
    out = model(x)
    assert out["sr_image"].shape == (1, 4, 32, 32)


def test_edsr_baseline():
    model = EDSRBaseline(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        n_feats=32,
        n_resblocks=4,
    )
    model.eval()

    x = torch.rand(2, 10, 16, 16)
    with torch.no_grad():
        out = model(x)

    sr = out["sr_image"]
    assert sr.shape == (2, 4, 64, 64)
    assert torch.all(sr >= 0.0) and torch.all(sr <= 1.0)


def test_edsr_backward():
    model = EDSRBaseline(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        n_feats=16,
        n_resblocks=2,
    )
    model.train()

    x = torch.rand(1, 10, 8, 8, requires_grad=True)
    out = model(x)
    loss = out["sr_image"].sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_srresnet_baseline():
    model = SRResNetBaseline(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        n_feats=32,
        n_resblocks=4,
    )
    model.eval()

    x = torch.rand(2, 10, 16, 16)
    with torch.no_grad():
        out = model(x)

    sr = out["sr_image"]
    assert sr.shape == (2, 4, 64, 64)
    assert torch.all(sr >= 0.0) and torch.all(sr <= 1.0)


def test_srresnet_backward():
    model = SRResNetBaseline(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        n_feats=16,
        n_resblocks=2,
    )
    model.train()

    x = torch.rand(1, 10, 8, 8, requires_grad=True)
    out = model(x)
    loss = out["sr_image"].sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_a0_basenet():
    model = A0_BaseNet(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        dilation_rates=[1, 2],
        use_window_attention=True,
    )
    model.eval()

    x = torch.rand(2, 10, 16, 16)
    mask = torch.ones(2, 1, 16, 16)
    with torch.no_grad():
        out = model(x, validity_mask=mask)

    sr = out["sr_image"]
    assert sr.shape == (2, 4, 64, 64)
    assert torch.all(sr >= 0.0) and torch.all(sr <= 1.0)


def test_a0_basenet_backward():
    model = A0_BaseNet(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=16,
        dilation_rates=[1],
        use_window_attention=False,
    )
    model.train()

    x = torch.rand(1, 10, 8, 8, requires_grad=True)
    out = model(x)
    loss = out["sr_image"].sum()
    loss.backward()

    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_a1_contextnet():
    model = A1_ContextNet(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=32,
        context_channels=16,
        dilation_rates=[1, 2],
    )
    model.eval()

    x = torch.rand(2, 10, 16, 16)
    dem = torch.rand(2, 2, 16, 16)
    mask = torch.ones(2, 1, 16, 16)

    with torch.no_grad():
        out = model(x, validity_mask=mask, context_dem=dem)

    sr = out["sr_image"]
    assert sr.shape == (2, 4, 64, 64)
    assert "fused_features" in out
    assert torch.all(sr >= 0.0) and torch.all(sr <= 1.0)


def test_a1_contextnet_backward():
    model = A1_ContextNet(
        in_spectral_bands=10,
        out_sr_bands=4,
        scale_factor=4,
        base_channels=16,
        context_channels=8,
        dilation_rates=[1],
        use_window_attention=False,
    )
    model.train()

    x = torch.rand(1, 10, 8, 8, requires_grad=True)
    dem = torch.rand(1, 2, 8, 8, requires_grad=True)
    out = model(x, context_dem=dem)

    loss = out["sr_image"].sum()
    loss.backward()

    assert x.grad is not None
    assert dem.grad is not None
    assert not torch.isnan(x.grad).any()
    assert not torch.isnan(dem.grad).any()
