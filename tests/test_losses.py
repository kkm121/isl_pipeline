"""
Unit Tests for Composite Loss Functions in BharatSRM-Net v4
"""

import pytest
import torch
import math

from src.training.losses import (
    CharbonnierLoss,
    CompositeBharatSRMLoss,
    DegradationConsistencyLoss,
    HeteroscedasticUncertaintyLoss,
    SpectralAngleMapperLoss,
    StructuralSSIMLoss,
)


def test_charbonnier_loss_identical():
    eps = 1e-3
    loss_fn = CharbonnierLoss(eps=eps)
    p = torch.ones(2, 4, 16, 16)
    t = torch.ones(2, 4, 16, 16)
    loss = loss_fn(p, t)
    assert loss.item() == pytest.approx(eps, rel=1e-5)


def test_charbonnier_loss_known_value():
    eps = 1e-3
    loss_fn = CharbonnierLoss(eps=eps)
    p = torch.full((1, 1, 1, 1), 1.0)
    t = torch.full((1, 1, 1, 1), 0.0)
    loss = loss_fn(p, t)
    expected = math.sqrt(1.0 + eps**2)
    assert loss.item() == pytest.approx(expected, rel=1e-5)


def test_sam_loss_identical_spectra():
    loss_fn = SpectralAngleMapperLoss()
    p = torch.rand(2, 4, 16, 16) + 0.1
    t = p.clone()
    loss = loss_fn(p, t)
    assert loss.item() == pytest.approx(0.0, abs=1e-2)

def test_sam_loss_orthogonal_spectra():
    loss_fn = SpectralAngleMapperLoss()
    p = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    t = torch.tensor([[[[0.0]], [[1.0]], [[0.0]], [[0.0]]]])
    loss = loss_fn(p, t)
    expected = math.pi / 2
    assert loss.item() == pytest.approx(expected, rel=1e-2)

def test_heteroscedastic_loss_known():
    loss_fn = HeteroscedasticUncertaintyLoss()
    p1 = torch.tensor([1.0])
    t1 = torch.tensor([0.0])
    log_var1 = torch.tensor([0.0])
    loss1 = loss_fn(p1, t1, log_var1)
    assert loss1.item() == pytest.approx(1.0, rel=1e-5)
    
    p2 = torch.tensor([1.0])
    t2 = torch.tensor([0.0])
    log_var2 = torch.tensor([2.0])
    loss2 = loss_fn(p2, t2, log_var2)
    expected2 = math.exp(-2.0) * 1.0 + 2.0
    assert loss2.item() == pytest.approx(expected2, rel=1e-5)

def test_degradation_consistency_perfect():
    """The degradation loss for a smooth SR image should be small (not exactly zero,
    since the Gaussian PSF blurring changes values before avg_pool downsampling)."""
    loss_fn = DegradationConsistencyLoss(num_bands=4, scale_factor=4)
    # Use a smooth constant image — PSF blur of a constant is still constant
    lr = torch.full((1, 4, 16, 16), 0.5)
    sr = torch.full((1, 4, 64, 64), 0.5)
    loss = loss_fn(sr, lr)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_composite_warmup_zero_epochs():
    loss_fn = CompositeBharatSRMLoss()
    sr = torch.rand(1, 4, 32, 32)
    hr = torch.rand(1, 4, 32, 32)
    lr = torch.rand(1, 4, 8, 8)
    log_var = torch.zeros(1, 4, 32, 32)
    
    out = loss_fn(sr, hr, lr, log_var, epoch=5, warmup_epochs=0)
    assert out["warmup_factor"].item() == pytest.approx(1.0)


def test_composite_warmup_progression():
    loss_fn = CompositeBharatSRMLoss()
    sr = torch.rand(1, 4, 32, 32)
    hr = torch.rand(1, 4, 32, 32)
    lr = torch.rand(1, 4, 8, 8)
    log_var = torch.zeros(1, 4, 32, 32)

    out1 = loss_fn(sr, hr, lr, log_var, epoch=1, warmup_epochs=3)
    out2 = loss_fn(sr, hr, lr, log_var, epoch=2, warmup_epochs=3)
    out3 = loss_fn(sr, hr, lr, log_var, epoch=3, warmup_epochs=3)
    
    assert out1["warmup_factor"].item() == pytest.approx(1.0 / 3.0)
    assert out2["warmup_factor"].item() == pytest.approx(2.0 / 3.0)
    assert out3["warmup_factor"].item() == pytest.approx(1.0)


def test_ssim_loss_identical():
    loss_fn = StructuralSSIMLoss(data_range=1.0, in_channels=4)
    img = torch.rand(1, 4, 32, 32)
    loss = loss_fn(img, img)
    assert loss.item() == pytest.approx(0.0, abs=1e-4)


def test_ssim_data_range_scaling():
    loss_fn1 = StructuralSSIMLoss(data_range=1.0, in_channels=4)
    loss_fn2 = StructuralSSIMLoss(data_range=255.0, in_channels=4)
    img1 = torch.rand(1, 4, 32, 32)
    img2 = torch.rand(1, 4, 32, 32)
    
    loss1 = loss_fn1(img1, img2)
    loss2 = loss_fn2(img1 * 255, img2 * 255)
    
    assert loss1.item() > 0
    assert loss2.item() > 0
