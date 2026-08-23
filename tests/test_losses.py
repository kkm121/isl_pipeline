"""
Unit Tests for Composite Loss Functions in BharatSRM-Net v4
"""

import pytest
import torch

from src.training.losses import (
    CharbonnierLoss,
    CompositeBharatSRMLoss,
    DegradationConsistencyLoss,
    HeteroscedasticUncertaintyLoss,
    SpectralAngleMapperLoss,
)


def test_charbonnier_loss():
    loss_fn = CharbonnierLoss(eps=1e-3)
    p = torch.ones(2, 4, 16, 16)
    t = torch.ones(2, 4, 16, 16)
    assert loss_fn(p, t).item() < 1e-2


def test_spectral_angle_mapper_loss():
    loss_fn = SpectralAngleMapperLoss()
    p = torch.tensor([[[[1.0]], [[0.0]], [[0.0]], [[0.0]]]])
    t = torch.tensor([[[[0.0]], [[1.0]], [[0.0]], [[0.0]]]])
    # 90 degrees = pi / 2 = ~1.57 radians
    loss = loss_fn(p, t)
    assert abs(loss.item() - 1.5707) < 1e-2


def test_degradation_consistency_loss():
    loss_fn = DegradationConsistencyLoss(num_bands=4, scale_factor=4)
    sr = torch.rand(1, 4, 64, 64)
    lr = torch.rand(1, 4, 16, 16)
    loss = loss_fn(sr, lr)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)


def test_heteroscedastic_uncertainty_loss():
    loss_fn = HeteroscedasticUncertaintyLoss()
    p = torch.ones(1, 4, 8, 8)
    t = torch.zeros(1, 4, 8, 8)
    log_var = torch.zeros(1, 4, 8, 8)  # s=0 -> sigma^2=1
    loss = loss_fn(p, t, log_var)
    assert loss.item() > 0.0


def test_composite_loss_warmup():
    loss_fn = CompositeBharatSRMLoss()
    sr = torch.rand(1, 4, 32, 32)
    hr = torch.rand(1, 4, 32, 32)
    lr = torch.rand(1, 4, 8, 8)
    log_var = torch.zeros(1, 4, 32, 32)

    # Epoch 1 (Warmup active)
    out1 = loss_fn(sr, hr, lr, log_var, epoch=1, warmup_epochs=3)
    assert out1["warmup_factor"].item() == pytest.approx(1.0 / 3.0)

    # Epoch 3 (Warmup complete)
    out3 = loss_fn(sr, hr, lr, log_var, epoch=3, warmup_epochs=3)
    assert out3["warmup_factor"].item() == pytest.approx(1.0)
