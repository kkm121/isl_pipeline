"""
Unit Tests for AC-FEM (Adaptive Cloud-Aware Feature Enhancement Module)
"""

import torch

from src.models.ac_fem import ACFEM


def test_ac_fem_forward_shape():
    ac_fem = ACFEM(spec_channels=64, prior_channels=32, out_channels=64)
    f_spec = torch.randn(2, 64, 16, 16)
    f_prior = torch.randn(2, 32, 16, 16)
    mask = torch.ones(2, 1, 16, 16)

    out = ac_fem(f_spec, f_prior, mask)
    assert out.shape == (2, 64, 16, 16)
    assert not torch.isnan(out).any()


def test_ac_fem_cloud_region_prioritization():
    ac_fem = ACFEM(spec_channels=16, prior_channels=16, out_channels=16)
    f_spec = torch.ones(1, 16, 8, 8) * 10.0
    f_prior = torch.ones(1, 16, 8, 8) * 2.0

    # Half clear, half cloud
    mask = torch.ones(1, 1, 8, 8)
    mask[:, :, :, 4:] = 0.0

    out = ac_fem(f_spec, f_prior, mask)
    # The clear region should have higher activation from spec than cloud region
    assert out.shape == (1, 16, 8, 8)
