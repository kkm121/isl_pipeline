"""
Unit Tests for PartialConv2d in BharatSRM-Net v4
"""

import torch

from src.models.partial_conv import PartialConv2d


def test_partial_conv_forward_shape():
    pconv = PartialConv2d(in_channels=10, out_channels=64, kernel_size=3, padding=1)
    x = torch.randn(2, 10, 32, 32)
    mask = torch.ones(2, 1, 32, 32)

    out, new_mask = pconv(x, mask)
    assert out.shape == (2, 64, 32, 32)
    assert new_mask.shape == (2, 1, 32, 32)
    assert torch.all(new_mask == 1.0)


def test_partial_conv_cloud_mask_update():
    pconv = PartialConv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(1, 4, 16, 16)
    mask = torch.ones(1, 1, 16, 16)
    # Mask out a 4x4 hole
    mask[:, :, 6:10, 6:10] = 0.0

    out, new_mask = pconv(x, mask)
    # The mask should progressively infill the hole
    assert new_mask.sum() > mask.sum()
    assert not torch.isnan(out).any()


def test_partial_conv_all_masked_zeros():
    pconv = PartialConv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1)
    x = torch.randn(1, 4, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)

    out, new_mask = pconv(x, mask)
    assert torch.all(out == 0.0)
    assert torch.all(new_mask == 0.0)
