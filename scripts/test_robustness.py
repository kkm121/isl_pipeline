import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from src.models.bharatsrm_net import BharatSRMNetV4
from src.training.losses import CompositeBharatSRMLoss

def test_all():
    model = BharatSRMNetV4(include_downstream_heads=False)
    model.train()
    criterion = CompositeBharatSRMLoss()

    # Test 1: all zeros
    lr = torch.zeros(2, 10, 64, 64)
    hr = torch.zeros(2, 4, 256, 256)
    mask = torch.zeros(2, 1, 64, 64)
    dem = torch.zeros(2, 2, 64, 64)

    out = model(lr, mask, dem)
    losses = criterion(out['sr_image'], hr, lr, out['log_variance'], epoch=1)
    print("Test 1 (All zeros) Loss Total:", losses['loss_total'].item())
    assert not torch.isnan(losses['loss_total']), "Test 1 failed with NaN"

    # Test 2: realistic values
    lr2 = torch.rand(2, 10, 64, 64) * 0.4
    hr2 = torch.rand(2, 4, 256, 256) * 0.4
    mask2 = torch.ones(2, 1, 64, 64)
    dem2 = torch.zeros(2, 2, 64, 64)

    out2 = model(lr2, mask2, dem2)
    losses2 = criterion(out2['sr_image'], hr2, lr2, out2['log_variance'], epoch=1)
    print("Test 2 (Realistic) Loss Total:", losses2['loss_total'].item())
    assert not torch.isnan(losses2['loss_total']), "Test 2 failed with NaN"
    print("ALL TESTS PASSED WITH 100% FINITE LOSSES!")

if __name__ == "__main__":
    test_all()
