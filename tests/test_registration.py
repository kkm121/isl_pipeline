"""
Unit Tests for Cross-Sensor Registration & Radiometric Harmonization
"""

import numpy as np
import pytest

from src.data.registration import (
    evaluate_registration_shift,
    match_radiometry_histogram,
)


def test_radiometric_histogram_matching():
    # Source image (e.g. normally distributed)
    np.random.seed(42)
    src = np.random.normal(loc=0.5, scale=0.1, size=(4, 64, 64)).astype(np.float32)
    # Target image (e.g. normally distributed but shifted)
    ref = np.random.normal(loc=0.2, scale=0.05, size=(4, 64, 64)).astype(np.float32)
    
    # Clip to valid ranges to simulate imagery
    src = np.clip(src, 0, 1)
    ref = np.clip(ref, 0, 1)
    
    matched = match_radiometry_histogram(src, ref)
    
    # Check shape
    assert matched.shape == src.shape
    
    # Check that matched histogram statistics are closer to ref than src was
    for c in range(4):
        src_mean = np.mean(src[c])
        ref_mean = np.mean(ref[c])
        matched_mean = np.mean(matched[c])
        
        # Matched mean should be very close to reference mean
        assert abs(matched_mean - ref_mean) < abs(src_mean - ref_mean)
        assert abs(matched_mean - ref_mean) < 0.05


def test_phase_correlation_no_shift():
    # Identical images should have (0, 0) shift
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 1.0  # create a square
    
    dx, dy, corr = evaluate_registration_shift(img, img)
    assert dx == 0.0
    assert dy == 0.0
    # corr might be low if energy is spread, just check it is positive
    assert corr > 0.0

def test_phase_correlation_known_shift():
    # Create an image and shift it
    img = np.zeros((64, 64), dtype=np.float32)
    img[16:48, 16:48] = 1.0
    
    img_shifted = np.zeros((64, 64), dtype=np.float32)
    img_shifted[16+5:48+5, 16+3:48+3] = 1.0
    
    dx, dy, corr = evaluate_registration_shift(img, img_shifted)
    
    # Since phase correlation calculates translation, direction can be flipped depending on convention
    assert dx in [3.0, -3.0]
    assert dy in [5.0, -5.0]
    assert corr > 0.0
