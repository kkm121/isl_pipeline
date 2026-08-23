"""
=============================================================================
BharatSRM-Net v4: Cross-Sensor Registration & Radiometric Harmonization
=============================================================================
Section 5.2 Specification:
  1. Feature-based co-registration: AKAZE keypoint matching + RANSAC homography.
  2. Sub-pixel alignment: Rejects tile pairs with residual registration error > 1 LR pixel (~10m).
  3. Radiometric harmonization: Band-wise histogram matching to Sentinel-2 reflectance distribution.
=============================================================================
"""


import numpy as np
from skimage.exposure import match_histograms


def match_radiometry_histogram(
    source_image: np.ndarray, reference_image: np.ndarray
) -> np.ndarray:
    """
    Performs band-wise histogram matching of source (e.g. HR reference)
    to reference (Sentinel-2) reflectance conventions.

    Args:
        source_image: (C, H, W) or (H, W, C) Source multi-band image
        reference_image: (C, H_ref, W_ref) Reference multi-band image

    Returns:
        harmonized_image: Radiometrically matched array with same shape as source_image.
    """
    is_chw = source_image.ndim == 3 and source_image.shape[0] in [3, 4, 10]
    if is_chw:
        src = np.transpose(source_image, (1, 2, 0))
        ref = np.transpose(reference_image, (1, 2, 0))
    else:
        src = source_image
        ref = reference_image

    num_channels = min(src.shape[-1], ref.shape[-1])
    matched = np.zeros_like(src)

    for c in range(num_channels):
        matched[..., c] = match_histograms(src[..., c], ref[..., c])

    if is_chw:
        return np.transpose(matched, (2, 0, 1))
    return matched


def evaluate_registration_shift(
    img1: np.ndarray, img2: np.ndarray
) -> tuple[float, float, float]:
    """
    Estimates translation shift (dx, dy) and normalized cross-correlation
    between two single-band aligned rasters using phase correlation.
    """
    f1 = np.fft.fft2(img1)
    f2 = np.fft.fft2(img2)
    cross_power = (f1 * np.conj(f2)) / (np.abs(f1 * np.conj(f2)) + 1e-10)
    correlation = np.fft.ifft2(cross_power)
    corr_real = np.real(correlation)

    y_shift, x_shift = np.unravel_index(np.argmax(corr_real), corr_real.shape)
    H, W = img1.shape
    if y_shift > H // 2:
        y_shift -= H
    if x_shift > W // 2:
        x_shift -= W

    max_corr = float(np.max(corr_real))
    return float(x_shift), float(y_shift), max_corr
