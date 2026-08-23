"""
=============================================================================
BharatSRM-Net v4: Cross-Sensor Registration & Radiometric Harmonization
=============================================================================
Section 5.2 Specification:
  1. Feature-based co-registration: Phase correlation / Fourier shift estimation.
  2. Sub-pixel alignment: Rejects tile pairs with residual registration error > 1 LR pixel (~10m).
  3. Radiometric harmonization: Band-wise histogram matching to Sentinel-2 reflectance distribution.
=============================================================================
"""

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    from skimage.exposure import match_histograms
except ImportError:

    def match_histograms(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Pure NumPy implementation of 2D histogram matching."""
        orig_shape = image.shape
        img_flat = image.ravel()
        ref_flat = reference.ravel()

        s_values, bin_idx, s_counts = np.unique(
            img_flat, return_inverse=True, return_counts=True
        )
        t_values, t_counts = np.unique(ref_flat, return_counts=True)

        s_quantiles = np.cumsum(s_counts).astype(np.float64) / img_flat.size
        t_quantiles = np.cumsum(t_counts).astype(np.float64) / ref_flat.size

        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        return interp_t_values[bin_idx].reshape(orig_shape).astype(image.dtype)


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


def akaze_keypoint_match(src_gray: np.ndarray, ref_gray: np.ndarray, ratio_threshold: float = 0.75) -> tuple[np.ndarray, np.ndarray, list]:
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for AKAZE keypoint matching.")
    
    if src_gray.dtype != np.uint8:
        src_gray = cv2.normalize(src_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if ref_gray.dtype != np.uint8:
        ref_gray = cv2.normalize(ref_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    akaze = cv2.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(src_gray, None)
    kp2, des2 = akaze.detectAndCompute(ref_gray, None)
    
    if des1 is None or des2 is None or len(kp1) < 2 or len(kp2) < 2:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32), []
        
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.knnMatch(des1, des2, k=2)
    
    good_matches = []
    src_pts = []
    ref_pts = []
    
    for m_n in matches:
        if len(m_n) == 2:
            m, n = m_n
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
                src_pts.append(kp1[m.queryIdx].pt)
                ref_pts.append(kp2[m.trainIdx].pt)
                
    if len(good_matches) > 0:
        src_pts = np.float32(src_pts)
        ref_pts = np.float32(ref_pts)
    else:
        src_pts = np.empty((0, 2), dtype=np.float32)
        ref_pts = np.empty((0, 2), dtype=np.float32)
        
    return src_pts, ref_pts, good_matches

def ransac_homography(src_pts: np.ndarray, ref_pts: np.ndarray, reproj_threshold: float = 3.0) -> tuple[np.ndarray, np.ndarray, float]:
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for RANSAC homography estimation.")
    
    if len(src_pts) < 4:
        return np.eye(3), np.zeros((len(src_pts), 1), dtype=np.uint8), float('inf')
        
    H, mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, reproj_threshold)
    
    if H is None:
        return np.eye(3), np.zeros((len(src_pts), 1), dtype=np.uint8), float('inf')
        
    inlier_mask = mask.ravel() == 1
    if not np.any(inlier_mask):
        return H, mask, float('inf')
        
    src_inliers = src_pts[inlier_mask]
    ref_inliers = ref_pts[inlier_mask]
    
    src_inliers_hom = np.hstack([src_inliers, np.ones((len(src_inliers), 1))])
    proj_pts_hom = (H @ src_inliers_hom.T).T
    proj_pts = proj_pts_hom[:, :2] / (proj_pts_hom[:, 2:] + 1e-8)
    
    errors = np.linalg.norm(proj_pts - ref_inliers, axis=1)
    rms_error = float(np.sqrt(np.mean(errors**2)))
    
    return H, mask, rms_error

def register_image_pair(source: np.ndarray, reference: np.ndarray, max_error_pixels: float = 1.0) -> tuple[np.ndarray, float, bool]:
    if not HAS_CV2:
        raise ImportError("OpenCV (cv2) is required for image registration.")
        
    def get_gray(img):
        if img.ndim == 2:
            return img
        elif img.shape[0] in [3, 4, 10]:
            return np.mean(img[:3], axis=0)
        else:
            return np.mean(img[..., :3], axis=-1)

    src_gray = get_gray(source)
    ref_gray = get_gray(reference)
    
    src_pts, ref_pts, _ = akaze_keypoint_match(src_gray, ref_gray)
    H, mask, rms_error = ransac_homography(src_pts, ref_pts)
    
    is_valid = rms_error <= max_error_pixels
    
    if source.ndim == 2:
        warped = cv2.warpPerspective(source, H, (reference.shape[1], reference.shape[0]))
    elif source.shape[0] in [3, 4, 10]:
        warped = np.zeros_like(source)
        for i in range(source.shape[0]):
            warped[i] = cv2.warpPerspective(source[i], H, (reference.shape[2], reference.shape[1]))
    else:
        warped = cv2.warpPerspective(source, H, (reference.shape[1], reference.shape[0]))
        
    return warped, rms_error, is_valid
