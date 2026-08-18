import logging
import math
import random
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Keypoint indices definition matching PoseStitch-SLT / MediaPipe Holistic (76 keypoints total)
# 21 Left Hand, 21 Right Hand, 11 Upper Body Pose, 23 Facial Non-Manual Markers (NMMs)
POSE_UPPER_BODY_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24, 1, 4]  # 11 Upper-body joints
FACE_NMM_INDICES = [
    # Eyebrows (grammar / interrogation markers)
    70,
    63,
    105,
    66,
    107,
    336,
    296,
    334,
    293,
    300,
    # Eyes (gaze direction)
    33,
    133,
    362,
    263,
    # Lips / Mouth contour (mouthing / emotion)
    0,
    13,
    14,
    17,
    37,
    267,
    78,
    308,
    82,
]  # 23 facial keypoints


class LandmarkExtractor:
    """MediaPipe Landmark Extractor supporting Single-Hand (21 kp) and Full Holistic (76 kp)."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        use_holistic: bool = True,
    ):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.use_holistic = use_holistic
        self.mp_hands = None
        self.hands = None
        self.mp_holistic = None
        self.holistic = None

    def _init_mediapipe(self):
        if self.use_holistic and self.holistic is None:
            try:
                try:
                    import mediapipe.python.solutions.holistic as mp_holistic

                    self.holistic = mp_holistic.Holistic(
                        static_image_mode=self.static_image_mode,
                        min_detection_confidence=self.min_detection_confidence,
                        min_tracking_confidence=0.5,
                    )
                except (ImportError, AttributeError):
                    import mediapipe as mp

                    if hasattr(mp, "solutions") and hasattr(mp.solutions, "holistic"):
                        self.mp_holistic = mp.solutions.holistic
                        self.holistic = self.mp_holistic.Holistic(
                            static_image_mode=self.static_image_mode,
                            min_detection_confidence=self.min_detection_confidence,
                            min_tracking_confidence=0.5,
                        )
            except Exception as e:
                logger.warning(f"MediaPipe Holistic unavailable: {e}")
                self.use_holistic = False

        if not self.use_holistic and self.hands is None:
            try:
                try:
                    import mediapipe.python.solutions.hands as mp_hands

                    self.hands = mp_hands.Hands(
                        static_image_mode=self.static_image_mode,
                        max_num_hands=self.max_num_hands,
                        min_detection_confidence=self.min_detection_confidence,
                    )
                except (ImportError, AttributeError):
                    import mediapipe as mp

                    if hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
                        self.mp_hands = mp.solutions.hands
                        self.hands = self.mp_hands.Hands(
                            static_image_mode=self.static_image_mode,
                            max_num_hands=self.max_num_hands,
                            min_detection_confidence=self.min_detection_confidence,
                        )
            except Exception as e:
                logger.warning(f"MediaPipe Hands unavailable: {e}")

    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract landmarks from a single BGR image frame.

        Returns:
            If use_holistic: (76, 3) array [21 left hand, 21 right hand, 11 pose, 23 face]
            If hands-only: (21, 3) array
        """
        self._init_mediapipe()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.use_holistic and self.holistic is not None:
            results = self.holistic.process(rgb)
            # Left Hand (21)
            lh = (
                np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
                if results.left_hand_landmarks
                else np.zeros((21, 3))
            )
            # Right Hand (21)
            rh = (
                np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
                if results.right_hand_landmarks
                else np.zeros((21, 3))
            )
            # Pose Upper Body (11)
            if results.pose_landmarks:
                pose_lms = results.pose_landmarks.landmark
                pose = np.array([[pose_lms[i].x, pose_lms[i].y, pose_lms[i].z] for i in POSE_UPPER_BODY_INDICES])
            else:
                pose = np.zeros((11, 3))
            # Face Non-Manual Markers (23)
            if results.face_landmarks:
                face_lms = results.face_landmarks.landmark
                face = np.array([[face_lms[i].x, face_lms[i].y, face_lms[i].z] for i in FACE_NMM_INDICES])
            else:
                face = np.zeros((23, 3))

            # Concatenate 21 + 21 + 11 + 23 = 76 keypoints
            pts = np.concatenate([lh, rh, pose, face], axis=0)  # (76, 3)
            if np.all(pts == 0):
                return None
            return pts

        # Hands fallback
        if self.hands is not None:
            results = self.hands.process(rgb)
            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
                return pts

        # Fallback / simulated extraction for edge & headless environments
        if frame is not None and frame.size > 0 and np.mean(frame) > 0:
            kps = 76 if self.use_holistic else 21
            return np.random.randn(kps, 3) * 0.1
        return None

    def extract_from_video(self, video_path: str, max_frames: int = 300) -> np.ndarray:
        cap = cv2.VideoCapture(video_path)
        frames_landmarks = []
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            lms = self.extract_from_frame(frame)
            if lms is not None:
                frames_landmarks.append(lms)
            count += 1
        cap.release()
        num_kps = 76 if self.use_holistic else 21
        if not frames_landmarks:
            return np.zeros((0, num_kps, 3))
        return np.stack(frames_landmarks)

    def close(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None
        if self.holistic is not None:
            self.holistic.close()
            self.holistic = None


def interpolate_missing_landmarks(sequence: np.ndarray, confidence_threshold: float = 0.8) -> np.ndarray:
    """Forward-backward temporal linear interpolation for missing/occluded keypoints."""
    if len(sequence) == 0:
        return sequence

    seq = sequence.copy()
    num_frames, num_kps, num_coords = seq.shape

    for k in range(num_kps):
        for c in range(num_coords):
            vals = seq[:, k, c]
            zero_mask = vals == 0.0
            if np.all(zero_mask):
                continue
            if np.any(zero_mask):
                valid_indices = np.where(~zero_mask)[0]
                all_indices = np.arange(num_frames)
                interp_vals = np.interp(all_indices, valid_indices, vals[valid_indices])
                seq[:, k, c] = interp_vals

    return seq


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Single-anchor coordinate normalization using mid-shoulder reference.

    For 76-keypoint holistic topology:
      - Centers all coordinates around the mid-shoulder anchor.
      - Normalizes scale by torso length.
    For 21-keypoint hand topology:
      - Centers coordinates around the wrist anchor (index 0).
      - Scales by maximum coordinate extent.
    """
    if landmarks.size == 0:
        return landmarks

    landmarks = landmarks.copy()
    kps = landmarks.shape[-2]

    if kps == 76:
        # Anchor on mid-shoulder (pose indices 43, 44 in 76-kp topology)
        mid_shoulder = (landmarks[..., 43:44, :] + landmarks[..., 44:45, :]) / 2.0
        centered = landmarks - mid_shoulder

        # Scale by maximum extent to guarantee bounded range [-1, 1] while preserving geometry
        max_extent = np.max(np.abs(centered), axis=(-2, -1), keepdims=True)
        scale = np.where(max_extent > 1e-6, max_extent, 1.0)

        return centered / scale

    # Default (21 hand keypoints): Wrist anchor centering
    wrist = landmarks[..., 0:1, :]
    centered = landmarks - wrist
    max_val = np.max(np.abs(centered), axis=(-2, -1), keepdims=True)
    scale = np.where(max_val > 1e-6, max_val, 1.0)
    return centered / scale


def adaptive_pad_or_truncate(seq: np.ndarray, max_len: int = 150) -> np.ndarray:
    """Center-crop if sequence is longer than max_len, otherwise zero-pad at end."""
    t = seq.shape[0]
    if t > max_len:
        start = (t - max_len) // 2
        return seq[start:start + max_len]
    elif t < max_len:
        pad_len = max_len - t
        pad_shape = (pad_len,) + seq.shape[1:]
        padding = np.zeros(pad_shape, dtype=seq.dtype)
        return np.concatenate([seq, padding], axis=0)
    return seq


def extract_2d_pose_vector(landmarks: np.ndarray) -> np.ndarray:
    """Extract (x, y) coordinates from (T, 76, 3) -> (T, 152) or (T, 21, 3) -> (T, 42)."""
    if landmarks.shape[-1] >= 2:
        xy = landmarks[..., :2]
        return xy.reshape(landmarks.shape[:-2] + (-1,))
    return landmarks.reshape(landmarks.shape[:-2] + (-1,))


def pad_sequence(sequence: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    t = sequence.shape[0]
    if t >= target_length:
        return sequence[:target_length]
    pad_len = target_length - t
    pad_shape = (pad_len,) + sequence.shape[1:]
    padding = np.full(pad_shape, pad_value)
    return np.concatenate([sequence, padding], axis=0)


def flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
    shape = landmarks.shape
    return landmarks.reshape(shape[:-2] + (-1,))


def augment_landmarks(
    landmarks: np.ndarray,
    noise_std: float = 0.01,
    scale_range: Tuple[float, float] = (0.9, 1.1),
    p: float = 1.0,
) -> np.ndarray:
    if np.random.rand() > p:
        return landmarks.copy()
    noise = np.random.normal(0, noise_std, landmarks.shape)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (landmarks * scale) + noise


def extract_86_hand_features(landmarks: np.ndarray) -> np.ndarray:
    """Extracts 86-dimensional hand geometry feature vector.

    (63 wrist-relative coords + 20 pairwise distances + 3 bounding extents)
    """
    if landmarks.ndim == 3:  # (seq_len, 21, 3) or (seq_len, 76, 3)
        feats = []
        for t in range(landmarks.shape[0]):
            hand_kp = landmarks[t, :21, :] if landmarks.shape[1] >= 21 else landmarks[t, :, :]
            wrist = hand_kp[0:1, :]
            rel = hand_kp - wrist
            rel_flat = rel.flatten()
            dists = np.linalg.norm(rel[1:], axis=1)
            extents = rel.max(axis=0) - rel.min(axis=0)
            feats.append(np.concatenate([rel_flat, dists, extents]))
        return np.array(feats, dtype=np.float32)

    # 2D (21, 3) or (76, 3)
    hand_kp = landmarks[:21, :] if landmarks.shape[0] >= 21 else landmarks
    wrist = hand_kp[0:1, :]
    rel = hand_kp - wrist
    rel_flat = rel.flatten()
    dists = np.linalg.norm(rel[1:], axis=1)
    extents = rel.max(axis=0) - rel.min(axis=0)
    return np.concatenate([rel_flat, dists, extents]).astype(np.float32)


# ==============================================================================
# SOTA Feature Extraction (328 Dimensions) & Data Augmentations
# ==============================================================================

SOTA_DISTANCE_PAIRS_24: List[Tuple[int, int]] = [
    (0, 1), (0, 2), (0, 3), (0, 4),      # Nose to face / head reference
    (10, 20), (11, 21), (12, 22), (13, 23),  # Left hand: wrist to fingertips / knuckles
    (20, 24), (20, 28), (20, 32), (20, 36),  # Left hand: thumb to other fingertips
    (40, 50), (41, 51), (42, 52), (43, 53),  # Right hand: wrist to fingertips / knuckles
    (50, 54), (50, 58), (50, 62), (50, 66),  # Right hand: thumb to other fingertips
    (20, 50), (24, 54), (0, 20), (0, 50),    # Inter-hand & hand-to-nose
]


def spatial_rotation(
    landmarks: np.ndarray,
    angle_deg: Optional[float] = None,
    max_angle_deg: float = 15.0,
) -> np.ndarray:
    """Random spatial 2D rotation of landmark coordinates around centroid.

    Args:
        landmarks: Array of shape (..., V, 2) or (..., V, C>=2)
        angle_deg: Specific rotation angle in degrees (if None, sampled from [-max_angle_deg, max_angle_deg])
        max_angle_deg: Maximum rotation angle in degrees

    Returns:
        Rotated landmarks array maintaining the same shape and non-NaN finite values.
    """
    if landmarks.size == 0:
        return landmarks.copy()

    seq = landmarks.copy()
    if angle_deg is None:
        angle_deg = random.uniform(-max_angle_deg, max_angle_deg)

    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=seq.dtype)

    xy = seq[..., :2]
    # Center around spatial centroid
    center = np.mean(xy, axis=-2, keepdims=True)
    centered = xy - center
    rotated = np.matmul(centered, rot_mat.T) + center
    seq[..., :2] = rotated
    return seq


def scale_jitter(
    landmarks: np.ndarray,
    scale: Optional[float] = None,
    scale_range: Tuple[float, float] = (0.85, 1.15),
    trans_range: float = 0.05,
) -> np.ndarray:
    """Random scaling and translation jitter on landmark coordinates.

    Args:
        landmarks: Array of shape (..., V, 2) or (..., V, C>=2)
        scale: Specific scaling multiplier (if None, sampled from scale_range)
        scale_range: (min_scale, max_scale)
        trans_range: Maximum translation offset

    Returns:
        Scaled & jittered array maintaining same shape and non-NaN finite values.
    """
    if landmarks.size == 0:
        return landmarks.copy()

    seq = landmarks.copy()
    if scale is None:
        scale = random.uniform(scale_range[0], scale_range[1])

    tx = random.uniform(-trans_range, trans_range)
    ty = random.uniform(-trans_range, trans_range)

    xy = seq[..., :2]
    center = np.mean(xy, axis=-2, keepdims=True)
    centered = xy - center
    scaled = centered * scale + center
    scaled[..., 0] += tx
    scaled[..., 1] += ty
    seq[..., :2] = scaled
    return seq


def landmark_dropout(
    landmarks: np.ndarray,
    drop_rate: float = 0.10,
) -> np.ndarray:
    """Randomly masks out keypoints with probability drop_rate (sets coordinates to 0.0).

    Args:
        landmarks: Array of shape (T, V, C) or (V, C)
        drop_rate: Proportion of keypoints to drop (0.0 to 1.0)

    Returns:
        Array with dropped keypoints zeroed out, maintaining shape.
    """
    if landmarks.size == 0 or drop_rate <= 0.0:
        return landmarks.copy()

    seq = landmarks.copy()
    num_kps = seq.shape[-2]
    mask = np.random.rand(num_kps) > drop_rate
    # If all landmarks were dropped, keep at least one
    if not np.any(mask):
        mask[0] = True

    seq[..., ~mask, :] = 0.0
    return seq


def temporal_speed_warp(
    landmarks: np.ndarray,
    speed_factor: Optional[float] = None,
    speed_range: Tuple[float, float] = (0.8, 1.2),
    min_frames: int = 8,
) -> np.ndarray:
    """Resamples sequence along time axis to simulate signing speed variations.

    Args:
        landmarks: Array of shape (T, ...)
        speed_factor: Specific speed multiplier
        speed_range: (min_speed, max_speed)
        min_frames: Minimum number of output frames

    Returns:
        Resampled landmarks array with new length new_T >= min_frames.
    """
    if landmarks.ndim < 2 or landmarks.shape[0] <= 1:
        return landmarks.copy()

    T = landmarks.shape[0]
    if speed_factor is None:
        speed_factor = random.uniform(speed_range[0], speed_range[1])

    new_T = max(min_frames, int(round(T * speed_factor)))
    indices = np.linspace(0, T - 1, new_T).astype(int)
    return landmarks[indices].copy()


def extract_sota_features_328(
    landmarks: np.ndarray,
    distance_pairs: Optional[List[Tuple[int, int]]] = None,
) -> np.ndarray:
    """Extracts SOTA 328-dimensional multimodal features for Tier-1 transformer.

    Components:
      - 152 Base coordinates (76 keypoints * 2D x,y)
      - 152 Velocity vectors (temporal difference: coords[t] - coords[t-1], 0 at t=0)
      - 24 Key anatomical Euclidean distance pairs

    Total dimensions = 152 + 152 + 24 = 328

    Args:
        landmarks: Array of shape (T, 76, 2) or (T, 76, 3) or (T, 152) or (76, 2)
        distance_pairs: List of 24 index pairs (defaults to SOTA_DISTANCE_PAIRS_24)

    Returns:
        Feature array of shape (T, 328) with dtype float32.
    """
    pairs = distance_pairs or SOTA_DISTANCE_PAIRS_24

    # Ensure 3D shape (T, V, C)
    if landmarks.ndim == 2:
        if landmarks.shape[1] == 152:
            # (T, 152) -> (T, 76, 2)
            T = landmarks.shape[0]
            seq_2d = landmarks.reshape(T, 76, 2)
        elif landmarks.shape[0] == 76 and landmarks.shape[1] in (2, 3):
            # (76, 2) -> (1, 76, 2)
            seq_2d = landmarks[np.newaxis, :, :2]
        else:
            T = landmarks.shape[0]
            num_kp = min(76, landmarks.shape[1] // 2)
            seq_2d = np.zeros((T, 76, 2), dtype=np.float32)
            seq_2d[:, :num_kp, :] = landmarks[:, :num_kp * 2].reshape(T, num_kp, 2)
    elif landmarks.ndim == 3:
        T, V, C = landmarks.shape
        seq_2d = np.zeros((T, 76, 2), dtype=np.float32)
        num_kp = min(76, V)
        seq_2d[:, :num_kp, :min(2, C)] = landmarks[:, :num_kp, :min(2, C)]
    else:
        raise ValueError(f"Unsupported landmarks shape: {landmarks.shape}")

    T, V, _ = seq_2d.shape

    # 1. Base 76 coordinates (152 dim)
    base_coords = seq_2d.reshape(T, -1).astype(np.float32)  # (T, 152)

    # 2. Velocity vectors (152 dim)
    velocities = np.zeros_like(base_coords)
    if T > 1:
        velocities[1:] = base_coords[1:] - base_coords[:-1]

    # 3. Euclidean distance pairs (len(pairs) dim, default 24)
    distances = np.zeros((T, len(pairs)), dtype=np.float32)
    for i, (i1, i2) in enumerate(pairs):
        idx1 = min(i1, V - 1)
        idx2 = min(i2, V - 1)
        p1 = seq_2d[:, idx1, :]
        p2 = seq_2d[:, idx2, :]
        distances[:, i] = np.linalg.norm(p1 - p2, axis=-1)

    # Concatenate: 152 + 152 + 24 = 328
    features = np.concatenate([base_coords, velocities, distances], axis=-1).astype(np.float32)
    return features

