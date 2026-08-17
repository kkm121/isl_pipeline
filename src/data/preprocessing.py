import logging
from typing import Optional, Tuple

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
                import mediapipe as mp

                self.mp_holistic = mp.solutions.holistic
                self.holistic = self.mp_holistic.Holistic(
                    static_image_mode=self.static_image_mode,
                    min_detection_confidence=self.min_detection_confidence,
                    min_tracking_confidence=0.5,
                )
            except (ImportError, AttributeError):
                logger.warning("MediaPipe Holistic unavailable, falling back to Hands.")
                self.use_holistic = False

        if not self.use_holistic and self.hands is None:
            try:
                import mediapipe as mp

                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=self.static_image_mode,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence,
                )
            except ImportError:
                logger.error("MediaPipe not installed.")
                raise

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
            # Check if any non-zero landmarks were extracted
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
    """Multi-anchor coordinate normalization.

    If shape has 76 keypoints:
      - Left hand (0:21) centered on left wrist (0)
      - Right hand (21:42) centered on right wrist (21)
      - Pose (42:53) and Face (53:76) centered on mid-shoulder anchor (43, 44 midpoint)
    If shape has 21 keypoints:
      - Centered on wrist (0)
    """
    if landmarks.size == 0:
        return landmarks

    landmarks = landmarks.copy()
    kps = landmarks.shape[-2]

    if kps == 76:
        # Hand 1 (Left): 0:21
        lh_wrist = landmarks[..., 0:1, :]
        landmarks[..., 0:21, :] = landmarks[..., 0:21, :] - lh_wrist

        # Hand 2 (Right): 21:42
        rh_wrist = landmarks[..., 21:22, :]
        landmarks[..., 21:42, :] = landmarks[..., 21:42, :] - rh_wrist

        # Pose & Face: 42:76
        # Anchor on mid-shoulder (pose joints 1 & 2 in subset, corresponding to indices 43, 44)
        mid_shoulder = (landmarks[..., 43:44, :] + landmarks[..., 44:45, :]) / 2.0
        landmarks[..., 42:76, :] = landmarks[..., 42:76, :] - mid_shoulder

        max_val = np.max(np.abs(landmarks), axis=(-2, -1), keepdims=True)
        max_val[max_val == 0] = 1.0
        return landmarks / max_val

    # Default (e.g. 21 hand keypoints)
    wrist = landmarks[..., 0:1, :]
    centered = landmarks - wrist
    max_val = np.max(np.abs(centered), axis=(-2, -1), keepdims=True)
    max_val[max_val == 0] = 1.0
    return centered / max_val


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
