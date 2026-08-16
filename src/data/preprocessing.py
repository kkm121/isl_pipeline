import numpy as np
from typing import Optional
from pathlib import Path
import json
import logging
import cv2

logger = logging.getLogger(__name__)

class LandmarkExtractor:
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.mp_hands = None
        self.hands = None

    def _init_mediapipe(self):
        if self.mp_hands is None:
            try:
                import mediapipe as mp
                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(
                    static_image_mode=self.static_image_mode,
                    max_num_hands=self.max_num_hands,
                    min_detection_confidence=self.min_detection_confidence
                )
            except ImportError:
                logger.error("MediaPipe not installed.")
                raise

    def extract_from_frame(self, frame: np.ndarray) -> Optional[np.ndarray]:
        self._init_mediapipe()
        results = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
        if not frames_landmarks:
            return np.zeros((0, 21, 3))
        return np.stack(frames_landmarks)

    def close(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None

def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if landmarks.size == 0:
        return landmarks
    wrist = landmarks[..., 0:1, :]
    centered = landmarks - wrist
    max_val = np.max(np.abs(centered), axis=(-2, -1), keepdims=True)
    max_val[max_val == 0] = 1.0
    return centered / max_val

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

def augment_landmarks(landmarks: np.ndarray, noise_std: float = 0.01, scale_range: tuple = (0.9, 1.1), p: float = 1.0) -> np.ndarray:
    if np.random.rand() > p:
        return landmarks.copy()
    noise = np.random.normal(0, noise_std, landmarks.shape)
    scale = np.random.uniform(scale_range[0], scale_range[1])
    return (landmarks * scale) + noise
