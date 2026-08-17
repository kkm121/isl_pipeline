import argparse
import json
import logging
import time
from collections import deque
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np
import torch

from src.data.dataset import CLASSROOM_VOCABULARY_200
from src.data.preprocessing import (
    LandmarkExtractor,
    extract_2d_pose_vector,
    flatten_landmarks,
    interpolate_missing_landmarks,
    normalize_landmarks,
    pad_sequence,
)
from src.models.classifier import ISLClassifier, Tier1TemporalCNN
from src.models.config import BufferConfig, PipelineConfig

logger = logging.getLogger(__name__)

ISL_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


class UIStreamState(str, Enum):
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    BUFFERING = "BUFFERING"
    PROCESSING = "PROCESSING"
    PREDICTED = "PREDICTED"


class StreamingSignPredictor:
    """Real-Time Temporal Sliding Window Buffer & State Machine for ISL Live Demos.

    Manages continuous video frame streaming:
    - Buffers 1.5 - 2.5s (e.g. 45 frames @ 30fps).
    - Transitions through UI states: LISTENING -> BUFFERING -> PROCESSING -> PREDICTED.
    - Applies temporal consensus smoothing across sliding strides (e.g. 5 frames).
    """

    def __init__(
        self,
        model: Union[ISLClassifier, Tier1TemporalCNN],
        config: Optional[PipelineConfig] = None,
        class_names: Optional[List[str]] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.config = config or PipelineConfig()
        self.device = device
        self.buffer_config: BufferConfig = self.config.buffer

        # Set vocabulary list
        num_classes = getattr(model.config, "num_classes", 26)
        if class_names is not None:
            self.class_names = class_names
        elif num_classes > 26:
            self.class_names = CLASSROOM_VOCABULARY_200[:num_classes]
        else:
            self.class_names = ISL_CLASSES[:num_classes]

        # Temporal landmark buffer: stores (num_kps, dim)
        self.window_size = self.buffer_config.window_size
        self.landmark_buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
        self.prediction_history: Deque[int] = deque(maxlen=self.buffer_config.consensus_frames)

        self.state: UIStreamState = UIStreamState.IDLE
        self.frame_counter: int = 0
        self.last_stable_prediction: Optional[Dict[str, Any]] = None

    def reset_buffer(self) -> None:
        self.landmark_buffer.clear()
        self.prediction_history.clear()
        self.state = UIStreamState.IDLE
        self.frame_counter = 0
        self.last_stable_prediction = None

    def process_frame_landmarks(self, landmarks: Optional[np.ndarray]) -> Dict[str, Any]:
        """Ingests landmarks for a single video frame and advances the state engine.

        Args:
            landmarks: (76, 3) or (21, 3) keypoint array, or None if no sign detected.

        Returns:
            Structured stream status packet including UI state, buffer fill %, and prediction.
        """
        start_time = time.perf_counter()

        if landmarks is None or np.all(landmarks == 0):
            if len(self.landmark_buffer) == 0:
                self.state = UIStreamState.IDLE
            else:
                self.state = UIStreamState.LISTENING
            return {
                "state": self.state.value,
                "buffer_fill_ratio": len(self.landmark_buffer) / self.window_size,
                "current_frames": len(self.landmark_buffer),
                "target_frames": self.window_size,
                "prediction": self.last_stable_prediction,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            }

        # Valid landmarks detected: push into rolling buffer
        self.landmark_buffer.append(landmarks)
        self.frame_counter += 1

        # Check if buffer is still accumulating temporal sequence
        if len(self.landmark_buffer) < self.window_size:
            self.state = UIStreamState.BUFFERING
            return {
                "state": self.state.value,
                "buffer_fill_ratio": len(self.landmark_buffer) / self.window_size,
                "current_frames": len(self.landmark_buffer),
                "target_frames": self.window_size,
                "prediction": None,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            }

        # Buffer is full -> evaluate at step intervals
        if self.frame_counter % self.buffer_config.step_size != 0 and self.last_stable_prediction:
            return {
                "state": self.state.value,
                "buffer_fill_ratio": 1.0,
                "current_frames": len(self.landmark_buffer),
                "target_frames": self.window_size,
                "prediction": self.last_stable_prediction,
                "latency_ms": (time.perf_counter() - start_time) * 1000.0,
            }

        # Execute Model Inference
        self.state = UIStreamState.PROCESSING
        seq = np.stack(list(self.landmark_buffer))
        seq = interpolate_missing_landmarks(seq)
        seq = normalize_landmarks(seq)
        seq = pad_sequence(seq, self.config.data.sequence_length)

        # Coordinate extraction: check if model expects 152-dim (76 * 2) or config specifies 2D
        expected_input = getattr(self.model.config, "input_size", 152)
        if (expected_input == 152 or getattr(self.config.data, "landmark_dim", 2) == 2) and seq.shape[-1] > 2:
            seq = extract_2d_pose_vector(seq)
        else:
            seq = flatten_landmarks(seq)

        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        infer_start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)
        infer_latency_ms = (time.perf_counter() - infer_start) * 1000.0

        top_prob, top_idx = torch.topk(probs, k=1)
        best_prob = float(top_prob.item())
        best_idx = int(top_idx.item())

        # Temporal Consensus Smoothing
        self.prediction_history.append(best_idx)
        is_consensus = (
            len(self.prediction_history) == self.buffer_config.consensus_frames
            and len(set(self.prediction_history)) == 1
            and best_prob >= self.buffer_config.min_confidence
        )

        class_name = self.class_names[best_idx] if best_idx < len(self.class_names) else f"CLASS_{best_idx}"

        if is_consensus:
            self.state = UIStreamState.PREDICTED
            self.last_stable_prediction = {
                "class_id": best_idx,
                "class_name": class_name,
                "confidence": best_prob,
                "is_stable": True,
            }
        elif self.last_stable_prediction is not None:
            self.state = UIStreamState.PREDICTED
        else:
            self.state = UIStreamState.BUFFERING

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "state": self.state.value,
            "buffer_fill_ratio": 1.0,
            "current_frames": len(self.landmark_buffer),
            "target_frames": self.window_size,
            "prediction": self.last_stable_prediction
            or {
                "class_id": best_idx,
                "class_name": class_name,
                "confidence": best_prob,
                "is_stable": False,
            },
            "inference_latency_ms": infer_latency_ms,
            "total_latency_ms": total_latency_ms,
        }


class ISLPredictor:
    """High-level Predictor for Offline & Batch Sequence Evaluation."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        if config_path:
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()

        resolved_device = self.config.resolve_device() if device == "auto" else device
        self.device = resolved_device

        # Load checkpoint and instantiate model
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model_type = checkpoint.get("model_type", "bilstm_attention")
        self.model: Union[Tier1TemporalCNN, ISLClassifier, torch.nn.Module]
        if model_type == "temporal_cnn":
            self.model = Tier1TemporalCNN.load(checkpoint_path, self.device)
        else:
            self.model = ISLClassifier.load(checkpoint_path, self.device)
        self.model.eval()

        num_classes = getattr(self.model.config, "num_classes", 26)
        if num_classes > 26:
            self.class_names = CLASSROOM_VOCABULARY_200[:num_classes]
        else:
            self.class_names = ISL_CLASSES[:num_classes]

    def predict(self, landmarks: np.ndarray) -> Dict[str, Any]:
        seq = normalize_landmarks(landmarks)
        seq = pad_sequence(seq, self.config.data.sequence_length)
        expected_input = getattr(self.model.config, "input_size", 152)
        if (expected_input == 152 or getattr(self.config.data, "landmark_dim", 2) == 2) and seq.shape[-1] > 2:
            seq = extract_2d_pose_vector(seq)
        else:
            seq = flatten_landmarks(seq)

        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)

        k_val = min(5, len(self.class_names))
        top_k_probs, top_k_indices = torch.topk(probs, k=k_val)

        top_k = [
            {"class": self.class_names[int(idx.item())], "prob": float(prob.item())}
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]

        best_idx = int(top_k_indices[0].item())
        best_prob = float(top_k_probs[0].item())

        return {
            "class_id": best_idx,
            "class_name": self.class_names[best_idx],
            "confidence": best_prob,
            "top_k": top_k,
        }

    def predict_batch(self, landmarks_batch: List[np.ndarray]) -> List[Dict[str, Any]]:
        results = []
        for lms in landmarks_batch:
            results.append(self.predict(lms))
        return results

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        extractor = LandmarkExtractor()
        landmarks = extractor.extract_from_video(video_path)
        extractor.close()

        if len(landmarks) == 0:
            return {"error": "No landmarks detected in video."}

        return self.predict(landmarks)


def main() -> None:
    parser = argparse.ArgumentParser(description="ISL Predictor")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to input video or .npy file")
    parser.add_argument("--config", type=str, default=None, help="Path to config yaml")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    args = parser.parse_args()

    predictor = ISLPredictor(args.checkpoint, args.config, args.device)

    input_path = Path(args.input)
    if input_path.suffix == ".npy":
        landmarks = np.load(args.input)
        result = predictor.predict(landmarks)
    else:
        result = predictor.predict_video(args.input)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
