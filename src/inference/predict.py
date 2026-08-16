import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.data.preprocessing import (
    LandmarkExtractor,
    flatten_landmarks,
    normalize_landmarks,
    pad_sequence,
)
from src.models.classifier import ISLClassifier
from src.models.config import PipelineConfig

logger = logging.getLogger(__name__)

ISL_CLASSES = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


class ISLPredictor:
    def __init__(
        self,
        checkpoint_path: str,
        config_path: Optional[str] = None,
        device: str = "auto",
    ):
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.model = ISLClassifier.load(checkpoint_path, self.device)
        self.model.eval()

        if config_path:
            self.config = PipelineConfig.from_yaml(config_path)
        else:
            self.config = PipelineConfig()

    def predict(self, landmarks: np.ndarray) -> Dict[str, Any]:
        # landmarks: (T, 21, 3)
        seq = normalize_landmarks(landmarks)
        seq = pad_sequence(seq, self.config.data.sequence_length)
        seq = flatten_landmarks(seq)

        tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0)

        top_k_probs, top_k_indices = torch.topk(probs, k=min(5, len(ISL_CLASSES)))

        top_k = [
            {"class": ISL_CLASSES[int(idx.item())], "prob": float(prob.item())}
            for prob, idx in zip(top_k_probs, top_k_indices)
        ]

        best_idx = int(top_k_indices[0].item())
        best_prob = float(top_k_probs[0].item())

        return {
            "class_id": best_idx,
            "class_name": ISL_CLASSES[best_idx],
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
            return {"error": "No hand landmarks detected in video."}

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
