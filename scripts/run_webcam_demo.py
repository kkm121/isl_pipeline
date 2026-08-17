"""Live Interactive Webcam Indian Sign Language (ISL) Recognition Demo.

Real-time streaming pipeline:
1. Captures live webcam frames via OpenCV.
2. Extracts 76-keypoint MediaPipe Holistic landmarks (Hands, Upper Pose, Facial NMMs).
3. Evaluates temporal sliding buffer via Tier-1 Temporal 1D-CNN (sub-5ms CPU latency).
4. Translates recognized ISL signs into Indian regional languages (Hindi, Tamil, Telugu, Bengali).
5. Renders live HUD with state machine status, landmark skeletons, and top predictions.

Usage:
    python scripts/run_webcam_demo.py
    python scripts/run_webcam_demo.py --camera 0 --checkpoint kaggle_output/tier1_best.pth
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import CLASSROOM_VOCABULARY_200  # noqa: E402
from src.data.preprocessing import LandmarkExtractor  # noqa: E402
from src.inference.predict import StreamingSignPredictor  # noqa: E402
from src.inference.translation_tts import RegionalSynthesisEngine  # noqa: E402
from src.models.classifier import Tier1TemporalCNN  # noqa: E402
from src.models.config import PipelineConfig, Tier1ModelConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ISL_Webcam_Demo")

SUPPORTED_LANGUAGES = [
    ("eng_Latn", "English"),
    ("hin_Deva", "Hindi (हिन्दी)"),
    ("tam_Taml", "Tamil (தமிழ்)"),
    ("tel_Telu", "Telugu (తెలుగు)"),
    ("ben_Beng", "Bengali (বাংলা)"),
]


def draw_hud(
    frame: np.ndarray,
    stream_packet: dict,
    fps: float,
    active_lang: tuple,
    synthesis_engine: RegionalSynthesisEngine,
) -> np.ndarray:
    """Renders professional HUD overlays with state machine, buffer meters, and multilingual translations."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Top Status Bar Background
    cv2.rectangle(overlay, (0, 0), (w, 100), (20, 24, 33), -1)
    # Bottom Controls Bar Background
    cv2.rectangle(overlay, (0, h - 50), (w, h), (20, 24, 33), -1)

    # Blend header and footer
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # 1. State Machine Badge
    state = stream_packet.get("state", "IDLE")
    state_colors = {
        "IDLE": (128, 128, 128),
        "LISTENING": (255, 180, 50),
        "BUFFERING": (0, 165, 255),
        "PROCESSING": (255, 255, 0),
        "PREDICTED": (50, 205, 50),
    }
    badge_color = state_colors.get(state, (200, 200, 200))
    cv2.rectangle(frame, (15, 15), (140, 50), badge_color, -1)
    cv2.putText(frame, state, (25, 40), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 2)

    # 2. Buffer Progress Bar
    fill_ratio = stream_packet.get("buffer_fill_ratio", 0.0)
    current_frames = stream_packet.get("current_frames", 0)
    target_frames = stream_packet.get("target_frames", 45)

    bar_x, bar_y, bar_w, bar_h = 160, 22, 180, 18
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 65, 80), -1)
    fill_w = int(bar_w * fill_ratio)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), (0, 200, 255), -1)
    cv2.putText(
        frame,
        f"Buffer: {current_frames}/{target_frames}",
        (bar_x, bar_y + 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 190, 205),
        1,
    )

    # 3. Telemetry (FPS & Latency)
    infer_ms = stream_packet.get("inference_latency_ms", 0.0)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)
    cv2.putText(frame, f"Latency: {infer_ms:.1f} ms", (w - 180, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 1)

    # 4. Recognized ISL Sign & Regional Translation
    pred = stream_packet.get("prediction")
    if pred and pred.get("class_name"):
        sign_name = pred["class_name"]
        conf = pred.get("confidence", 0.0) * 100.0

        # Translation lookup
        lang_code, lang_name = active_lang
        trans = synthesis_engine.translate_text(sign_name, target_lang=lang_code)
        translated_text = trans.get("translated_text", sign_name)

        # Center Card for Prediction
        card_w, card_h = 420, 90
        card_x = (w - card_w) // 2
        card_y = 110

        card_overlay = frame.copy()
        cv2.rectangle(
            card_overlay,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (15, 20, 28),
            -1,
        )
        cv2.addWeighted(card_overlay, 0.80, frame, 0.20, 0, frame)
        cv2.rectangle(
            frame,
            (card_x, card_y),
            (card_x + card_w, card_y + card_h),
            (0, 200, 100) if conf > 70 else (0, 165, 255),
            2,
        )

        cv2.putText(
            frame,
            f"ISL SIGN: {sign_name}",
            (card_x + 20, card_y + 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Confidence: {conf:.1f}%",
            (card_x + 260, card_y + 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
        )
        cv2.putText(
            frame,
            f"{lang_name}: {translated_text}",
            (card_x + 20, card_y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 215, 0),
            2,
        )

    # 5. Footer Instructions Bar
    lang_label = f"Lang: {active_lang[1]}"
    controls_text = f"[Q] Quit  |  [R] Reset Buffer  |  [L] Cycle {lang_label}"
    cv2.putText(frame, controls_text, (20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 230, 240), 1)

    return frame


def run_webcam_stream(
    camera_id: int = 0,
    checkpoint_path: str = "kaggle_output/tier1_best.pth",
    device: str = "cpu",
):
    print("\n" + "=" * 70, flush=True)
    print("=== INITIALIZING INDIAN SIGN LANGUAGE (ISL) LIVE WEBCAM DEMO ===", flush=True)
    print("=" * 70, flush=True)
    print(f"Camera ID   : {camera_id}", flush=True)
    print(f"Checkpoint  : {checkpoint_path}", flush=True)
    print(f"Inference   : {device.upper()}", flush=True)
    print("=" * 70 + "\n", flush=True)

    # 1. Instantiate Model
    num_classes = 200
    t1_cfg = Tier1ModelConfig(num_classes=num_classes, input_size=152)
    model = Tier1TemporalCNN(config=t1_cfg)

    # Load weights if available
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...", flush=True)
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
        # Filter matching weights
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"Loaded {len(pretrained_dict)} matching tensor layers.", flush=True)
    else:
        print("Checkpoint not found; using initialized model weights.", flush=True)

    model.eval()

    # 2. Setup Predictor and Synthesis Engine
    pipe_cfg = PipelineConfig()
    pipe_cfg.buffer.window_size = 45
    pipe_cfg.buffer.step_size = 5
    pipe_cfg.buffer.consensus_frames = 3

    predictor = StreamingSignPredictor(
        model=model,
        config=pipe_cfg,
        class_names=CLASSROOM_VOCABULARY_200,
        device=device,
    )
    synthesis_engine = RegionalSynthesisEngine()
    extractor = LandmarkExtractor()

    # 3. Open Video Stream
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"Error: Unable to open webcam on index {camera_id}.", flush=True)
        print(
            "Tip: If you have an external webcam or virtual camera, try passing --camera 1 or --camera 2.", flush=True
        )
        return

    # Set frame resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    lang_idx = 0
    prev_time = time.perf_counter()
    fps = 30.0

    print("Live camera stream started! Press 'q' in the window to quit.\n", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from webcam.")
                break

            # Flip horizontally for natural mirror display
            frame = cv2.flip(frame, 1)

            # Compute FPS
            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-5, now - prev_time))
            prev_time = now

            # Extract 76-keypoint holistic landmarks
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            landmarks = extractor.extract_from_frame(rgb_frame)

            # Draw visual landmarks
            if landmarks is not None:
                h, w, _ = frame.shape
                # Draw hands in cyan
                for i in range(42):
                    pt = landmarks[i]
                    if pt[0] != 0 or pt[1] != 0:
                        px, py = int(pt[0] * w), int(pt[1] * h)
                        cv2.circle(frame, (px, py), 3, (255, 255, 0), -1)
                # Draw upper body pose in yellow
                for i in range(42, 53):
                    pt = landmarks[i]
                    if pt[0] != 0 or pt[1] != 0:
                        px, py = int(pt[0] * w), int(pt[1] * h)
                        cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)
                # Draw facial NMM in green
                for i in range(53, 76):
                    pt = landmarks[i]
                    if pt[0] != 0 or pt[1] != 0:
                        px, py = int(pt[0] * w), int(pt[1] * h)
                        cv2.circle(frame, (px, py), 2, (0, 255, 0), -1)

            # Process through sliding window buffer
            stream_packet = predictor.process_frame_landmarks(landmarks)

            # Render HUD
            active_lang = SUPPORTED_LANGUAGES[lang_idx % len(SUPPORTED_LANGUAGES)]
            frame = draw_hud(frame, stream_packet, fps, active_lang, synthesis_engine)

            # Display Window
            cv2.imshow("Indian Sign Language (ISL) Recognition - Live Demo", frame)

            # Keyboard Input Handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quitting demo...")
                break
            elif key == ord("r"):
                predictor.reset_buffer()
                print("Buffer reset.")
            elif key == ord("l"):
                lang_idx = (lang_idx + 1) % len(SUPPORTED_LANGUAGES)
                print(f"Switched regional language to: {SUPPORTED_LANGUAGES[lang_idx][1]}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()
        print("Webcam released and resources cleaned up.")


def main():
    parser = argparse.ArgumentParser(description="Live ISL Webcam Demo")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="kaggle_output/tier1_best.pth",
        help="Path to model checkpoint",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda)")
    args = parser.parse_args()

    run_webcam_stream(
        camera_id=args.camera,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )


if __name__ == "__main__":
    main()
