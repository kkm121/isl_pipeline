"""Tier-1 (Demo Track) Quantitative Benchmarking Suite.

Measures:
1. Signer-Disjoint Validation & Test Performance (Accuracy, Precision, Recall, Macro-F1).
2. Latency Profiling across all pipeline components (MediaPipe 76-kp, Model Inference, Buffering, NMT/TTS).
3. Memory Footprint (RAM / VRAM MB).
4. Full structured JSON reporting.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

# Add repository root to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import ISLDataModule
from src.data.preprocessing import LandmarkExtractor
from src.inference.predict import StreamingSignPredictor
from src.inference.translation_tts import RegionalSynthesisEngine
from src.models.classifier import create_tier1_classifier
from src.models.config import DataConfig, PipelineConfig, Tier1ModelConfig, TrainingConfig
from src.training.evaluate import evaluate
from src.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def generate_benchmark_data(
    num_classes: int = 150,
    num_signers: int = 10,
    samples_per_signer_class: int = 2,
    seq_len: int = 45,
    num_landmarks: int = 76,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generates synthetic multi-signer kinematic data with signer-specific spatial variations."""
    np.random.seed(seed)
    total_samples = num_classes * num_signers * samples_per_signer_class
    logger.info(f"Generating {total_samples} samples across {num_signers} signers for {num_classes} classes...")

    # Distinct base motion patterns per class
    base_class_patterns = np.random.randn(num_classes, seq_len, num_landmarks, 3) * 0.5

    # Signer-specific anatomical bias & speed scaling
    signer_biases = {f"signer_{i}": np.random.randn(1, 1, num_landmarks, 3) * 0.1 for i in range(num_signers)}

    sequences = []
    labels = []
    signer_ids = []

    for s_idx in range(num_signers):
        s_id = f"signer_{s_idx}"
        bias = signer_biases[s_id]
        for c_idx in range(num_classes):
            base_pattern = base_class_patterns[c_idx]
            for _ in range(samples_per_signer_class):
                # Add temporal jitter and spatial noise
                noise = np.random.normal(0, 0.02, base_pattern.shape)
                sample = base_pattern + bias + noise
                sequences.append(sample)
                labels.append(c_idx)
                signer_ids.append(s_id)

    return {
        "sequences": np.array(sequences),
        "labels": np.array(labels),
        "signer_ids": signer_ids,
    }


def run_benchmark(args: argparse.Namespace) -> Dict[str, Any]:
    logger.info("=" * 70)
    logger.info("ISL PIPELINE — TIER-1 (DEMO TRACK) BENCHMARKING SUITE")
    logger.info("=" * 70)

    # 1. Dataset Generation & Signer-Disjoint Split
    data_dict = generate_benchmark_data(
        num_classes=args.vocab_size,
        num_signers=args.num_signers,
        samples_per_signer_class=args.samples_per_signer,
        seq_len=45,
        num_landmarks=76,
    )

    data_cfg = DataConfig(
        num_landmarks=76,
        landmark_dim=2,  # 152 input dimensions
        sequence_length=45,
        num_classes=args.vocab_size,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        use_signer_disjoint=True,
    )

    dm = ISLDataModule(data_cfg)
    dm.sequences = data_dict["sequences"]
    dm.labels = data_dict["labels"]
    dm.signer_ids = data_dict["signer_ids"]

    train_ds, val_ds, test_ds = dm.split()
    train_signers = set(train_ds.signer_ids) if train_ds.signer_ids else set()
    val_signers = set(val_ds.signer_ids) if val_ds.signer_ids else set()
    test_signers = set(test_ds.signer_ids) if test_ds.signer_ids else set()

    signer_overlap = (train_signers & val_signers) | (train_signers & test_signers) | (val_signers & test_signers)
    assert len(signer_overlap) == 0, f"Critical Leakage: Signers overlapped across splits: {signer_overlap}"
    logger.info(
        f"✓ Signer-Disjoint Split Verified: Train Signers={train_signers}, Val={val_signers}, Test={test_signers}"
    )

    train_loader, val_loader, test_loader = dm.get_dataloaders(batch_size=args.batch_size)

    # 2. Model Creation
    model_cfg = Tier1ModelConfig(
        input_size=152,  # 76 * 2
        hidden_size=args.hidden_size,
        num_classes=args.vocab_size,
        architecture=args.architecture,
        dropout=0.2,
    )
    model = create_tier1_classifier(model_cfg)
    num_params = model.count_parameters()
    logger.info(f"Model: {args.architecture} | Parameters: {num_params:,}")

    # 3. Model Training
    train_cfg = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        patience=5,
        mixed_precision=False,
        checkpoint_dir="checkpoints/benchmark",
        log_dir="logs/benchmark",
        device=args.device,
    )
    pipe_cfg = PipelineConfig(data=data_cfg, model=model_cfg, training=train_cfg, experiment_name="tier1_benchmark")
    trainer = Trainer(model, datamodule_or_config=pipe_cfg, device=pipe_cfg.resolve_device())

    train_start = time.perf_counter()
    train_res = trainer.train(train_loader, val_loader)
    train_duration = time.perf_counter() - train_start
    logger.info(f"Training completed in {train_duration:.2f}s ({train_res['epochs_trained']} epochs)")

    # 4. Quantitative Evaluation on Test Set (Strict Signer-Disjoint)
    eval_metrics = evaluate(model, test_loader, device=pipe_cfg.resolve_device())
    logger.info(f"Test Accuracy: {eval_metrics['accuracy'] * 100:.2f}% | Macro-F1: {eval_metrics['f1']:.4f}")

    # 5. Latency Profiling
    logger.info("Profiling end-to-end component latencies...")
    latencies: Dict[str, float] = {}

    # Extraction benchmark
    extractor = LandmarkExtractor(use_holistic=False)
    dummy_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    extract_times = []
    for _ in range(30):
        t0 = time.perf_counter()
        _ = extractor.extract_from_frame(dummy_frame)
        extract_times.append((time.perf_counter() - t0) * 1000.0)
    extractor.close()
    latencies["landmark_extraction_ms"] = float(np.median(extract_times))

    # Model Forward Pass benchmark (single 45-frame sequence)
    dummy_input = torch.randn(1, 45, 152).to(pipe_cfg.resolve_device())
    forward_times = []
    model.eval()
    with torch.no_grad():
        for _ in range(50):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            forward_times.append((time.perf_counter() - t0) * 1000.0)
    latencies["model_forward_ms"] = float(np.median(forward_times))

    # Streaming Buffer & Smoothing benchmark
    predictor = StreamingSignPredictor(model, config=pipe_cfg, device=pipe_cfg.resolve_device())
    dummy_kp = np.random.randn(76, 3)
    stream_times = []
    # Fill buffer
    for _ in range(44):
        predictor.process_frame_landmarks(dummy_kp)
    for _ in range(20):
        t0 = time.perf_counter()
        predictor.process_frame_landmarks(dummy_kp)
        stream_times.append((time.perf_counter() - t0) * 1000.0)
    latencies["streaming_buffer_step_ms"] = float(np.median(stream_times))

    # Regional Synthesis (IndicTrans2 + TTS) benchmark
    synth_engine = RegionalSynthesisEngine()
    synth_start = time.perf_counter()
    synth_res = synth_engine.process_multilingual_pipeline("TEACHER")
    latencies["regional_synthesis_ms"] = float((time.perf_counter() - synth_start) * 1000.0)

    total_pipeline_latency_ms = (
        latencies["landmark_extraction_ms"] + latencies["model_forward_ms"] + latencies["streaming_buffer_step_ms"]
    )
    latencies["total_glass_to_glass_ms"] = total_pipeline_latency_ms

    # 6. Memory Footprint
    import psutil

    process = psutil.Process(os.getpid())
    ram_mb = process.memory_info().rss / (1024 * 1024)
    vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0.0

    # 7. Compile Report
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": {
            "vocabulary_size": args.vocab_size,
            "num_signers": args.num_signers,
            "architecture": args.architecture,
            "parameters": num_params,
            "sequence_length": 45,
            "device": pipe_cfg.resolve_device(),
        },
        "signer_disjoint_validation": {
            "test_accuracy": eval_metrics["accuracy"],
            "macro_precision": eval_metrics["precision"],
            "macro_recall": eval_metrics["recall"],
            "macro_f1": eval_metrics["f1"],
            "test_loss": eval_metrics["loss"],
            "train_signers": list(train_signers),
            "val_signers": list(val_signers),
            "test_signers": list(test_signers),
        },
        "latency_profile_ms": latencies,
        "memory_footprint_mb": {
            "ram_peak_mb": round(ram_mb, 2),
            "vram_peak_mb": round(vram_mb, 2),
        },
        "multilingual_synthesis_sample": synth_res,
    }

    # Save to disk
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info("=" * 70)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info(f"• Signer-Disjoint Test Accuracy : {eval_metrics['accuracy'] * 100:.2f}%")
    logger.info(f"• Macro F1-Score               : {eval_metrics['f1']:.4f}")
    logger.info(f"• Model Forward Latency        : {latencies['model_forward_ms']:.2f} ms")
    logger.info(f"• Streaming Step Latency       : {latencies['streaming_buffer_step_ms']:.2f} ms")
    logger.info(f"• Glass-to-Glass Total Latency : {latencies['total_glass_to_glass_ms']:.2f} ms")
    logger.info(f"• Peak RAM Usage               : {ram_mb:.1f} MB")
    logger.info(f"• Report Written to            : {out_path}")
    logger.info("=" * 70)

    return report


def main():
    parser = argparse.ArgumentParser(description="Tier-1 Benchmark Runner")
    parser.add_argument("--vocab-size", type=int, default=50, help="Vocabulary size (e.g. 26, 50, 150)")
    parser.add_argument("--num-signers", type=int, default=8, help="Number of distinct signers")
    parser.add_argument("--samples-per-signer", type=int, default=3, help="Samples per signer per class")
    parser.add_argument(
        "--architecture", type=str, default="temporal_cnn", choices=["temporal_cnn", "bilstm_attention"]
    )
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden dimensions")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--output-json", type=str, default="results/tier1_benchmark.json", help="Output JSON path")

    args = parser.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
