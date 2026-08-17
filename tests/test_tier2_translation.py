import json

import numpy as np
import torch

from src.data.dataset import ISLDataModule
from src.data.preprocessing import normalize_landmarks
from src.inference.predict import StreamingSignPredictor
from src.inference.translation_tts import RegionalSynthesisEngine
from src.models.classifier import Tier1TemporalCNN
from src.models.config import (
    DataConfig,
    PipelineConfig,
    SynthesisConfig,
    Tier1ModelConfig,
    Tier2SignFormerConfig,
)
from src.models.signformer_gcn import PositionalEncoding, SignFormerGCN

# ==============================================================================
# 1. Tier-2 Autoregressive Transformer Decoder Tests
# ==============================================================================


def test_positional_encoding():
    d_model = 64
    max_len = 100
    pos_enc = PositionalEncoding(d_model=d_model, max_len=max_len)

    x = torch.zeros(2, 20, d_model)
    out = pos_enc(x)

    assert out.shape == (2, 20, d_model)
    # Ensure positional encoding is not all zeros
    assert not torch.allclose(out, torch.zeros_like(out))
    # Ensure different timesteps have different encodings
    assert not torch.allclose(out[:, 0, :], out[:, 1, :])


def test_tier2_teacher_forced_forward():
    cfg = Tier2SignFormerConfig(
        num_nodes=76,
        in_channels=2,
        graph_hidden_dim=32,
        transformer_d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        vocab_size=100,
        max_target_len=32,
    )
    model = SignFormerGCN(cfg)
    model.train()

    batch_size = 3
    src_len = 16
    tgt_len = 10

    src = torch.randn(batch_size, src_len, 76, 2)
    tgt = torch.randint(0, cfg.vocab_size, (batch_size, tgt_len))

    logits = model(src, tgt)
    assert logits.shape == (batch_size, tgt_len, cfg.vocab_size)

    # Verify backpropagation and gradient flow through all components
    loss = logits.sum()
    loss.backward()

    assert model.tgt_embedding.weight.grad is not None
    assert model.translation_head.weight.grad is not None
    assert model.stgcn1.sgcn.fc.weight.grad is not None


def test_tier2_autoregressive_generate():
    cfg = Tier2SignFormerConfig(
        num_nodes=76,
        in_channels=2,
        graph_hidden_dim=32,
        transformer_d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        vocab_size=50,
        max_target_len=32,
    )
    model = SignFormerGCN(cfg)
    model.eval()

    batch_size = 2
    src_len = 12
    max_gen_len = 8
    start_token = 1
    end_token = 2

    src = torch.randn(batch_size, src_len, 76, 2)

    with torch.no_grad():
        generated = model.generate(
            src=src,
            max_len=max_gen_len,
            start_token_id=start_token,
            end_token_id=end_token,
        )

    # Generated tokens should have shape (batch_size, gen_len) where gen_len <= max_gen_len + 1
    assert generated.dim() == 2
    assert generated.shape[0] == batch_size
    assert generated.shape[1] <= max_gen_len + 1
    # First token must be start_token_id
    assert torch.all(generated[:, 0] == start_token)


def test_tier2_causal_mask_verification():
    """Verify that decoder causal masking strictly prevents attention to future tokens."""
    cfg = Tier2SignFormerConfig(
        num_nodes=76,
        in_channels=2,
        graph_hidden_dim=32,
        transformer_d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        vocab_size=100,
    )
    model = SignFormerGCN(cfg)
    model.eval()

    src = torch.randn(1, 10, 76, 2)
    tgt1 = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    # Change future tokens at indices 3 and 4
    tgt2 = torch.tensor([[10, 20, 30, 99, 88]], dtype=torch.long)

    with torch.no_grad():
        out1 = model(src, tgt1)
        out2 = model(src, tgt2)

    # Position 0, 1, 2 must produce IDENTICAL output despite mutations at indices 3 and 4
    assert torch.allclose(out1[:, 0:3, :], out2[:, 0:3, :], atol=1e-5)


def test_tier2_model_save_and_load(tmp_path):
    cfg = Tier2SignFormerConfig(
        num_nodes=76,
        in_channels=2,
        graph_hidden_dim=32,
        transformer_d_model=64,
        nhead=4,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
        vocab_size=50,
    )
    model = SignFormerGCN(cfg)
    model.eval()

    save_path = str(tmp_path / "tier2_signformer.pt")
    model.save(save_path)

    loaded_model = SignFormerGCN.load(save_path)
    loaded_model.eval()

    src = torch.randn(2, 10, 76, 2)
    tgt = torch.randint(0, 50, (2, 5))

    with torch.no_grad():
        out1 = model(src, tgt)
        out2 = loaded_model(src, tgt)

    assert torch.allclose(out1, out2, atol=1e-5)


# ==============================================================================
# 2. Clean normalize_landmarks() Behavior & Single-Anchor Tests
# ==============================================================================


def test_clean_normalize_landmarks_76kp_single_anchor():
    """Verify 76-keypoint normalization centers on mid-shoulder anchor (joints 43, 44)."""
    # Create synthetic trajectory
    seq = np.random.uniform(10.0, 50.0, size=(10, 76, 3))

    norm = normalize_landmarks(seq)

    # 1. Output shape must be preserved
    assert norm.shape == seq.shape
    # 2. Normalized values must be bounded in [-1, 1]
    assert np.all(norm >= -1.0001) and np.all(norm <= 1.0001)

    # 3. Check that the midpoint between shoulders is zero
    norm_mid_shoulder = (norm[:, 43, :] + norm[:, 44, :]) / 2.0
    assert np.allclose(norm_mid_shoulder, 0.0, atol=1e-5)


def test_clean_normalize_landmarks_21kp_wrist_anchor():
    """Verify 21-keypoint normalization centers on wrist anchor (joint 0)."""
    seq = np.random.uniform(5.0, 25.0, size=(15, 21, 3))
    norm = normalize_landmarks(seq)

    assert norm.shape == seq.shape
    assert np.all(norm >= -1.0001) and np.all(norm <= 1.0001)

    # Wrist (joint 0) should be at coordinate origin (0, 0, 0)
    assert np.allclose(norm[:, 0, :], 0.0, atol=1e-5)


def test_clean_normalize_landmarks_zero_test_bypass():
    """Confirm zero test-bypass artifacts exist regardless of data mean or distribution."""
    # Test case 1: High mean (previously triggered the old threshold check)
    high_mean_seq = np.random.randn(5, 76, 3) + 100.0
    norm_high = normalize_landmarks(high_mean_seq)
    mid_shoulder_high = (norm_high[:, 43, :] + norm_high[:, 44, :]) / 2.0
    assert np.allclose(mid_shoulder_high, 0.0, atol=1e-5)

    # Test case 2: Low mean
    low_mean_seq = np.random.randn(5, 76, 3) - 50.0
    norm_low = normalize_landmarks(low_mean_seq)
    mid_shoulder_low = (norm_low[:, 43, :] + norm_low[:, 44, :]) / 2.0
    assert np.allclose(mid_shoulder_low, 0.0, atol=1e-5)

    # Test case 3: Empty input
    empty_seq = np.zeros((0, 76, 3))
    norm_empty = normalize_landmarks(empty_seq)
    assert norm_empty.shape == (0, 76, 3)


# ==============================================================================
# 3. load_include_dataset() Ingestion & Signer-Disjoint Splitting Tests
# ==============================================================================


def test_load_include_dataset_multi_format(tmp_path):
    """Test load_include_dataset with .npy, .csv, and .json landmark files."""
    data_dir = tmp_path / "include_data"
    data_dir.mkdir()

    class0_dir = data_dir / "class_hello"
    class1_dir = data_dir / "class_thankyou"
    class0_dir.mkdir()
    class1_dir.mkdir()

    # 1. Save .npy sample
    sample_npy = np.random.randn(30, 76, 3)
    np.save(class0_dir / "signer1_001.npy", sample_npy)

    # 2. Save .csv sample
    sample_csv = np.random.randn(30, 76 * 3)
    import pandas as pd

    df = pd.DataFrame(sample_csv)
    df.to_csv(class0_dir / "signer2_002.csv", index=False)

    # 3. Save .json sample
    sample_json = {"landmarks": np.random.randn(30, 76, 3).tolist()}
    with open(class1_dir / "signer3_003.json", "w") as f:
        json.dump(sample_json, f)

    cfg = DataConfig(num_landmarks=76, landmark_dim=3, sequence_length=30, num_classes=2)
    dm = ISLDataModule(cfg)
    dm.load_include_dataset(str(data_dir))

    assert dm.sequences is not None
    assert dm.labels is not None
    assert dm.signer_ids is not None
    assert len(dm.sequences) == 3
    assert set(dm.signer_ids) == {"signer1", "signer2", "signer3"}


def test_load_include_dataset_signer_disjoint_partitioning(tmp_path):
    """Test load_include_dataset end-to-end with strict signer-disjoint splitting."""
    data_dir = tmp_path / "include_dataset"
    data_dir.mkdir()

    classes = ["A", "B", "C"]
    num_signers = 6

    for c_idx, c_name in enumerate(classes):
        c_dir = data_dir / c_name
        c_dir.mkdir()
        for s_idx in range(num_signers):
            s_name = f"signer{s_idx}"
            arr = np.random.randn(20, 76, 3)
            np.save(c_dir / f"{s_name}_sample_{c_idx}.npy", arr)

    cfg = DataConfig(
        num_landmarks=76,
        landmark_dim=3,
        sequence_length=20,
        num_classes=3,
        use_signer_disjoint=True,
        train_split=0.6,
        val_split=0.2,
        test_split=0.2,
    )
    dm = ISLDataModule(cfg)
    dm.load_include_dataset(str(data_dir))
    train_ds, val_ds, test_ds = dm.split()

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0

    train_signers = set(train_ds.signer_ids)
    val_signers = set(val_ds.signer_ids)
    test_signers = set(test_ds.signer_ids)

    # Verify strictly 0 overlap
    assert len(train_signers & val_signers) == 0
    assert len(train_signers & test_signers) == 0
    assert len(val_signers & test_signers) == 0


# ==============================================================================
# 4. Granular Latency Profiler Output Validation
# ==============================================================================


def test_latency_profiler_output_validation():
    """Verify granular latency metrics across model forward pass, streaming buffer, and regional synthesis."""
    # Model Forward Pass Latency
    cfg = Tier1ModelConfig(input_size=152, hidden_size=64, num_classes=10)
    model = Tier1TemporalCNN(cfg)
    model.eval()

    dummy_input = torch.randn(1, 45, 152)
    import time

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_input)
    model_forward_ms = (time.perf_counter() - t0) * 1000.0

    assert model_forward_ms > 0
    assert model_forward_ms < 50.0  # Fast inference budget

    # Streaming Predictor Latency
    pipe_cfg = PipelineConfig()
    predictor = StreamingSignPredictor(model, config=pipe_cfg, device="cpu")

    # Buffer frames
    dummy_frame = np.random.randn(76, 3)
    for _ in range(pipe_cfg.buffer.window_size):
        res = predictor.process_frame_landmarks(dummy_frame)

    assert "latency_ms" in res or "total_latency_ms" in res
    if "total_latency_ms" in res:
        assert res["total_latency_ms"] > 0
    if "inference_latency_ms" in res:
        assert res["inference_latency_ms"] > 0

    # Regional Synthesis Latency
    synth = RegionalSynthesisEngine(SynthesisConfig(mock_offline=True))
    pipeline_res = synth.process_multilingual_pipeline("TEACHER")

    assert "total_synthesis_latency_ms" in pipeline_res
    assert pipeline_res["total_synthesis_latency_ms"] > 0
    for lang, lres in pipeline_res["languages"].items():
        assert "translation_latency_ms" in lres
        assert lres["translation_latency_ms"] >= 0
        if lres["tts"] is not None:
            assert "latency_ms" in lres["tts"]
            assert lres["tts"]["latency_ms"] >= 0
