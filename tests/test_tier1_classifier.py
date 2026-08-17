import numpy as np
import torch

from src.data.dataset import (
    ISLDataModule,
    SignerDisjointSplitter,
)
from src.data.preprocessing import (
    extract_2d_pose_vector,
    interpolate_missing_landmarks,
    normalize_landmarks,
)
from src.inference.predict import StreamingSignPredictor, UIStreamState
from src.inference.translation_tts import RegionalSynthesisEngine
from src.models.classifier import (
    ISLClassifier,
    Tier1TemporalCNN,
    create_tier1_classifier,
)
from src.models.config import (
    BufferConfig,
    DataConfig,
    ModelConfig,
    PipelineConfig,
    SynthesisConfig,
    Tier1ModelConfig,
    Tier2SignFormerConfig,
)
from src.models.signformer_gcn import (
    EuclideanSelfAttention,
    SignFormerGCN,
    build_76_keypoint_adjacency,
)

# ==============================================================================
# 1. 76-Keypoint Preprocessing & Normalization Tests
# ==============================================================================


def test_76_keypoint_normalization():
    # 76 keypoints: (T, 76, 3)
    seq = np.random.randn(30, 76, 3) + 10.0
    norm = normalize_landmarks(seq)
    assert norm.shape == (30, 76, 3)
    assert np.max(np.abs(norm)) <= 1.0001


def test_2d_pose_vector_extraction():
    seq = np.random.randn(45, 76, 3)
    xy = extract_2d_pose_vector(seq)
    assert xy.shape == (45, 152)  # 76 * 2


def test_temporal_interpolation_missing_frames():
    seq = np.ones((10, 76, 3))
    # Artificially blank out frames 3, 4
    seq[3, :, :] = 0.0
    seq[4, :, :] = 0.0
    interpolated = interpolate_missing_landmarks(seq)
    assert not np.all(interpolated[3] == 0.0)
    assert np.allclose(interpolated[3], 1.0)


# ==============================================================================
# 2. Signer-Disjoint Splitting Tests
# ==============================================================================


def test_signer_disjoint_splitting_zero_leakage():
    signer_ids = [f"signer_{i % 8}" for i in range(160)]
    train_idx, val_idx, test_idx = SignerDisjointSplitter.split_by_signer(
        signer_ids, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42
    )

    train_signers = {signer_ids[i] for i in train_idx}
    val_signers = {signer_ids[i] for i in val_idx}
    test_signers = {signer_ids[i] for i in test_idx}

    # Verify strictly zero intersection across all splits
    assert len(train_signers & val_signers) == 0
    assert len(train_signers & test_signers) == 0
    assert len(val_signers & test_signers) == 0
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(signer_ids)


def test_datamodule_synthetic_signer_disjoint():
    cfg = DataConfig(num_landmarks=76, landmark_dim=2, num_classes=50, sequence_length=45)
    dm = ISLDataModule(cfg)
    train_ds, val_ds, test_ds = dm.create_synthetic(n_samples=100, num_signers=6)

    assert len(train_ds) > 0
    assert len(val_ds) > 0
    assert len(test_ds) > 0


# ==============================================================================
# 3. Tier-1 Model Architecture Tests
# ==============================================================================


def test_tier1_temporal_cnn_forward():
    cfg = Tier1ModelConfig(input_size=152, hidden_size=128, num_classes=50)
    model = Tier1TemporalCNN(cfg)

    x = torch.randn(4, 45, 152)  # (batch, seq_len, 152)
    out = model(x)
    assert out.shape == (4, 50)


def test_tier1_classifier_factory():
    cnn_cfg = Tier1ModelConfig(input_size=152, architecture="temporal_cnn", num_classes=26)
    lstm_cfg = ModelConfig(input_size=152, architecture="bilstm_attention", num_classes=26)

    cnn_model = create_tier1_classifier(cnn_cfg)
    lstm_model = create_tier1_classifier(lstm_cfg)

    assert isinstance(cnn_model, Tier1TemporalCNN)
    assert isinstance(lstm_model, ISLClassifier)


def test_tier1_model_save_and_load(tmp_path):
    cfg = Tier1ModelConfig(input_size=152, hidden_size=64, num_classes=10)
    model = Tier1TemporalCNN(cfg)
    model.eval()
    save_path = str(tmp_path / "tier1_model.pt")

    model.save(save_path)
    loaded_model = Tier1TemporalCNN.load(save_path)
    loaded_model.eval()

    x = torch.randn(2, 45, 152)
    with torch.no_grad():
        out1 = model(x)
        out2 = loaded_model(x)
    assert torch.allclose(out1, out2, atol=1e-5)


# ==============================================================================
# 4. Streaming Buffer & Temporal Smoothing State Machine Tests
# ==============================================================================


def test_streaming_predictor_states():
    cfg = Tier1ModelConfig(input_size=152, hidden_size=64, num_classes=10)
    model = Tier1TemporalCNN(cfg)
    pipe_cfg = PipelineConfig(buffer=BufferConfig(window_size=10, step_size=2, consensus_frames=2, min_confidence=0.1))
    predictor = StreamingSignPredictor(model, config=pipe_cfg, device="cpu")

    # Initial state
    assert predictor.state == UIStreamState.IDLE

    # Push frames to fill buffer
    dummy_frame_kp = np.random.randn(76, 3)
    for _ in range(5):
        res = predictor.process_frame_landmarks(dummy_frame_kp)
        assert res["state"] == UIStreamState.BUFFERING.value
        assert res["buffer_fill_ratio"] < 1.0

    # Fill to window size (10)
    for _ in range(5):
        res = predictor.process_frame_landmarks(dummy_frame_kp)

    # Now buffer is full -> should transition to PROCESSING or PREDICTED
    assert res["buffer_fill_ratio"] == 1.0
    assert res["prediction"] is not None


# ==============================================================================
# 5. Multilingual Regional Synthesis & TTS Tests
# ==============================================================================


def test_regional_synthesis_engine():
    engine = RegionalSynthesisEngine(SynthesisConfig(mock_offline=True))

    # Test dictionary translation
    res_hi = engine.translate_text("TEACHER", target_lang="hin_Deva")
    assert "अध्यापक" in res_hi["translated_text"]

    # Test TTS audio synthesis contract
    tts_res = engine.synthesize_speech("अध्यापक", target_lang="hin_Deva")
    assert tts_res["audio_format"] == "wav"
    assert tts_res["audio_duration_sec"] > 0

    # Test full pipeline
    multi_res = engine.process_multilingual_pipeline("HELP", target_languages=["hin_Deva", "tam_Taml"])
    assert "hin_Deva" in multi_res["languages"]
    assert "tam_Taml" in multi_res["languages"]


# ==============================================================================
# 6. Tier-2 SignFormer-GCN & Euclidean Attention Tests
# ==============================================================================


def test_76_keypoint_adjacency():
    adj = build_76_keypoint_adjacency()
    assert adj.shape == (76, 76)
    assert not torch.isnan(adj).any()


def test_euclidean_self_attention():
    attn = EuclideanSelfAttention(d_model=64, nhead=4)
    x = torch.randn(2, 20, 64)
    out = attn(x)
    assert out.shape == (2, 20, 64)


def test_signformer_gcn_forward():
    cfg = Tier2SignFormerConfig(
        num_nodes=76,
        in_channels=2,
        graph_hidden_dim=32,
        transformer_d_model=64,
        nhead=4,
        num_encoder_layers=1,
        vocab_size=100,
    )
    model = SignFormerGCN(cfg)
    x = torch.randn(2, 15, 76, 2)  # (batch, seq_len, 76 nodes, 2 coords)
    logits = model(x)
    assert logits.shape == (2, 15, 100)
