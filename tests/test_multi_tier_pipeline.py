"""Unit and Integration Tests for 3-Tier Multi-Tier Pipeline.

Covers:
1. Tier-1 Temporal CNN feature extractor forward shape (Batch=2, T=150, Feat=152) -> (Batch=2, NumClasses=263)
2. Tier-2 SignFormer GCN spatial-temporal forward shape (Batch=2, T=45, V=76, C=2) -> (Batch=2, VocabSize=200)
3. Tier-3 Conversational ISL Chatbot Engine multilingual dialogue across Hindi, Tamil, Telugu, Bengali, and English with audio synthesis payloads
4. Euclidean Self-Attention numerical stability and non-NaN guarantees under zero-padded sequences
"""

import torch

from src.inference.translation_tts import (
    ConversationalISLChatbotEngine,
    RegionalSynthesisEngine,
)
from src.models.classifier import Tier1TemporalCNN
from src.models.config import (
    SynthesisConfig,
    Tier1ModelConfig,
    Tier2SignFormerConfig,
)
from src.models.signformer_gcn import EuclideanSelfAttention, SignFormerGCN


# ==============================================================================
# 1. Tier-1 Temporal CNN Forward Shape Test
# ==============================================================================
def test_tier1_temporal_cnn_forward_shape():
    """Verifies input shape (Batch=2, T=150, Feat=152) -> Output shape (Batch=2, NumClasses=263)."""
    batch_size = 2
    seq_len = 150
    input_dim = 152  # 76 keypoints * 2 (x, y)
    num_classes = 263  # 263 ISL lexicon categories

    cfg = Tier1ModelConfig(
        input_size=input_dim,
        num_classes=num_classes,
        hidden_size=256,
        cnn_channels=[64, 128, 256],
        temporal_pooling="avg_max",
        dropout=0.2,
    )
    model = Tier1TemporalCNN(config=cfg)
    model.eval()

    # Input tensor shape: (Batch=2, T=150, Feat=152)
    x = torch.randn(batch_size, seq_len, input_dim)

    with torch.no_grad():
        out = model(x)

    # 1. Verify exact output shape
    assert out.shape == (batch_size, num_classes), (
        f"Expected output shape ({batch_size}, {num_classes}), got {out.shape}"
    )

    # 2. Verify all output activations are finite and non-NaN
    assert torch.isfinite(out).all(), "Output contains NaN or Inf values"
    assert not torch.isnan(out).any(), "Output contains NaN values"

    # 3. Verify distinct predictions for different batch entries
    assert not torch.allclose(out[0], out[1]), "Batch entries produced identical outputs"


# ==============================================================================
# 2. Tier-2 SignFormer GCN Forward Shape Test
# ==============================================================================
def test_tier2_signformer_gcn_forward_shape():
    """Verifies input shape (Batch=2, T=45, V=76, C=2) -> Output shape (Batch=2, VocabSize=200)."""
    batch_size = 2
    seq_len = 45
    num_nodes = 76  # 76 holistic skeleton keypoints
    in_channels = 2  # 2D coordinates (x, y)
    vocab_size = 200  # 200 Continuous ISL sentence/gloss vocabulary

    cfg = Tier2SignFormerConfig(
        num_nodes=num_nodes,
        in_channels=in_channels,
        graph_hidden_dim=64,
        transformer_d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        vocab_size=vocab_size,
        max_target_len=45,
    )
    model = SignFormerGCN(config=cfg)
    model.eval()

    # Input tensor: (Batch=2, T=45, V=76, C=2)
    src = torch.randn(batch_size, seq_len, num_nodes, in_channels)

    with torch.no_grad():
        # Encode spatial-temporal graph features
        memory = model.encode(src)  # (Batch=2, T=45, d_model=128)
        assert memory.shape == (batch_size, seq_len, cfg.transformer_d_model)

        # Sequence translation head pooled across temporal dimension
        logits = model.translation_head(memory.mean(dim=1))  # (Batch=2, VocabSize=200)

        # Also test full sequence classification pass
        seq_logits = model(src)  # (Batch=2, T=45, VocabSize=200)

    # 1. Verify pooled output shape
    assert logits.shape == (batch_size, vocab_size), (
        f"Expected pooled output shape ({batch_size}, {vocab_size}), got {logits.shape}"
    )

    # 2. Verify sequence output shape
    assert seq_logits.shape == (batch_size, seq_len, vocab_size), (
        f"Expected sequence output shape ({batch_size}, {seq_len}, {vocab_size}), got {seq_logits.shape}"
    )

    # 3. Verify numerical stability
    assert torch.isfinite(logits).all(), "Logits contain NaN or Inf values"
    assert not torch.isnan(logits).any(), "Logits contain NaN values"


# ==============================================================================
# 3. Tier-3 Conversational Chatbot Engine Multilingual Dialogue Test
# ==============================================================================
def test_tier3_chatbot_dialogue_multilingual():
    """Verifies that ConversationalISLChatbotEngine correctly handles queries across

    Hindi, Tamil, Telugu, Bengali, and English, returning structured dialogue
    packets with speech audio payloads.
    """
    synth = RegionalSynthesisEngine(SynthesisConfig(mock_offline=True))
    chatbot = ConversationalISLChatbotEngine(synthesis_engine=synth)

    target_languages = [
        ("hin_Deva", "HELLO", "Hindi (Devanagari)"),
        ("tam_Taml", "HELP", "Tamil"),
        ("tel_Telu", "TEACHER", "Telugu"),
        ("ben_Beng", "NEWS", "Bengali"),
        ("eng_Latn", "THANK YOU", "English"),
        ("hin", "SCHOOL", "Hindi short-code"),
        ("tam", "HOSPITAL", "Tamil short-code"),
        ("tel", "GOODBYE", "Telugu short-code"),
        ("ben", "HELP", "Bengali short-code"),
        ("eng", "HELLO", "English short-code"),
    ]

    for lang_code, query_sign, lang_desc in target_languages:
        dialogue_packet = chatbot.process_sign_dialogue(
            sign_sentence=query_sign,
            target_lang=lang_code,
        )

        # 1. Verify structured packet keys
        assert isinstance(dialogue_packet, dict), f"Failed for {lang_desc}: Expected dict"
        required_keys = [
            "user_sign_input",
            "bot_reply_english",
            "bot_reply_regional",
            "target_language",
            "tts_audio",
            "timestamp",
        ]
        for key in required_keys:
            assert key in dialogue_packet, f"Missing key '{key}' in dialogue packet for {lang_desc}"

        # 2. Verify sign input and replies
        assert dialogue_packet["user_sign_input"] == query_sign
        assert len(dialogue_packet["bot_reply_english"]) > 0
        assert len(dialogue_packet["bot_reply_regional"]) > 0

        # 3. Verify speech audio payload structure
        audio_payload = dialogue_packet["tts_audio"]
        assert audio_payload is not None, f"TTS audio payload missing for {lang_desc}"
        assert isinstance(audio_payload, dict), f"TTS audio payload is not a dict for {lang_desc}"
        assert audio_payload.get("audio_format") == "wav"
        assert audio_payload.get("sampling_rate") == 22050
        assert audio_payload.get("audio_duration_sec", 0.0) > 0.0
        assert audio_payload.get("latency_ms", 0.0) >= 0.0

    # 4. Verify dialogue memory history tracking
    assert len(chatbot.dialogue_memory) == len(target_languages)

    # 5. Verify memory reset functionality
    chatbot.reset_dialogue_history()
    assert len(chatbot.dialogue_memory) == 0


# ==============================================================================
# 4. Euclidean Self-Attention Stability Test (Zero-Padded Sequences)
# ==============================================================================
def test_euclidean_attention_stability():
    """Verifies that EuclideanSelfAttention outputs finite non-NaN activations

    even with zero-padded sequences.
    """
    d_model = 64
    nhead = 4
    attn = EuclideanSelfAttention(d_model=d_model, nhead=nhead, dropout=0.0)
    attn.eval()

    # Case 1: Fully zero-padded sequence (e.g., missing video frames / zero sensor input)
    x_all_zeros = torch.zeros(2, 30, d_model)
    with torch.no_grad():
        out_zeros = attn(x_all_zeros)

    assert out_zeros.shape == (2, 30, d_model)
    assert torch.isfinite(out_zeros).all(), "Zero sequence produced non-finite or NaN output"
    assert not torch.isnan(out_zeros).any(), "Zero sequence produced NaN output"

    # Case 2: Partially zero-padded sequence (e.g., padded batch)
    x_partially_padded = torch.randn(4, 50, d_model)
    x_partially_padded[:, 25:, :] = 0.0  # Zero out second half of sequence
    with torch.no_grad():
        out_partial = attn(x_partially_padded)

    assert out_partial.shape == (4, 50, d_model)
    assert torch.isfinite(out_partial).all(), "Partially padded sequence produced non-finite output"
    assert not torch.isnan(out_partial).any(), "Partially padded sequence produced NaN output"

    # Case 3: Zero-padded input with explicit causal / padding attention mask
    seq_len = 20
    x_masked = torch.zeros(2, seq_len, d_model)
    # Mask where valid positions are 1 and padded positions are 0
    mask = torch.ones(2, 1, seq_len, seq_len)
    mask[:, :, :, seq_len // 2 :] = 0.0

    with torch.no_grad():
        out_masked = attn(x_masked, mask=mask)

    assert out_masked.shape == (2, seq_len, d_model)
    # Positions with valid mask must remain strictly finite and non-NaN
    assert torch.isfinite(out_masked[:, : seq_len // 2, :]).all()
    assert not torch.isnan(out_masked[:, : seq_len // 2, :]).any()

    # Case 4: Gradient backpropagation stability check with zero-padded input
    attn.train()
    x_train = torch.zeros(2, 15, d_model, requires_grad=True)
    out_train = attn(x_train)
    loss = out_train.sum()
    loss.backward()

    assert x_train.grad is not None
    assert torch.isfinite(x_train.grad).all(), "Gradients with respect to zero-padded input contain NaNs"
    assert not torch.isnan(x_train.grad).any(), "Gradients with respect to zero-padded input contain NaNs"
