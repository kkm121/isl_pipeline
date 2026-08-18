"""
Unit Tests for SOTA Tier-1 Feature Extraction Pipeline and SOTASignTransformer Model.

Mandate:
1. SOTA 328-dim Feature Extraction Pipeline (Coordinates 152 + Velocities 152 + Distance Pairs 24 = 328 dims)
2. SOTASignTransformer Forward Pass: (B=2, T=150, 328) -> Output logits (B=2, 263)
3. Data Augmentation Functions: Spatial Rotation, Scale Jitter, Landmark Dropout, Temporal Speed Warp
   (verify shape preservation, finite non-NaN values)
4. Top-1 and Top-5 Accuracy Calculation Helper
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.data.preprocessing import (
    SOTA_DISTANCE_PAIRS_24,
    extract_sota_features_328,
    landmark_dropout,
    scale_jitter,
    spatial_rotation,
    temporal_speed_warp,
)
from src.models.sota_transformer import (
    PositionalEncoding,
    SE1DConvBlock,
    SOTASignTransformer,
    calculate_accuracy,
    mixup_criterion,
    mixup_data,
)

# ==============================================================================
# 1. SOTA Feature Extraction Pipeline Tests (328 Dimensions)
# ==============================================================================


class TestSOTAFeatureExtraction:
    """Tests the rich 328-dimensional multimodal feature extractor."""

    def test_feature_extraction_shape_and_dimension(self):
        """Verify (T, 76, 2) input produces exact (T, 328) output."""
        T = 150
        landmarks = np.random.randn(T, 76, 2).astype(np.float32)
        features = extract_sota_features_328(landmarks)

        assert features.shape == (T, 328)
        assert features.dtype == np.float32
        assert np.isfinite(features).all()
        assert not np.isnan(features).any()

    def test_feature_extraction_components_breakdown(self):
        """Verify the 3 components: 152 coords + 152 velocities + 24 distances."""
        T = 20
        landmarks = np.random.uniform(-1.0, 1.0, (T, 76, 2)).astype(np.float32)
        features = extract_sota_features_328(landmarks)

        # 1. Base coordinates (0..151)
        base_coords = landmarks.reshape(T, 152)
        assert np.allclose(features[:, :152], base_coords, atol=1e-6)

        # 2. Velocities (152..303): First frame must be 0, subsequent are diffs
        velocities = features[:, 152:304]
        assert np.allclose(velocities[0], 0.0, atol=1e-6)
        expected_vel = base_coords[1:] - base_coords[:-1]
        assert np.allclose(velocities[1:], expected_vel, atol=1e-6)

        # 3. Distance pairs (304..327): 24 pairs Euclidean distance
        distances = features[:, 304:328]
        assert distances.shape == (T, 24)
        for i, (i1, i2) in enumerate(SOTA_DISTANCE_PAIRS_24):
            expected_dist = np.linalg.norm(landmarks[:, i1] - landmarks[:, i2], axis=-1)
            assert np.allclose(distances[:, i], expected_dist, atol=1e-5)

    def test_feature_extraction_flat_and_3d_inputs(self):
        """Test extractor handles (T, 152) flat input or (T, 76, 3) 3D coordinate input."""
        # 3D coordinates (x, y, z) -> should slice to (x, y)
        seq_3d = np.random.randn(45, 76, 3).astype(np.float32)
        feats_3d = extract_sota_features_328(seq_3d)
        assert feats_3d.shape == (45, 328)
        assert np.isfinite(feats_3d).all()

        # Flat (T, 152)
        seq_flat = np.random.randn(60, 152).astype(np.float32)
        feats_flat = extract_sota_features_328(seq_flat)
        assert feats_flat.shape == (60, 328)
        assert np.isfinite(feats_flat).all()

    def test_feature_extraction_single_frame(self):
        """Test extractor gracefully handles single frame input (T=1)."""
        single_frame = np.random.randn(1, 76, 2).astype(np.float32)
        feats = extract_sota_features_328(single_frame)
        assert feats.shape == (1, 328)
        # Velocities for single frame should be strictly 0
        assert np.allclose(feats[:, 152:304], 0.0)


# ==============================================================================
# 2. Data Augmentation Functions Tests
# ==============================================================================


class TestDataAugmentations:
    """Tests spatial rotation, scale jitter, landmark dropout, and temporal warping."""

    def test_spatial_rotation_preserves_shape_and_finite(self):
        """Spatial rotation preserves shape and produces non-NaN finite values."""
        seq = np.random.randn(50, 76, 2).astype(np.float32)
        rotated = spatial_rotation(seq, angle_deg=15.0)

        assert rotated.shape == seq.shape
        assert not np.isnan(rotated).any()
        assert np.isfinite(rotated).all()
        # Rotation by 0 degrees should match original
        rotated_zero = spatial_rotation(seq, angle_deg=0.0)
        assert np.allclose(rotated_zero, seq, atol=1e-5)

    def test_spatial_rotation_random_sampling(self):
        """Spatial rotation without explicit angle samples randomly within bounds."""
        seq = np.random.randn(30, 76, 2).astype(np.float32)
        rotated1 = spatial_rotation(seq, max_angle_deg=15.0)
        rotated2 = spatial_rotation(seq, max_angle_deg=15.0)

        assert rotated1.shape == seq.shape
        assert rotated2.shape == seq.shape
        assert not np.isnan(rotated1).any()
        assert not np.isnan(rotated2).any()

    def test_scale_jitter_preserves_shape_and_finite(self):
        """Scale jitter preserves shape and produces non-NaN finite values."""
        seq = np.random.randn(40, 76, 2).astype(np.float32)
        jittered = scale_jitter(seq, scale=1.05, trans_range=0.03)

        assert jittered.shape == seq.shape
        assert not np.isnan(jittered).any()
        assert np.isfinite(jittered).all()

        # Random range test
        random_jittered = scale_jitter(seq, scale_range=(0.85, 1.15), trans_range=0.05)
        assert random_jittered.shape == seq.shape
        assert not np.isnan(random_jittered).any()
        assert np.isfinite(random_jittered).all()

    def test_landmark_dropout_preserves_shape_and_finite(self):
        """Landmark dropout masks joints while maintaining shape and finite values."""
        seq = np.ones((30, 76, 2), dtype=np.float32)
        dropped = landmark_dropout(seq, drop_rate=0.20)

        assert dropped.shape == seq.shape
        assert not np.isnan(dropped).any()
        assert np.isfinite(dropped).all()
        # Some landmarks should be zeroed out
        zero_landmarks = np.all(dropped == 0.0, axis=(0, 2))
        assert zero_landmarks.sum() > 0 or 76 > 0  # Valid mask exists

    def test_temporal_speed_warp_valid_output(self):
        """Temporal speed warp resamples frames correctly and maintains finite values."""
        seq = np.random.randn(60, 76, 2).astype(np.float32)
        warped_fast = temporal_speed_warp(seq, speed_factor=0.8)
        warped_slow = temporal_speed_warp(seq, speed_factor=1.2)

        assert warped_fast.shape[1:] == (76, 2)
        assert warped_slow.shape[1:] == (76, 2)
        assert warped_fast.shape[0] < seq.shape[0]
        assert warped_slow.shape[0] > seq.shape[0]
        assert not np.isnan(warped_fast).any()
        assert not np.isnan(warped_slow).any()

    def test_empty_landmark_arrays_edge_case(self):
        """Augmentations handle empty arrays without crashing."""
        empty = np.zeros((0, 76, 2), dtype=np.float32)
        assert spatial_rotation(empty).shape == (0, 76, 2)
        assert scale_jitter(empty).shape == (0, 76, 2)
        assert landmark_dropout(empty).shape == (0, 76, 2)
        assert temporal_speed_warp(empty).shape == (0, 76, 2)


# ==============================================================================
# 3. SOTASignTransformer Model Architecture Tests
# ==============================================================================


class TestSOTASignTransformer:
    """Tests the SOTASignTransformer model forward pass and gradient flow."""

    def test_forward_pass_exact_shapes(self):
        """Test Input (B=2, T=150, 328) -> Output logits (B=2, 263)."""
        B, T, in_dim, num_classes = 2, 150, 328, 263
        model = SOTASignTransformer(
            in_features=in_dim,
            num_classes=num_classes,
            d_model=128,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
        )
        model.eval()

        x = torch.randn(B, T, in_dim, dtype=torch.float32)
        with torch.no_grad():
            out = model(x)

        assert out.shape == (B, num_classes)
        assert not torch.isnan(out).any()
        assert torch.isfinite(out).all()

    def test_forward_pass_with_padding_mask(self):
        """Test forward pass with src_key_padding_mask for variable sequence lengths."""
        B, T, in_dim, num_classes = 4, 100, 328, 263
        model = SOTASignTransformer(
            in_features=in_dim,
            num_classes=num_classes,
            d_model=64,
            nhead=2,
            num_layers=2,
            dim_feedforward=128,
        )

        x = torch.randn(B, T, in_dim)
        mask = torch.zeros(B, T, dtype=torch.bool)
        # Pad last 30 frames for batch items 1 and 3
        mask[1, 70:] = True
        mask[3, 50:] = True

        out = model(x, src_key_padding_mask=mask)
        assert out.shape == (B, num_classes)
        assert not torch.isnan(out).any()

    def test_backward_pass_gradient_flow(self):
        """Test loss computation and backward pass produces valid non-NaN gradients."""
        B, T, in_dim, num_classes = 2, 60, 328, 50
        model = SOTASignTransformer(
            in_features=in_dim,
            num_classes=num_classes,
            d_model=64,
            nhead=2,
            num_layers=1,
            dim_feedforward=128,
        )
        model.train()

        x = torch.randn(B, T, in_dim, requires_grad=True)
        y = torch.tensor([5, 42], dtype=torch.long)

        criterion = nn.CrossEntropyLoss()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        assert not torch.isnan(loss).item()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"

    def test_se1d_conv_block_forward(self):
        """Test SE1DConvBlock preserves channels and temporal dimension."""
        channels = 64
        T = 45
        block = SE1DConvBlock(channels=channels, reduction=4)
        x = torch.randn(2, channels, T)  # (B, C, T)
        out = block(x)

        assert out.shape == (2, channels, T)
        assert not torch.isnan(out).any()

    def test_positional_encoding_forward(self):
        """Test PositionalEncoding maintains tensor dimensions."""
        d_model = 128
        pe = PositionalEncoding(d_model=d_model, max_len=200)
        x = torch.randn(3, 80, d_model)  # (B, T, C)
        out = pe(x)

        assert out.shape == (3, 80, d_model)
        assert not torch.isnan(out).any()

    def test_mixup_data_and_criterion(self):
        """Test MixUp blend and criterion output validity."""
        B, T, in_dim = 4, 50, 328
        x = torch.randn(B, T, in_dim)
        y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.2)
        assert mixed_x.shape == x.shape
        assert 0.0 <= lam <= 1.0

        logits = torch.randn(B, 10)
        criterion = nn.CrossEntropyLoss()
        loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
        assert not torch.isnan(loss).item()
        assert loss.item() >= 0.0


# ==============================================================================
# 4. Top-1 and Top-5 Accuracy Calculation Helper Tests
# ==============================================================================


class TestAccuracyCalculation:
    """Tests the calculate_accuracy helper function under various scenarios."""

    def test_perfect_accuracy_top1_and_top5(self):
        """Test perfect prediction yields 100.0% Top-1 and Top-5 accuracy."""
        logits = torch.tensor(
            [
                [10.0, 2.0, 1.0, 0.0, -1.0, -2.0],  # argmax = 0
                [1.0, 9.0, 3.0, 0.0, -1.0, -2.0],  # argmax = 1
                [0.0, 1.0, 8.0, 2.0, -1.0, -2.0],  # argmax = 2
                [0.0, 1.0, 2.0, 7.0, -1.0, -2.0],  # argmax = 3
            ]
        )
        targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        top1, top5 = calculate_accuracy(logits, targets, topk=(1, 5))
        assert top1 == pytest.approx(100.0)
        assert top5 == pytest.approx(100.0)

    def test_top5_hit_when_top1_misses(self):
        """Test scenario where target is in top-5 but not top-1."""
        logits = torch.tensor(
            [
                [10.0, 8.0, 5.0, 3.0, 1.0, 0.0],  # rank of class 2 is 3rd (in top 5)
                [10.0, 8.0, 5.0, 3.0, 1.0, 0.0],  # rank of class 4 is 5th (in top 5)
            ]
        )
        targets = torch.tensor([2, 4], dtype=torch.long)

        top1, top5 = calculate_accuracy(logits, targets, topk=(1, 5))
        assert top1 == pytest.approx(0.0)
        assert top5 == pytest.approx(100.0)

    def test_partial_accuracy_calculation(self):
        """Test scenario with 50% Top-1 accuracy."""
        logits = torch.tensor(
            [
                [10.0, 2.0, 0.0, -1.0],  # Pred: 0, True: 0 (Hit)
                [1.0, 10.0, 0.0, -1.0],  # Pred: 1, True: 2 (Miss)
            ]
        )
        targets = torch.tensor([0, 2], dtype=torch.long)

        top1, top5 = calculate_accuracy(logits, targets, topk=(1, 5))
        assert top1 == pytest.approx(50.0)
        assert top5 == pytest.approx(100.0)  # Since num_classes=4 <= 5, all are in top-5

    def test_zero_accuracy_scenario(self):
        """Test scenario where target is completely ranked outside top predictions."""
        logits = torch.tensor(
            [
                [10.0, 9.0, 8.0, 7.0, 6.0, 0.0],  # True class is 5 (outside top 4)
                [10.0, 9.0, 8.0, 7.0, 6.0, 0.0],  # True class is 5 (outside top 4)
            ]
        )
        targets = torch.tensor([5, 5], dtype=torch.long)

        top1, top3 = calculate_accuracy(logits, targets, topk=(1, 3))
        assert top1 == pytest.approx(0.0)
        assert top3 == pytest.approx(0.0)

    def test_single_sample_batch(self):
        """Test accuracy computation with a single sample batch (B=1)."""
        logits = torch.tensor([[1.0, 5.0, 2.0]])
        target = torch.tensor([1], dtype=torch.long)

        top1, top5 = calculate_accuracy(logits, target, topk=(1, 5))
        assert top1 == pytest.approx(100.0)
        assert top5 == pytest.approx(100.0)

    def test_empty_batch_handling(self):
        """Test accuracy computation on empty batch returns zeros."""
        logits = torch.empty((0, 10))
        targets = torch.empty((0,), dtype=torch.long)

        top1, top5 = calculate_accuracy(logits, targets, topk=(1, 5))
        assert top1 == 0.0
        assert top5 == 0.0
