import pytest
import torch

from src.models.classifier import Attention, ISLClassifier
from src.models.config import ModelConfig


@pytest.fixture
def model_config():
    return ModelConfig(
        input_size=63,
        hidden_size=32,
        num_layers=1,
        dropout=0.1,
        bidirectional=True,
        attention=True,
        num_classes=10,
    )


def test_attention_forward():
    attention = Attention(hidden_size=64)  # 32 * 2 (bidirectional)
    x = torch.rand(4, 10, 64)
    out, weights = attention(x)
    assert out.shape == (4, 64)
    assert weights.shape == (4, 10, 1)


def test_attention_weights_sum():
    attention = Attention(hidden_size=64)
    x = torch.rand(4, 10, 64)
    _, weights = attention(x)
    sums = weights.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums))


def test_classifier_forward(model_config):
    model = ISLClassifier(model_config)
    x = torch.rand(4, 10, 63)
    out = model(x)
    assert out.shape == (4, 10)


def test_classifier_forward_different_seq_lengths(model_config):
    model = ISLClassifier(model_config)
    x1 = torch.rand(4, 5, 63)
    x2 = torch.rand(4, 15, 63)
    out1 = model(x1)
    out2 = model(x2)
    assert out1.shape == (4, 10)
    assert out2.shape == (4, 10)


def test_classifier_no_attention(model_config):
    model_config.attention = False
    model = ISLClassifier(model_config)
    x = torch.rand(4, 10, 63)
    out = model(x)
    assert out.shape == (4, 10)


def test_classifier_bidirectional(model_config):
    model_config.bidirectional = False
    model = ISLClassifier(model_config)
    x = torch.rand(4, 10, 63)
    out = model(x)
    assert out.shape == (4, 10)


def test_classifier_output_logits(model_config):
    model = ISLClassifier(model_config)
    x = torch.rand(4, 10, 63)
    out = model(x)
    # Check that they don't strictly sum to 1 like probabilities
    sums = out.sum(dim=1)
    assert not torch.allclose(sums, torch.ones_like(sums))


def test_classifier_count_parameters(model_config):
    model = ISLClassifier(model_config)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert params > 0


def test_classifier_save_load(model_config, tmp_path):
    model = ISLClassifier(model_config)
    model.eval()
    x = torch.rand(4, 10, 63)
    out1 = model(x)

    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    model2 = ISLClassifier(model_config)
    model2.load_state_dict(torch.load(path))
    model2.eval()
    out2 = model2(x)

    torch.testing.assert_close(out1, out2)


def test_classifier_gradient_flow(model_config):
    model = ISLClassifier(model_config)
    x = torch.rand(4, 10, 63)
    y = torch.randint(0, 10, (4,))

    out = model(x)
    loss = torch.nn.functional.cross_entropy(out, y)
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"No gradient for {name}"
        assert param.grad.abs().sum().item() > 0, f"Zero gradient for {name}"
