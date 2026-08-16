import pytest
import torch

from src.data.dataset import ISLDataModule
from src.models.classifier import ISLClassifier
from src.models.config import DataConfig, ModelConfig, PipelineConfig, TrainingConfig
from src.training.evaluate import compute_confusion_matrix, evaluate, per_class_metrics
from src.training.trainer import Trainer
from src.utils.metrics import MetricsTracker, RunningAverage


@pytest.fixture
def configs():
    data_cfg = DataConfig(data_dir=".", num_classes=5)
    model_cfg = ModelConfig(input_size=63, hidden_size=16, num_layers=1, num_classes=5)
    train_cfg = TrainingConfig(batch_size=4, epochs=1)
    pipe_cfg = PipelineConfig(experiment_name="test", data=data_cfg, model=model_cfg, training=train_cfg)
    return pipe_cfg


@pytest.fixture
def datamodule(configs):
    dm = ISLDataModule(configs.data)
    dm.create_synthetic(num_samples=20)
    return dm


def test_trainer_one_epoch(configs, datamodule, tmp_path):
    configs.training.checkpoint_dir = str(tmp_path)
    model = ISLClassifier(configs.model)
    trainer = Trainer(model, datamodule, configs.training)
    trainer.train()
    assert len(trainer.history["train_loss"]) > 0


def test_trainer_loss_decreases(configs, datamodule, tmp_path):
    configs.training.epochs = 15
    configs.training.learning_rate = 0.05
    configs.training.checkpoint_dir = str(tmp_path)
    model = ISLClassifier(configs.model)
    trainer = Trainer(model, datamodule, configs.training)
    trainer.train()
    history = trainer.history["train_loss"]
    assert len(history) == 15
    assert history[0] > history[-1]


def test_trainer_checkpoint_saved(configs, datamodule, tmp_path):
    configs.training.checkpoint_dir = str(tmp_path)
    model = ISLClassifier(configs.model)
    trainer = Trainer(model, datamodule, configs.training)
    trainer.train()
    checkpoints = list(tmp_path.glob("*.pt"))
    assert len(checkpoints) > 0


def test_trainer_nan_detection(configs, datamodule, tmp_path):
    configs.training.checkpoint_dir = str(tmp_path)
    model = ISLClassifier(configs.model)
    trainer = Trainer(model, datamodule, configs.training)

    # Force a NaN loss scenario
    class NanModel(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.size(0), configs.model.num_classes), float("nan"), requires_grad=True)

    trainer.model = NanModel()

    # Trainer should handle the exception or stop
    with pytest.raises(RuntimeError):
        trainer.train()


def test_evaluate_synthetic(configs, datamodule):
    model = ISLClassifier(configs.model)
    metrics = evaluate(model, datamodule.val_dataloader(), configs.training.device)
    assert "loss" in metrics
    assert "accuracy" in metrics


def test_confusion_matrix():
    preds = torch.tensor([0, 1, 1, 2, 2])
    labels = torch.tensor([0, 1, 2, 2, 2])
    cm = compute_confusion_matrix(preds, labels, num_classes=3)
    assert cm.shape == (3, 3)
    assert cm[2, 1] == 1  # true 2, pred 1


def test_per_class_metrics():
    preds = torch.tensor([0, 1, 1, 2, 2])
    labels = torch.tensor([0, 1, 2, 2, 2])
    metrics = per_class_metrics(preds, labels, num_classes=3)
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics


def test_metrics_tracker(tmp_path):
    tracker = MetricsTracker()
    tracker.update("acc", 0.5)
    tracker.update("acc", 0.8)
    tracker.save(tmp_path / "metrics.json")

    tracker2 = MetricsTracker()
    tracker2.load(tmp_path / "metrics.json")
    assert tracker2.metrics["acc"] == [0.5, 0.8]


def test_running_average():
    avg = RunningAverage()
    avg.update(2.0)
    avg.update(4.0)
    assert avg.compute() == 3.0
