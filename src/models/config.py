from dataclasses import dataclass, field

import yaml


@dataclass
class DataConfig:
    data_dir: str = "data/"
    num_landmarks: int = 21
    landmark_dim: int = 3
    sequence_length: int = 30
    num_classes: int = 26
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42


@dataclass
class ModelConfig:
    input_size: int = 63
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    attention: bool = True
    num_classes: int = 26


@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    patience: int = 15
    gradient_clip: float = 1.0
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints/"
    log_dir: str = "logs/local/"
    device: str = "auto"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_name: str = "default"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        config = cls()
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "experiment_name" in data:
            config.experiment_name = data["experiment_name"]
        return config

    def to_yaml(self, path: str):
        with open(path, "w") as f:
            yaml.dump(
                {
                    "data": self.data.__dict__,
                    "model": self.model.__dict__,
                    "training": self.training.__dict__,
                    "experiment_name": self.experiment_name,
                },
                f,
            )

    def resolve_device(self) -> str:
        import torch

        if self.training.device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return self.training.device
