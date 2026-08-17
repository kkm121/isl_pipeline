from dataclasses import dataclass, field
from typing import List

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
    use_signer_disjoint: bool = True


@dataclass
class ModelConfig:
    input_size: int = 63
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    attention: bool = True
    num_classes: int = 26
    architecture: str = "bilstm_attention"


@dataclass
class Tier1ModelConfig(ModelConfig):
    """Specialized configuration for Tier-1 (Demo Track) Classifiers."""

    input_size: int = 152  # 76 * 2 (PoseStitch-SLT / MediaPipe Holistic)
    hidden_size: int = 256
    num_classes: int = 26
    architecture: str = "temporal_cnn"
    cnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 5, 7])
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    temporal_pooling: str = "avg_max"  # "avg", "max", "avg_max"


@dataclass
class Tier2SignFormerConfig:
    """Specialized configuration for Tier-2 (Research Track) Continuous Translation."""

    num_nodes: int = 76
    in_channels: int = 2  # (x, y)
    graph_hidden_dim: int = 128
    transformer_d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    use_euclidean_attention: bool = True
    vocab_size: int = 5000
    max_target_len: int = 64


@dataclass
class BufferConfig:
    """Configuration for real-time temporal buffering and UI states."""

    window_size: int = 45  # 1.5s buffer @ 30fps
    step_size: int = 5  # Sliding stride (every 5 frames = ~166ms)
    min_confidence: float = 0.65  # Threshold to trigger sign detection
    consensus_frames: int = 3  # Consecutive consistent predictions required for output
    idle_reset_seconds: float = 2.0  # Reset buffer after inactivity


@dataclass
class SynthesisConfig:
    """Configuration for AI4Bharat IndicTrans2 NMT & VITS/Rasa TTS."""

    source_language: str = "eng_Latn"
    target_languages: List[str] = field(
        default_factory=lambda: ["hin_Deva", "tam_Taml", "tel_Telu", "ben_Beng", "mar_Mrai", "kan_Knda"]
    )
    nmt_model_name: str = "ai4bharat/indictrans2-en-indic-dist-200M"
    tts_model_name: str = "ai4bharat/vits_indic"
    enable_tts: bool = True
    mock_offline: bool = True  # Fallback to local deterministic synth in sandbox/tests


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
    buffer: BufferConfig = field(default_factory=BufferConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    tier2: Tier2SignFormerConfig = field(default_factory=Tier2SignFormerConfig)
    experiment_name: str = "tier1_classroom_demo"

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}
        config = cls()
        if "data" in data:
            config.data = DataConfig(**data["data"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "buffer" in data:
            config.buffer = BufferConfig(**data["buffer"])
        if "synthesis" in data:
            config.synthesis = SynthesisConfig(**data["synthesis"])
        if "tier2" in data:
            config.tier2 = Tier2SignFormerConfig(**data["tier2"])
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
                    "buffer": self.buffer.__dict__,
                    "synthesis": self.synthesis.__dict__,
                    "tier2": self.tier2.__dict__,
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
