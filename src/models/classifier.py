import logging
from pathlib import Path

import torch
from torch import nn

from src.models.config import ModelConfig

logger = logging.getLogger(__name__)


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output: torch.Tensor):
        # lstm_output: (batch, seq_len, hidden_size)
        weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch, seq_len, 1)
        context = torch.sum(weights * lstm_output, dim=1)  # (batch, hidden_size)
        return context, weights


class ISLClassifier(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Sequential(
            nn.Linear(config.input_size, config.hidden_size),
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
        )

        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size // 2 if config.bidirectional else config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=config.bidirectional,
        )

        if config.attention:
            self.attention = Attention(config.hidden_size)

        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        x = self.input_proj(x)
        out, _ = self.lstm(x)

        if self.config.attention:
            out, _ = self.attention(out)
        else:
            out = out[:, -1, :]

        return self.classifier(out)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": self.config.__dict__}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "ISLClassifier":
        checkpoint = torch.load(path, map_location=device)
        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return model
