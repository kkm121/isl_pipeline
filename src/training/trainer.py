import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from src.models.classifier import ISLClassifier
from src.models.config import PipelineConfig, TrainingConfig
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model: ISLClassifier,
        datamodule_or_config: Any = None,
        config_or_device: Any = None,
        device: str = "cpu",
    ):
        self.datamodule = None
        if hasattr(datamodule_or_config, "get_dataloaders"):
            self.datamodule = datamodule_or_config
            raw_cfg = config_or_device
        else:
            raw_cfg = datamodule_or_config

        self.config: Optional[PipelineConfig] = None
        if isinstance(raw_cfg, PipelineConfig):
            self.config = raw_cfg
            self.training_config = raw_cfg.training
        elif isinstance(raw_cfg, TrainingConfig):
            self.training_config = raw_cfg
        else:
            self.training_config = TrainingConfig()

        dev = device if isinstance(device, str) else "cpu"
        if isinstance(config_or_device, str):
            dev = config_or_device

        self.device = dev
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(self.training_config.epochs, 1)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=self.training_config.mixed_precision and torch.cuda.is_available())

        log_dir = getattr(self.training_config, "log_dir", "logs/local")
        exp_name = getattr(self.config, "experiment_name", "default") if self.config else "default"
        self.metrics = MetricsTracker(log_dir, exp_name)
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        checkpoint_dir = getattr(self.training_config, "checkpoint_dir", "checkpoints/")
        try:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        except OSError:
            self.training_config.checkpoint_dir = "/tmp/checkpoints"
            Path(self.training_config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _check_nan(self, loss: torch.Tensor) -> bool:
        return bool(torch.isnan(loss).any().item() or torch.isinf(loss).any().item())

    def _get_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)

    def _train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            with autocast(enabled=self.training_config.mixed_precision and torch.cuda.is_available()):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

            if self._check_nan(loss):
                raise RuntimeError("NaN loss encountered during training")

            self.scaler.scale(loss).backward()

            self.scaler.unscale_(self.optimizer)
            grad_norm = self._get_grad_norm()
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                raise RuntimeError("NaN gradient encountered during training")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.training_config.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": correct / max(total, 1),
        }

    def _validate(self, loader: DataLoader) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": correct / max(total, 1),
        }

    def _save_checkpoint(self, epoch: int, metrics: dict, is_best: bool = False) -> None:
        ckpt_dir = getattr(self.training_config, "checkpoint_dir", "checkpoints/")
        exp_name = getattr(self.config, "experiment_name", "default") if self.config else "default"
        ckpt_path = Path(ckpt_dir) / f"{exp_name}_latest.pt"
        if hasattr(self.model, "save"):
            self.model.save(str(ckpt_path))
        else:
            torch.save(self.model.state_dict(), str(ckpt_path))

        if is_best:
            best_path = Path(ckpt_dir) / f"{exp_name}_best.pt"
            if hasattr(self.model, "save"):
                self.model.save(str(best_path))
            else:
                torch.save(self.model.state_dict(), str(best_path))

    def train(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        if train_loader is None or val_loader is None:
            if self.datamodule is not None:
                train_loader, val_loader, _ = self.datamodule.get_dataloaders(self.training_config.batch_size)
            else:
                raise ValueError("Must provide train_loader and val_loader or initialize Trainer with datamodule")

        best_val_loss = float("inf")
        patience_counter = 0
        epochs = self.training_config.epochs

        for epoch in range(1, epochs + 1):
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            self.scheduler.step()

            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            self.metrics.log(epoch, train_metrics, "train")
            self.metrics.log(epoch, val_metrics, "val")

            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            self._save_checkpoint(epoch, val_metrics, is_best)

            if patience_counter >= self.training_config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        self.metrics.save()
        return {
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch,
            "history": self.history,
        }
