import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
import json
import time
import logging
from typing import Optional, Dict, Any
import math

from src.models.classifier import ISLClassifier
from src.models.config import PipelineConfig
from src.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(self, model: ISLClassifier, config: PipelineConfig, device: str):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.training.learning_rate, 
            weight_decay=config.training.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=config.training.epochs
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler(enabled=config.training.mixed_precision)
        
        self.metrics = MetricsTracker(config.training.log_dir, config.experiment_name)
        Path(config.training.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    def _check_nan(self, loss: torch.Tensor) -> bool:
        return torch.isnan(loss).any().item() or torch.isinf(loss).any().item()

    def _get_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return math.sqrt(total_norm)

    def _train_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            
            with autocast(enabled=self.config.training.mixed_precision):
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
            if self._check_nan(loss):
                logger.warning("NaN loss detected. Skipping batch.")
                continue
                
            self.scaler.scale(loss).backward()
            
            self.scaler.unscale_(self.optimizer)
            grad_norm = self._get_grad_norm()
            if math.isnan(grad_norm) or math.isinf(grad_norm):
                 logger.warning("NaN gradient detected. Skipping batch.")
                 self.optimizer.zero_grad()
                 continue
                 
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training.gradient_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()
            
        return {
            'loss': total_loss / max(len(loader), 1),
            'accuracy': correct / max(total, 1)
        }

    def _validate(self, loader: DataLoader) -> Dict[str, float]:
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
            'loss': total_loss / max(len(loader), 1),
            'accuracy': correct / max(total, 1)
        }

    def _save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        ckpt_path = Path(self.config.training.checkpoint_dir) / f"{self.config.experiment_name}_latest.pth"
        self.model.save(str(ckpt_path))
        if is_best:
            best_path = Path(self.config.training.checkpoint_dir) / f"{self.config.experiment_name}_best.pth"
            self.model.save(str(best_path))

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(1, self.config.training.epochs + 1):
            t0 = time.time()
            train_metrics = self._train_epoch(train_loader)
            val_metrics = self._validate(val_loader)
            self.scheduler.step()
            
            self.metrics.log(epoch, train_metrics, 'train')
            self.metrics.log(epoch, val_metrics, 'val')
            
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
            else:
                patience_counter += 1
                
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            logger.info(f"Epoch {epoch}/{self.config.training.epochs} - {time.time()-t0:.1f}s")
            logger.info(f"Train: Loss {train_metrics['loss']:.4f} Acc {train_metrics['accuracy']:.4f}")
            logger.info(f"Val: Loss {val_metrics['loss']:.4f} Acc {val_metrics['accuracy']:.4f}")
            
            if patience_counter >= self.config.training.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
                
        self.metrics.save()
        return {
            'best_val_loss': best_val_loss,
            'epochs_trained': epoch
        }
