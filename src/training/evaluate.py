import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
from collections import Counter
import json
import logging
from pathlib import Path

from src.models.classifier import ISLClassifier

logger = logging.getLogger(__name__)

def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

def per_class_metrics(confusion_matrix: np.ndarray, class_names: Optional[List[str]] = None) -> Dict:
    num_classes = confusion_matrix.shape[0]
    metrics = {}
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        cname = class_names[i] if class_names else str(i)
        metrics[cname] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
    return metrics

def evaluate(model: ISLClassifier, loader: DataLoader, device: str, class_names: Optional[List[str]] = None) -> Dict:
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    accuracy = (y_true == y_pred).mean()
    cm = compute_confusion_matrix(y_true, y_pred, model.config.num_classes)
    class_metrics = per_class_metrics(cm, class_names)
    
    return {
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class': class_metrics
    }

def save_evaluation_report(metrics: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
