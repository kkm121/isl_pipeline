import torch
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import logging
from pathlib import Path

from src.models.classifier import ISLClassifier

logger = logging.getLogger(__name__)

def resolve_device(device: str = "auto") -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device

def compute_confusion_matrix(y_pred: Union[torch.Tensor, np.ndarray], y_true: Union[torch.Tensor, np.ndarray], num_classes: int) -> np.ndarray:
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.cpu().numpy()

    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    for p, t in zip(y_pred, y_true):
        # cm[true_label, pred_label]
        cm[t, p] += 1
    return cm

def per_class_metrics(
    preds_or_cm: Union[torch.Tensor, np.ndarray],
    labels: Optional[Union[torch.Tensor, np.ndarray]] = None,
    num_classes: Optional[int] = None,
    class_names: Optional[List[str]] = None
) -> Dict:
    if labels is not None:
        n_cls = num_classes or max(int(np.max(preds_or_cm)), int(np.max(labels))) + 1
        cm = compute_confusion_matrix(preds_or_cm, labels, n_cls)
    else:
        cm = np.array(preds_or_cm)

    n_classes = cm.shape[0]
    per_class = {}
    precisions = []
    recalls = []
    f1s = []
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        
        cname = class_names[i] if class_names and i < len(class_names) else str(i)
        per_class[cname] = {
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }
        
    avg_precision = float(np.mean(precisions)) if precisions else 0.0
    avg_recall = float(np.mean(recalls)) if recalls else 0.0
    avg_f1 = float(np.mean(f1s)) if f1s else 0.0

    return {
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1,
        'per_class': per_class,
        **per_class
    }

def evaluate(model: ISLClassifier, loader: DataLoader, device: str = "auto", class_names: Optional[List[str]] = None) -> Dict:
    dev = resolve_device(device)
    model.eval()
    all_preds = []
    all_targets = []
    criterion = torch.nn.CrossEntropyLoss()
    total_loss = 0.0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    
    accuracy = (y_true == y_pred).mean() if len(y_true) > 0 else 0.0
    cm = compute_confusion_matrix(y_pred, y_true, model.config.num_classes)
    class_metrics = per_class_metrics(cm, class_names=class_names)
    
    return {
        'loss': float(total_loss / max(len(loader), 1)),
        'accuracy': float(accuracy),
        'confusion_matrix': cm.tolist(),
        'per_class': class_metrics.get('per_class', {}),
        'precision': class_metrics.get('precision', 0.0),
        'recall': class_metrics.get('recall', 0.0),
        'f1': class_metrics.get('f1', 0.0)
    }

def save_evaluation_report(metrics: Dict, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)
