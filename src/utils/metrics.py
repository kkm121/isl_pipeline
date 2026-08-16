import json
from typing import Dict, Tuple
from pathlib import Path

class MetricsTracker:
    def __init__(self, log_dir: str, experiment_name: str):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = {}
        self.file_path = self.log_dir / f"{experiment_name}_metrics.json"

    def log(self, epoch: int, metrics: Dict[str, float], phase: str = 'train'):
        if phase not in self.history:
            self.history[phase] = {}
        for k, v in metrics.items():
            if k not in self.history[phase]:
                self.history[phase][k] = []
            self.history[phase][k].append((epoch, v))

    def get_best(self, metric_name: str, mode: str = 'min') -> Tuple[int, float]:
        # Look in val phase primarily, fallback to train if not available
        phase = 'val' if 'val' in self.history and metric_name in self.history['val'] else 'train'
        if phase not in self.history or metric_name not in self.history[phase]:
            return -1, float('inf') if mode == 'min' else float('-inf')
            
        values = self.history[phase][metric_name]
        if mode == 'min':
            return min(values, key=lambda x: x[1])
        else:
            return max(values, key=lambda x: x[1])

    def save(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.history, f, indent=4)

    def load(self, path: str):
        with open(path, 'r') as f:
            self.history = json.load(f)

class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1):
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def reset(self):
        self.total = 0.0
        self.count = 0
