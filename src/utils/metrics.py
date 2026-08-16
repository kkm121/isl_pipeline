import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class MetricsTracker:
    def __init__(self, log_dir: str = "logs/local", experiment_name: str = "default"):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history: Dict[str, Dict[str, List[Tuple[int, float]]]] = {}
        self.metrics: Dict[str, List[float]] = {}  # key -> list of values
        self.file_path = self.log_dir / f"{experiment_name}_metrics.json"

    def log(self, epoch: int, metrics: Dict[str, float], phase: str = "train") -> None:
        if phase not in self.history:
            self.history[phase] = {}
        for k, v in metrics.items():
            if k not in self.history[phase]:
                self.history[phase][k] = []
            self.history[phase][k].append((epoch, v))

    def update(self, metric_name: str, value: float) -> None:
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)

    def get_best(self, metric_name: str, mode: str = "min") -> Tuple[int, float]:
        phase = "val" if "val" in self.history and metric_name in self.history["val"] else "train"
        if phase not in self.history or metric_name not in self.history[phase]:
            return -1, float("inf") if mode == "min" else float("-inf")

        values = self.history[phase][metric_name]
        if mode == "min":
            return min(values, key=lambda x: x[1])
        else:
            return max(values, key=lambda x: x[1])

    def save(self, path: Optional[Any] = None) -> None:
        target = Path(path) if path else self.file_path
        target.parent.mkdir(parents=True, exist_ok=True)
        save_data = {"history": self.history, "metrics": self.metrics}
        with open(target, "w") as f:
            json.dump(save_data, f, indent=4)

    def load(self, path: Any) -> None:
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "metrics" in data:
                self.history = data.get("history", {})
                self.metrics = data.get("metrics", {})
            else:
                self.history = data
                self.metrics = data


class RunningAverage:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count > 0 else 0.0

    def compute(self) -> float:
        return self.avg

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0
