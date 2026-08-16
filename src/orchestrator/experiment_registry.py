import hashlib
import json
import logging
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ExperimentProvenance:
    experiment_id: str
    commit_sha: str
    config_hash: str
    docker_image_digest: Optional[str] = None
    kaggle_kernel_id: Optional[str] = None
    kaggle_account_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    model_version: Optional[str] = None
    dataset_version: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    checkpoint_checksum: Optional[str] = None
    status: str = "created"
    diagnosis: Optional[Dict[str, Any]] = None


class ExperimentRegistry:
    def __init__(self, experiments_dir: str = "experiments/", project_root: str = "."):
        self.experiments_dir = Path(experiments_dir)
        self.project_root = Path(project_root)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def _get_commit_sha(self) -> str:
        try:
            res = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                check=True,
            )
            return res.stdout.strip()
        except (subprocess.SubprocessError, OSError) as e:
            logger.warning(f"Could not get git commit SHA: {e}")
            return "unknown"

    def _compute_hash(self, file_path: str) -> str:
        sha256 = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for block in iter(lambda: f.read(4096), b""):
                    sha256.update(block)
            return sha256.hexdigest()
        except FileNotFoundError:
            return "not_found"

    def _compute_checkpoint_checksum(self, path: str) -> str:
        return self._compute_hash(path)

    def _next_experiment_id(self) -> str:
        max_id = 0
        for p in self.experiments_dir.iterdir():
            if p.is_dir() and p.name.startswith("exp-"):
                m = re.match(r"exp-(\d+)", p.name)
                if m:
                    max_id = max(max_id, int(m.group(1)))
        return f"exp-{max_id + 1:04d}"

    def create_experiment(self, config_path: str, experiment_name: Optional[str] = None) -> ExperimentProvenance:
        exp_id = experiment_name if experiment_name else self._next_experiment_id()
        exp_dir = self.experiments_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        prov = ExperimentProvenance(
            experiment_id=exp_id,
            commit_sha=self._get_commit_sha(),
            config_hash=self._compute_hash(config_path),
        )

        try:
            shutil.copy2(config_path, exp_dir / "config.json")
        except FileNotFoundError:
            logger.warning(f"Config path {config_path} not found.")

        with open(exp_dir / "spec.json", "w") as f:
            json.dump(asdict(prov), f, indent=2)

        return prov

    def update_experiment(self, experiment_id: str, **kwargs: Any) -> None:
        prov = self.get_experiment(experiment_id)
        if not prov:
            raise ValueError(f"Experiment {experiment_id} not found")

        for k, v in kwargs.items():
            if hasattr(prov, k):
                setattr(prov, k, v)

        self._save_prov(prov)

    def record_metrics(self, experiment_id: str, metrics: Dict[str, float]) -> None:
        prov = self.get_experiment(experiment_id)
        if not prov:
            raise ValueError(f"Experiment {experiment_id} not found")
        prov.metrics.update(metrics)
        self._save_prov(prov)

    def record_diagnosis(self, experiment_id: str, diagnosis: Dict[str, Any]) -> None:
        prov = self.get_experiment(experiment_id)
        if not prov:
            raise ValueError(f"Experiment {experiment_id} not found")
        prov.diagnosis = diagnosis
        self._save_prov(prov)

    def complete_experiment(
        self, experiment_id: str, metrics: Dict[str, float], checkpoint_path: Optional[str] = None
    ) -> None:
        prov = self.get_experiment(experiment_id)
        if not prov:
            raise ValueError(f"Experiment {experiment_id} not found")
        prov.status = "completed"
        prov.metrics.update(metrics)
        if checkpoint_path:
            prov.checkpoint_path = checkpoint_path
            prov.checkpoint_checksum = self._compute_checkpoint_checksum(checkpoint_path)
        self._save_prov(prov)

    def fail_experiment(self, experiment_id: str, diagnosis: Dict[str, Any]) -> None:
        prov = self.get_experiment(experiment_id)
        if not prov:
            raise ValueError(f"Experiment {experiment_id} not found")
        prov.status = "failed"
        prov.diagnosis = diagnosis
        self._save_prov(prov)

    def get_experiment(self, experiment_id: str) -> Optional[ExperimentProvenance]:
        path = self.experiments_dir / experiment_id / "spec.json"
        if not path.exists():
            return None
        with open(path, "r") as f:
            data = json.load(f)
        return ExperimentProvenance(**data)

    def list_experiments(self, status: Optional[str] = None) -> List[ExperimentProvenance]:
        res = []
        for p in self.experiments_dir.iterdir():
            if p.is_dir() and (p / "spec.json").exists():
                with open(p / "spec.json", "r") as f:
                    data = json.load(f)
                    prov = ExperimentProvenance(**data)
                    if status is None or prov.status == status:
                        res.append(prov)
        return res

    def _save_prov(self, prov: ExperimentProvenance) -> None:
        path = self.experiments_dir / prov.experiment_id / "spec.json"
        with open(path, "w") as f:
            json.dump(asdict(prov), f, indent=2)
