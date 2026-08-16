from dataclasses import dataclass
from typing import Optional, Dict
import time
import logging

logger = logging.getLogger(__name__)

@dataclass
class ResourceBudget:
    max_runtime_seconds: int = 3600  # 1 hour default per task
    max_retry_count: int = 5
    max_kaggle_submissions: int = 10
    max_disk_mb: int = 10240  # 10GB
    max_artifact_size_mb: int = 1024  # 1GB per artifact
    max_container_memory_mb: int = 4096
    max_container_cpu_cores: int = 2
    max_concurrent_agents: int = 5
    max_concurrent_kaggle_jobs: int = 2

class ResourceTracker:
    def __init__(self, budget: Optional[ResourceBudget] = None):
        self.budget = budget or ResourceBudget()
        self.start_time = time.time()
        self.usage = {
            "retry_count": 0,
            "kaggle_submissions": 0,
            "disk_mb": 0.0,
            "artifact_size_mb": 0.0,
            "container_memory_mb": 0.0,
            "container_cpu_cores": 0.0,
            "concurrent_agents": 0,
            "concurrent_kaggle_jobs": 0
        }

    def _get_limit(self, resource: str) -> Optional[float]:
        attr_name = f"max_{resource}"
        if hasattr(self.budget, attr_name):
            return getattr(self.budget, attr_name)
        return None

    def check_budget(self, resource: str, amount: float = 1) -> bool:
        if resource == "runtime_seconds":
            return not self.is_runtime_exceeded()
        limit = self._get_limit(resource)
        if limit is None:
            return True  # Unbounded if not in budget definition
        current = self.usage.get(resource, 0)
        return (current + amount) <= limit

    def consume(self, resource: str, amount: float = 1):
        if resource in self.usage:
            self.usage[resource] += amount
        else:
            self.usage[resource] = amount

    def release(self, resource: str, amount: float = 1):
        if resource in self.usage:
            self.usage[resource] = max(0.0, self.usage[resource] - amount)

    def get_usage(self) -> Dict:
        return {
            "usage": dict(self.usage),
            "runtime_seconds": time.time() - self.start_time,
            "budget": self.budget.__dict__
        }

    def is_runtime_exceeded(self) -> bool:
        return (time.time() - self.start_time) > self.budget.max_runtime_seconds

    def get_remaining(self, resource: str) -> float:
        if resource == "runtime_seconds":
            return max(0.0, self.budget.max_runtime_seconds - (time.time() - self.start_time))
        limit = self._get_limit(resource)
        if limit is None:
            return float('inf')
        return max(0.0, limit - self.usage.get(resource, 0))
