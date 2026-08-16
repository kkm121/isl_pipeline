"""Deterministic State Machine Controller for Autonomous Agent Pipelines.

Enforces rigid transitions, explicit HUMAN_GATE escalations, hard retry limits,
before/after Git tree diff verification, and config mutation limits.
The LLM does NOT decide the state transitions; this controller does.
"""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    IDLE = auto()
    SPECIFICATION = auto()
    TEST_PLAN = auto()
    IMPLEMENTATION = auto()
    STATIC_VERIFY = auto()
    DYNAMIC_VERIFY = auto()
    DIFF_VERIFY = auto()
    INDEPENDENT_REVIEW = auto()
    ACCEPT = auto()
    RETRY = auto()
    HUMAN_GATE = auto()  # EXPLICIT terminal state for uncertain/unsafe situations
    FAILED = auto()
    COMPLETE = auto()


class HumanGateReason(Enum):
    UNCERTAIN_DIAGNOSIS = auto()
    DESTRUCTIVE_OPERATION = auto()
    REPEATED_FAILURE = auto()  # same failure signature 2x+
    UNSAFE_REMEDIATION = auto()
    CREDENTIAL_ISSUE = auto()
    POLICY_VIOLATION = auto()  # e.g. ToS concerns
    MAX_RETRIES_EXCEEDED = auto()
    RESOURCE_BUDGET_EXCEEDED = auto()
    CONFIG_MUTATION_LIMIT_EXCEEDED = auto()
    ZERO_DIFF_FAILURE = auto()


class InvalidTransitionError(Exception):
    """Raised when an illegal state transition is attempted."""


class HumanGateException(Exception):
    """Raised when an operation triggers an automatic escalation to HUMAN_GATE."""


@dataclass
class StateTransition:
    from_state: PipelineState
    to_state: PipelineState
    timestamp: float
    evidence: dict[str, Any]
    agent: str


@dataclass
class RetryPolicy:
    max_retries_per_failure: int = 2
    max_total_retries: int = 5
    max_config_mutations: int = 3
    current_retries: int = 0
    current_config_mutations: int = 0
    failure_signatures: dict[str, int] = field(default_factory=dict)


class PipelineStateMachine:
    """Rigid state machine with programmatic transition enforcement."""

    ALLOWED_TRANSITIONS = {
        PipelineState.IDLE: [PipelineState.SPECIFICATION],
        PipelineState.SPECIFICATION: [
            PipelineState.TEST_PLAN,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.TEST_PLAN: [
            PipelineState.IMPLEMENTATION,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.IMPLEMENTATION: [
            PipelineState.STATIC_VERIFY,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.STATIC_VERIFY: [
            PipelineState.DYNAMIC_VERIFY,
            PipelineState.RETRY,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.DYNAMIC_VERIFY: [
            PipelineState.DIFF_VERIFY,
            PipelineState.RETRY,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.DIFF_VERIFY: [
            PipelineState.INDEPENDENT_REVIEW,
            PipelineState.RETRY,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.INDEPENDENT_REVIEW: [
            PipelineState.ACCEPT,
            PipelineState.RETRY,
            PipelineState.HUMAN_GATE,
        ],
        PipelineState.ACCEPT: [PipelineState.COMPLETE],
        PipelineState.RETRY: [
            PipelineState.IMPLEMENTATION,
            PipelineState.STATIC_VERIFY,
            PipelineState.HUMAN_GATE,
            PipelineState.FAILED,
        ],
        PipelineState.HUMAN_GATE: [],  # terminal: requires manual intervention
        PipelineState.FAILED: [],  # terminal
        PipelineState.COMPLETE: [PipelineState.IDLE],
    }

    def __init__(
        self,
        retry_policy: RetryPolicy | None = None,
        log_dir: str = "logs/agents/",
        project_root: str = ".",
    ):
        self.retry_policy = retry_policy or RetryPolicy()
        self.log_dir = Path(log_dir)
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.log_dir = Path("/tmp/logs/agents")
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.project_root = Path(project_root).resolve()
        self.current_state = PipelineState.IDLE
        self.transition_history: list[StateTransition] = []
        self._tree_baseline: dict[str, Any] | None = None

    def capture_tree_baseline(self) -> dict[str, Any]:
        """Capture and persist the Git tree baseline state immediately before an implementation or remediation turn."""
        head_sha = "unknown"
        status_lines: list[str] = []
        try:
            head_proc = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False,
            )
            if head_proc.returncode == 0:
                head_sha = head_proc.stdout.strip()

            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False,
            )
            status_output = status_proc.stdout.strip()
            if status_output:
                status_lines = status_output.splitlines()
        except Exception as e:
            logger.warning("Could not capture Git tree baseline via CLI: %s", str(e))
            head_sha = "none"

        self._tree_baseline = {
            "head_sha": head_sha,
            "status_lines": status_lines,
            "timestamp": time.time(),
        }

        # Persist baseline to disk for consumption by verify.sh and downstream gates
        try:
            state_dir = self.project_root / ".state"
            state_dir.mkdir(parents=True, exist_ok=True)
            with open(state_dir / "tree_baseline.sha", "w") as f:
                f.write(head_sha)
            with open(state_dir / "tree_baseline.json", "w") as f:
                json.dump(self._tree_baseline, f, indent=2)
        except OSError:
            pass

        logger.info(
            "Captured and persisted Git tree baseline: HEAD=%s, dirty_files=%d",
            head_sha,
            len(status_lines),
        )
        return self._tree_baseline

    def verify_git_diff(self, expected_modifications: bool = True) -> tuple[bool, dict[str, Any]]:
        """Perform real before/after Git tree diff verification against the persisted turn baseline.

        If expected_modifications=True and the diff is zero or baseline is missing, rejects and flags failure.
        """
        try:
            state_dir = self.project_root / ".state"
            baseline_file = state_dir / "tree_baseline.json"
            if not self._tree_baseline and baseline_file.exists():
                with open(baseline_file, "r") as f:
                    self._tree_baseline = json.load(f)

            if not self._tree_baseline:
                logger.error("DIFF_VERIFY failed: No turn baseline found in memory or at .state/tree_baseline.json")
                return False, {
                    "error": "Missing baseline: Implementation turn did not establish a persisted Git tree baseline.",
                    "has_changes": False,
                }

            status_proc = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False,
            )
            current_status = status_proc.stdout.strip().splitlines() if status_proc.stdout.strip() else []

            diff_proc = subprocess.run(
                ["git", "diff", "--stat"],
                cwd=str(self.project_root),
                capture_output=True,
                text=True,
                check=False,
            )
            diff_stat = diff_proc.stdout.strip()

            baseline_status = set(self._tree_baseline.get("status_lines", [])) if self._tree_baseline else set()
            new_changes = set(current_status) - baseline_status

            has_changes = bool(new_changes) or bool(diff_stat)

            details = {
                "has_changes": has_changes,
                "diff_stat": diff_stat,
                "changed_files": list(new_changes) or current_status,
                "baseline_head": self._tree_baseline.get("head_sha") if self._tree_baseline else None,
            }

            if expected_modifications and not has_changes:
                logger.error("DIFF_VERIFY failed: Expected modifications but Git working tree shows ZERO diff.")
                return False, {
                    "error": "Zero diff detected: Agent claimed modifications but no actual file changes were recorded.",
                    **details,
                }

            return True, details
        except Exception as e:
            logger.error("Error during Git diff verification: %s", str(e))
            return False, {"error": f"Git diff verification exception: {e!s}"}

    def transition(self, target: PipelineState, evidence: dict[str, Any], agent: str) -> bool:
        """Execute a state transition with deterministic boundary checks."""
        allowed = self.ALLOWED_TRANSITIONS.get(self.current_state, [])
        if target not in allowed:
            raise InvalidTransitionError(f"Cannot transition from {self.current_state.name} to {target.name}")

        actual_target = target

        # Automatically capture baseline immediately upon entering IMPLEMENTATION or RETRY
        if target in (PipelineState.IMPLEMENTATION, PipelineState.RETRY):
            self.capture_tree_baseline()

        # Handle Diff Verification automatically if entering DIFF_VERIFY
        if target == PipelineState.DIFF_VERIFY:
            diff_ok, diff_details = self.verify_git_diff(expected_modifications=evidence.get("expected_changes", True))
            evidence["diff_verification"] = diff_details
            if not diff_ok:
                logger.warning("Diff verification failed. Transitioning to RETRY.")
                self.record_failure("zero_diff_error", diff_details)
                actual_target = PipelineState.RETRY

        # Handle RETRY checks
        if actual_target == PipelineState.RETRY:
            self.retry_policy.current_retries += 1
            if reason := self.check_retry_limits():
                logger.warning(f"Retry limits exceeded ({reason.name}), forcing transition to HUMAN_GATE.")
                actual_target = PipelineState.HUMAN_GATE
                evidence["forced_human_gate_reason"] = reason.name

        transition_record = StateTransition(
            from_state=self.current_state,
            to_state=actual_target,
            timestamp=time.time(),
            evidence=evidence,
            agent=agent,
        )
        self.transition_history.append(transition_record)
        self.current_state = actual_target
        return True

    def mutate_config(
        self,
        config_path: str,
        mutation_data: dict[str, Any],
        reason: str,
        agent: str = "ml-ops",
    ) -> dict[str, Any]:
        """Apply an autonomous configuration mutation with strict limit enforcement."""
        if self.retry_policy.current_config_mutations >= self.retry_policy.max_config_mutations:
            error_msg = (
                f"Config mutation limit exceeded ({self.retry_policy.current_config_mutations}/"
                f"{self.retry_policy.max_config_mutations}). Halting autonomous changes."
            )
            logger.critical(error_msg)
            # Force transition to HUMAN_GATE
            if (
                self.current_state in self.ALLOWED_TRANSITIONS
                and PipelineState.HUMAN_GATE in self.ALLOWED_TRANSITIONS[self.current_state]
            ):
                self.transition(
                    PipelineState.HUMAN_GATE,
                    {"reason": error_msg, "mutation_attempted": mutation_data},
                    agent=agent,
                )
            else:
                self.current_state = PipelineState.HUMAN_GATE
            raise HumanGateException(error_msg)

        # Apply mutation safely
        target_file = Path(config_path)
        if not target_file.is_absolute():
            target_file = self.project_root / config_path

        if not target_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Read existing config
        with open(target_file, "r") as f:
            if target_file.suffix in (".yaml", ".yml"):
                content = yaml.safe_load(f) or {}
            else:
                content = json.load(f) or {}

        # Merge updates
        def deep_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict) and isinstance(d.get(k), dict):
                    deep_update(d[k], v)
                else:
                    d[k] = v

        deep_update(content, mutation_data)

        # Write back
        with open(target_file, "w") as f:
            if target_file.suffix in (".yaml", ".yml"):
                yaml.safe_dump(content, f, sort_keys=False)
            else:
                json.dump(content, f, indent=2)

        self.retry_policy.current_config_mutations += 1
        logger.info(
            "Config mutated (%d/%d): %s (Reason: %s)",
            self.retry_policy.current_config_mutations,
            self.retry_policy.max_config_mutations,
            config_path,
            reason,
        )

        return {
            "status": "mutated",
            "mutations_used": self.retry_policy.current_config_mutations,
            "max_mutations": self.retry_policy.max_config_mutations,
            "updated_config": content,
        }

    def record_failure(self, signature: str, details: dict):
        """Track failure signatures to detect repetitive loops."""
        count = self.retry_policy.failure_signatures.get(signature, 0) + 1
        self.retry_policy.failure_signatures[signature] = count
        if count >= self.retry_policy.max_retries_per_failure:
            logger.warning(
                "Repeated failure signature '%s' reached limit (%d/%d).",
                signature,
                count,
                self.retry_policy.max_retries_per_failure,
            )

    def check_retry_limits(self) -> HumanGateReason | None:
        """Check if any retry or loop limit is exceeded."""
        if self.retry_policy.current_retries > self.retry_policy.max_total_retries:
            return HumanGateReason.MAX_RETRIES_EXCEEDED
        for sig, count in self.retry_policy.failure_signatures.items():
            if count >= self.retry_policy.max_retries_per_failure:
                return HumanGateReason.REPEATED_FAILURE
        if self.retry_policy.current_config_mutations >= self.retry_policy.max_config_mutations:
            return HumanGateReason.CONFIG_MUTATION_LIMIT_EXCEEDED
        return None

    def get_state(self) -> PipelineState:
        return self.current_state

    def get_history(self) -> list[StateTransition]:
        return self.transition_history

    def save_state(self, path: str):
        data = {
            "current_state": self.current_state.name,
            "tree_baseline": self._tree_baseline,
            "history": [
                {
                    "from_state": t.from_state.name,
                    "to_state": t.to_state.name,
                    "timestamp": t.timestamp,
                    "evidence": t.evidence,
                    "agent": t.agent,
                }
                for t in self.transition_history
            ],
            "retry_policy": {
                "max_retries_per_failure": self.retry_policy.max_retries_per_failure,
                "max_total_retries": self.retry_policy.max_total_retries,
                "max_config_mutations": self.retry_policy.max_config_mutations,
                "current_retries": self.retry_policy.current_retries,
                "current_config_mutations": self.retry_policy.current_config_mutations,
                "failure_signatures": self.retry_policy.failure_signatures,
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        self.current_state = PipelineState[data["current_state"]]
        self._tree_baseline = data.get("tree_baseline")
        self.transition_history = [
            StateTransition(
                from_state=PipelineState[t["from_state"]],
                to_state=PipelineState[t["to_state"]],
                timestamp=t["timestamp"],
                evidence=t["evidence"],
                agent=t["agent"],
            )
            for t in data["history"]
        ]
        rp = data["retry_policy"]
        self.retry_policy = RetryPolicy(
            max_retries_per_failure=rp["max_retries_per_failure"],
            max_total_retries=rp["max_total_retries"],
            max_config_mutations=rp["max_config_mutations"],
            current_retries=rp["current_retries"],
            current_config_mutations=rp["current_config_mutations"],
            failure_signatures=rp["failure_signatures"],
        )

    def reset(self):
        self.current_state = PipelineState.IDLE
        self.transition_history = []
        self.retry_policy.current_retries = 0
        self.retry_policy.current_config_mutations = 0
        self.retry_policy.failure_signatures.clear()
        self._tree_baseline = None
