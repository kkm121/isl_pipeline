from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable, List
import time
import json
import logging
from pathlib import Path

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

@dataclass
class StateTransition:
    from_state: PipelineState
    to_state: PipelineState
    timestamp: float
    evidence: Dict[str, Any]  # what triggered the transition
    agent: str  # which agent provided the evidence

@dataclass
class RetryPolicy:
    max_retries_per_failure: int = 2
    max_total_retries: int = 5
    max_config_mutations: int = 3
    current_retries: int = 0
    current_config_mutations: int = 0
    failure_signatures: Dict[str, int] = field(default_factory=dict)  # signature -> count

class InvalidTransitionError(Exception):
    pass

class PipelineStateMachine:
    ALLOWED_TRANSITIONS = {
        PipelineState.IDLE: [PipelineState.SPECIFICATION],
        PipelineState.SPECIFICATION: [PipelineState.TEST_PLAN, PipelineState.HUMAN_GATE],
        PipelineState.TEST_PLAN: [PipelineState.IMPLEMENTATION, PipelineState.HUMAN_GATE],
        PipelineState.IMPLEMENTATION: [PipelineState.STATIC_VERIFY, PipelineState.HUMAN_GATE],
        PipelineState.STATIC_VERIFY: [PipelineState.DYNAMIC_VERIFY, PipelineState.RETRY, PipelineState.HUMAN_GATE],
        PipelineState.DYNAMIC_VERIFY: [PipelineState.DIFF_VERIFY, PipelineState.RETRY, PipelineState.HUMAN_GATE],
        PipelineState.DIFF_VERIFY: [PipelineState.INDEPENDENT_REVIEW, PipelineState.RETRY, PipelineState.HUMAN_GATE],
        PipelineState.INDEPENDENT_REVIEW: [PipelineState.ACCEPT, PipelineState.RETRY, PipelineState.HUMAN_GATE],
        PipelineState.ACCEPT: [PipelineState.COMPLETE],
        PipelineState.RETRY: [PipelineState.IMPLEMENTATION, PipelineState.STATIC_VERIFY, PipelineState.HUMAN_GATE, PipelineState.FAILED],
        PipelineState.HUMAN_GATE: [], # terminal
        PipelineState.FAILED: [], # terminal
        PipelineState.COMPLETE: [PipelineState.IDLE]
    }

    def __init__(self, retry_policy: Optional[RetryPolicy] = None, log_dir: str = 'logs/agents/'):
        self.retry_policy = retry_policy or RetryPolicy()
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_state = PipelineState.IDLE
        self.transition_history: List[StateTransition] = []

    def transition(self, target: PipelineState, evidence: Dict[str, Any], agent: str) -> bool:
        allowed = self.ALLOWED_TRANSITIONS.get(self.current_state, [])
        if target not in allowed:
            raise InvalidTransitionError(f"Cannot transition from {self.current_state} to {target}")

        actual_target = target
        if target == PipelineState.RETRY:
            self.retry_policy.current_retries += 1
            if reason := self.check_retry_limits():
                logger.warning(f"Retry limits exceeded ({reason}), forcing transition to HUMAN_GATE.")
                actual_target = PipelineState.HUMAN_GATE
                evidence["forced_reason"] = reason.name

        transition_record = StateTransition(
            from_state=self.current_state,
            to_state=actual_target,
            timestamp=time.time(),
            evidence=evidence,
            agent=agent
        )
        self.transition_history.append(transition_record)
        self.current_state = actual_target
        return True

    def record_failure(self, signature: str, details: Dict):
        count = self.retry_policy.failure_signatures.get(signature, 0) + 1
        self.retry_policy.failure_signatures[signature] = count
        if count >= 2:
            logger.warning(f"Repeated failure signature: {signature}. May trigger HUMAN_GATE.")

    def check_retry_limits(self) -> Optional[HumanGateReason]:
        if self.retry_policy.current_retries > self.retry_policy.max_total_retries:
            return HumanGateReason.MAX_RETRIES_EXCEEDED
        for sig, count in self.retry_policy.failure_signatures.items():
            if count >= self.retry_policy.max_retries_per_failure:
                return HumanGateReason.REPEATED_FAILURE
        return None

    def get_state(self) -> PipelineState:
        return self.current_state

    def get_history(self) -> List[StateTransition]:
        return self.transition_history

    def save_state(self, path: str):
        data = {
            "current_state": self.current_state.name,
            "history": [
                {
                    "from_state": t.from_state.name,
                    "to_state": t.to_state.name,
                    "timestamp": t.timestamp,
                    "evidence": t.evidence,
                    "agent": t.agent
                } for t in self.transition_history
            ],
            "retry_policy": {
                "max_retries_per_failure": self.retry_policy.max_retries_per_failure,
                "max_total_retries": self.retry_policy.max_total_retries,
                "max_config_mutations": self.retry_policy.max_config_mutations,
                "current_retries": self.retry_policy.current_retries,
                "current_config_mutations": self.retry_policy.current_config_mutations,
                "failure_signatures": self.retry_policy.failure_signatures
            }
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_state(self, path: str):
        with open(path, 'r') as f:
            data = json.load(f)
        self.current_state = PipelineState[data["current_state"]]
        self.transition_history = [
            StateTransition(
                from_state=PipelineState[t["from_state"]],
                to_state=PipelineState[t["to_state"]],
                timestamp=t["timestamp"],
                evidence=t["evidence"],
                agent=t["agent"]
            ) for t in data["history"]
        ]
        rp = data["retry_policy"]
        self.retry_policy = RetryPolicy(
            max_retries_per_failure=rp["max_retries_per_failure"],
            max_total_retries=rp["max_total_retries"],
            max_config_mutations=rp["max_config_mutations"],
            current_retries=rp["current_retries"],
            current_config_mutations=rp["current_config_mutations"],
            failure_signatures=rp["failure_signatures"]
        )

    def reset(self):
        self.current_state = PipelineState.IDLE
        self.transition_history = []
        self.retry_policy.current_retries = 0
        self.retry_policy.current_config_mutations = 0
        self.retry_policy.failure_signatures.clear()
