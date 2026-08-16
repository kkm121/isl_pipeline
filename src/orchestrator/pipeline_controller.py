"""Pipeline Controller — Orchestration Entry Point Gating All Agent Operations.

Enforces the deterministic multi-plane architecture:
- Agent actions and MCP tool operations are strictly gated by PipelineStateMachine.
- LLM cannot execute tools out of state order or bypass mandatory verification gates.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from src.orchestrator.resource_budgets import ResourceTracker
from src.orchestrator.state_machine import (
    PipelineState,
    PipelineStateMachine,
    RetryPolicy,
)

logger = logging.getLogger(__name__)


class GateViolationError(Exception):
    """Raised when an agent or MCP operation is attempted outside its permitted state."""

    pass


class PipelineController:
    """Orchestration controller that gates all pipeline stages and MCP operations."""

    def __init__(
        self,
        state_machine: Optional[PipelineStateMachine] = None,
        budget_tracker: Optional[ResourceTracker] = None,
        project_root: str = ".",
    ):
        self.project_root = Path(project_root).resolve()
        self.state_machine = state_machine or PipelineStateMachine(
            project_root=str(self.project_root),
            retry_policy=RetryPolicy(),
        )
        self.budget_tracker = budget_tracker or ResourceTracker()
        self.mcp_handlers: Dict[str, Callable[..., Any]] = {}

    @property
    def current_state(self) -> PipelineState:
        return self.state_machine.current_state

    # -------------------------------------------------------------------------
    # Gated Agent Lifecycle Transitions
    # -------------------------------------------------------------------------

    def start_specification(self, task_spec: Dict[str, Any], agent: str = "principal-engineer") -> None:
        """Advance IDLE -> SPECIFICATION."""
        if not task_spec.get("objective") or not task_spec.get("requirements"):
            raise ValueError("Task specification must contain non-empty objective and requirements.")
        self.state_machine.transition(
            target=PipelineState.SPECIFICATION,
            evidence={"task_spec": task_spec},
            agent=agent,
        )

    def submit_test_plan(self, test_plan: Dict[str, Any], agent: str = "test-engineer") -> None:
        """Advance SPECIFICATION -> TEST_PLAN."""
        if not test_plan.get("test_cases") or not test_plan.get("acceptance_criteria"):
            raise ValueError("Test plan must contain test_cases and acceptance_criteria.")
        self.state_machine.transition(
            target=PipelineState.TEST_PLAN,
            evidence={"test_plan": test_plan},
            agent=agent,
        )

    def begin_implementation(self, implementation_goals: Dict[str, Any], agent: str = "code-writer") -> None:
        """Advance TEST_PLAN -> IMPLEMENTATION (automatically captures turn baseline)."""
        self.state_machine.transition(
            target=PipelineState.IMPLEMENTATION,
            evidence={"goals": implementation_goals},
            agent=agent,
        )

    def run_static_verification(
        self,
        static_verifier: Optional[Callable[[], tuple[bool, Dict[str, Any]]]] = None,
        agent: str = "test-engineer",
    ) -> bool:
        """Advance to STATIC_VERIFY and execute static checks."""
        self.state_machine.transition(
            target=PipelineState.STATIC_VERIFY,
            evidence={"trigger": "static_check_request"},
            agent=agent,
        )
        if static_verifier:
            passed, results = static_verifier()
            if not passed:
                self.state_machine.record_failure("static_verify_failure", results)
                self.state_machine.transition(
                    target=PipelineState.RETRY,
                    evidence={"failure": results},
                    agent=agent,
                )
                return False
        return True

    def run_dynamic_verification(
        self,
        test_runner: Optional[Callable[[], tuple[bool, Dict[str, Any]]]] = None,
        agent: str = "test-engineer",
    ) -> bool:
        """Advance to DYNAMIC_VERIFY and execute unit/integration test suites."""
        self.state_machine.transition(
            target=PipelineState.DYNAMIC_VERIFY,
            evidence={"trigger": "dynamic_test_request"},
            agent=agent,
        )
        if test_runner:
            passed, results = test_runner()
            if not passed:
                self.state_machine.record_failure("dynamic_test_failure", results)
                self.state_machine.transition(
                    target=PipelineState.RETRY,
                    evidence={"failure": results},
                    agent=agent,
                )
                return False
        return True

    def run_diff_verification(self, expected_changes: bool = True, agent: str = "principal-engineer") -> bool:
        """Advance to DIFF_VERIFY and validate non-zero changes against persisted turn baseline."""
        self.state_machine.transition(
            target=PipelineState.DIFF_VERIFY,
            evidence={"expected_changes": expected_changes},
            agent=agent,
        )
        return self.current_state == PipelineState.DIFF_VERIFY

    def submit_independent_review(
        self,
        review_evidence: Dict[str, Any],
        agent: str = "independent-reviewer",
    ) -> None:
        """Advance DIFF_VERIFY -> INDEPENDENT_REVIEW."""
        if "approved" not in review_evidence or "checklist" not in review_evidence:
            raise ValueError("Review evidence must include 'approved' decision and 'checklist'.")
        self.state_machine.transition(
            target=PipelineState.INDEPENDENT_REVIEW,
            evidence=review_evidence,
            agent=agent,
        )

    def approve_and_accept(self, approval_metadata: Dict[str, Any], agent: str = "principal-engineer") -> None:
        """Advance INDEPENDENT_REVIEW -> ACCEPT."""
        self.state_machine.transition(
            target=PipelineState.ACCEPT,
            evidence=approval_metadata,
            agent=agent,
        )

    def complete_task(self, completion_summary: Dict[str, Any], agent: str = "principal-engineer") -> None:
        """Advance ACCEPT -> COMPLETE."""
        self.state_machine.transition(
            target=PipelineState.COMPLETE,
            evidence=completion_summary,
            agent=agent,
        )

    # -------------------------------------------------------------------------
    # Strict MCP Operation Gatekeeping
    # -------------------------------------------------------------------------

    def register_mcp_handler(self, operation_name: str, handler: Callable[..., Any]) -> None:
        """Register an MCP tool handler."""
        self.mcp_handlers[operation_name] = handler

    def execute_mcp_operation(self, operation_name: str, **kwargs: Any) -> Any:
        """Execute an MCP tool operation after verifying state machine gating."""
        state = self.current_state

        # Gate 1: Code modifications only permitted in IMPLEMENTATION or RETRY
        write_operations = {"write_file", "edit_file", "replace_content", "patch_file", "create_file"}
        if operation_name in write_operations:
            if state not in (PipelineState.IMPLEMENTATION, PipelineState.RETRY):
                raise GateViolationError(
                    f"MCP operation '{operation_name}' is forbidden in state '{state.name}'. "
                    f"Code modification is only allowed during IMPLEMENTATION or RETRY."
                )

        # Gate 2: Static verification tools only permitted in STATIC_VERIFY
        static_tools = {"run_mypy", "run_ruff_check", "static_analysis"}
        if operation_name in static_tools:
            if state != PipelineState.STATIC_VERIFY:
                raise GateViolationError(
                    f"MCP static analysis tool '{operation_name}' is forbidden in state '{state.name}'. "
                    f"Must be in STATIC_VERIFY state."
                )

        # Gate 3: Test runner tools only permitted in DYNAMIC_VERIFY
        test_tools = {"run_pytest", "run_test_suite", "dynamic_test"}
        if operation_name in test_tools:
            if state != PipelineState.DYNAMIC_VERIFY:
                raise GateViolationError(
                    f"MCP test tool '{operation_name}' is forbidden in state '{state.name}'. "
                    f"Must be in DYNAMIC_VERIFY state."
                )

        # Gate 4: PR Merge and submission only permitted in ACCEPT or COMPLETE
        deploy_tools = {"git_merge_pr", "publish_release", "finalize_submission"}
        if operation_name in deploy_tools:
            if state not in (PipelineState.ACCEPT, PipelineState.COMPLETE):
                raise GateViolationError(
                    f"MCP deployment/merge operation '{operation_name}' is forbidden in state '{state.name}'. "
                    f"Must reach ACCEPT or COMPLETE."
                )

        # If registered handler exists, execute it
        if operation_name in self.mcp_handlers:
            return self.mcp_handlers[operation_name](**kwargs)

        return {"status": "ok", "operation": operation_name, "state": state.name}
