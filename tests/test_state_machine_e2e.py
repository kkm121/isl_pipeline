"""End-to-End State Machine Enforcement Tests.

Proves programmatically that the orchestration controller strictly blocks any attempts
to skip mandatory pipeline gates (SPECIFICATION, TEST_PLAN, STATIC_VERIFY, DYNAMIC_VERIFY,
DIFF_VERIFY, INDEPENDENT_REVIEW, ACCEPT) or self-approve work.
"""

from unittest.mock import patch

import pytest

from src.orchestrator.state_machine import (
    HumanGateReason,
    InvalidTransitionError,
    PipelineState,
    PipelineStateMachine,
)


def test_cannot_skip_specification_from_idle():
    sm = PipelineStateMachine()
    assert sm.get_state() == PipelineState.IDLE

    # Attempt illegal shortcuts directly from IDLE
    illegal_targets = [
        PipelineState.TEST_PLAN,
        PipelineState.IMPLEMENTATION,
        PipelineState.STATIC_VERIFY,
        PipelineState.DYNAMIC_VERIFY,
        PipelineState.DIFF_VERIFY,
        PipelineState.INDEPENDENT_REVIEW,
        PipelineState.ACCEPT,
        PipelineState.COMPLETE,
    ]
    for target in illegal_targets:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from IDLE"):
            sm.transition(target, {}, agent="rogue_agent")
        assert sm.get_state() == PipelineState.IDLE


def test_cannot_skip_test_plan_from_specification():
    sm = PipelineStateMachine()
    sm.transition(PipelineState.SPECIFICATION, {"spec": "valid_spec"}, agent="principal_engineer")

    # Attempt to jump straight to code without test plan
    illegal_targets = [
        PipelineState.IMPLEMENTATION,
        PipelineState.STATIC_VERIFY,
        PipelineState.DYNAMIC_VERIFY,
        PipelineState.DIFF_VERIFY,
        PipelineState.INDEPENDENT_REVIEW,
        PipelineState.ACCEPT,
        PipelineState.COMPLETE,
    ]
    for target in illegal_targets:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from SPECIFICATION"):
            sm.transition(target, {}, agent="rogue_agent")


def test_cannot_skip_static_verify_from_implementation():
    sm = PipelineStateMachine()
    sm.transition(PipelineState.SPECIFICATION, {"spec": "valid_spec"}, agent="principal_engineer")
    sm.transition(PipelineState.TEST_PLAN, {"test_plan": "tdd_tests"}, agent="test_engineer")
    sm.transition(PipelineState.IMPLEMENTATION, {"code": "new_model.py"}, agent="principal_engineer")

    # Attempt to skip static analysis (mypy/ruff)
    illegal_targets = [
        PipelineState.DYNAMIC_VERIFY,
        PipelineState.DIFF_VERIFY,
        PipelineState.INDEPENDENT_REVIEW,
        PipelineState.ACCEPT,
        PipelineState.COMPLETE,
    ]
    for target in illegal_targets:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from IMPLEMENTATION"):
            sm.transition(target, {}, agent="rogue_agent")


def test_cannot_skip_dynamic_verify_from_static_verify():
    sm = PipelineStateMachine()
    sm.transition(PipelineState.SPECIFICATION, {"spec": "valid_spec"}, agent="principal_engineer")
    sm.transition(PipelineState.TEST_PLAN, {"test_plan": "tdd_tests"}, agent="test_engineer")
    sm.transition(PipelineState.IMPLEMENTATION, {"code": "new_model.py"}, agent="principal_engineer")
    sm.transition(PipelineState.STATIC_VERIFY, {"mypy": "clean", "ruff": "clean"}, agent="test_engineer")

    # Attempt to skip running pytest
    illegal_targets = [
        PipelineState.DIFF_VERIFY,
        PipelineState.INDEPENDENT_REVIEW,
        PipelineState.ACCEPT,
        PipelineState.COMPLETE,
    ]
    for target in illegal_targets:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from STATIC_VERIFY"):
            sm.transition(target, {}, agent="rogue_agent")


def test_cannot_skip_diff_verify_from_dynamic_verify():
    sm = PipelineStateMachine()
    sm.transition(PipelineState.SPECIFICATION, {"spec": "valid_spec"}, agent="principal_engineer")
    sm.transition(PipelineState.TEST_PLAN, {"test_plan": "tdd_tests"}, agent="test_engineer")
    sm.transition(PipelineState.IMPLEMENTATION, {"code": "new_model.py"}, agent="principal_engineer")
    sm.transition(PipelineState.STATIC_VERIFY, {"mypy": "clean"}, agent="test_engineer")
    sm.transition(PipelineState.DYNAMIC_VERIFY, {"pytest": "all passed"}, agent="test_engineer")

    # Attempt to skip Git diff verification
    illegal_targets = [
        PipelineState.INDEPENDENT_REVIEW,
        PipelineState.ACCEPT,
        PipelineState.COMPLETE,
    ]
    for target in illegal_targets:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from DYNAMIC_VERIFY"):
            sm.transition(target, {}, agent="rogue_agent")


def test_cannot_skip_independent_review_from_diff_verify():
    sm = PipelineStateMachine()
    sm.transition(PipelineState.SPECIFICATION, {"spec": "valid_spec"}, agent="principal_engineer")
    sm.transition(PipelineState.TEST_PLAN, {"test_plan": "tdd_tests"}, agent="test_engineer")
    sm.transition(PipelineState.IMPLEMENTATION, {"code": "new_model.py"}, agent="principal_engineer")
    sm.transition(PipelineState.STATIC_VERIFY, {"mypy": "clean"}, agent="test_engineer")
    sm.transition(PipelineState.DYNAMIC_VERIFY, {"pytest": "all passed"}, agent="test_engineer")

    # Pass DIFF_VERIFY with mocked valid tree diff
    with patch.object(sm, "verify_git_diff", return_value=(True, {"diff_stat": "1 file changed"})):
        sm.transition(PipelineState.DIFF_VERIFY, {"expected_changes": True}, agent="reviewer")
        assert sm.get_state() == PipelineState.DIFF_VERIFY

    # Attempt to skip independent review
    illegal_targets = [
        PipelineState.ACCEPT,
        PipelineState.COMPLETE,
    ]
    for target in illegal_targets:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from DIFF_VERIFY"):
            sm.transition(target, {}, agent="rogue_agent")


def test_full_pipeline_sequential_gate_progression():
    """Verify that an agent can only complete work by passing every single gate in strict order."""
    sm = PipelineStateMachine()

    # 1. SPECIFICATION
    assert sm.transition(PipelineState.SPECIFICATION, {"spec": "ISL architecture spec"}, agent="principal_engineer")
    assert sm.get_state() == PipelineState.SPECIFICATION

    # 2. TEST_PLAN
    assert sm.transition(PipelineState.TEST_PLAN, {"plan": "TDD unit tests"}, agent="test_engineer")
    assert sm.get_state() == PipelineState.TEST_PLAN

    # 3. IMPLEMENTATION
    assert sm.transition(PipelineState.IMPLEMENTATION, {"code": "implemented module"}, agent="principal_engineer")
    assert sm.get_state() == PipelineState.IMPLEMENTATION

    # 4. STATIC_VERIFY
    assert sm.transition(PipelineState.STATIC_VERIFY, {"mypy": "pass", "ruff": "pass"}, agent="test_engineer")
    assert sm.get_state() == PipelineState.STATIC_VERIFY

    # 5. DYNAMIC_VERIFY
    assert sm.transition(PipelineState.DYNAMIC_VERIFY, {"pytest": "45 passed"}, agent="test_engineer")
    assert sm.get_state() == PipelineState.DYNAMIC_VERIFY

    # 6. DIFF_VERIFY (with non-empty tree diff)
    with patch.object(sm, "verify_git_diff", return_value=(True, {"diff_stat": "2 files modified"})):
        assert sm.transition(PipelineState.DIFF_VERIFY, {"expected_changes": True}, agent="reviewer")
        assert sm.get_state() == PipelineState.DIFF_VERIFY

    # 7. INDEPENDENT_REVIEW
    assert sm.transition(PipelineState.INDEPENDENT_REVIEW, {"review": "adversarial review approved"}, agent="reviewer")
    assert sm.get_state() == PipelineState.INDEPENDENT_REVIEW

    # 8. ACCEPT
    assert sm.transition(PipelineState.ACCEPT, {"decision": "merge approved"}, agent="principal_engineer")
    assert sm.get_state() == PipelineState.ACCEPT

    # 9. COMPLETE
    assert sm.transition(PipelineState.COMPLETE, {"pr_url": "https://github.com/..."}, agent="principal_engineer")
    assert sm.get_state() == PipelineState.COMPLETE

    # 10. Return to IDLE for next task
    assert sm.transition(PipelineState.IDLE, {"task": "ready for next task"}, agent="orchestrator")
    assert sm.get_state() == PipelineState.IDLE


def test_human_gate_is_strictly_terminal():
    sm = PipelineStateMachine()
    sm.transition(PipelineState.SPECIFICATION, {"spec": "uncertain spec"}, agent="principal_engineer")
    sm.transition(
        PipelineState.HUMAN_GATE,
        {"reason": HumanGateReason.UNCERTAIN_DIAGNOSIS.name},
        agent="principal_engineer",
    )
    assert sm.get_state() == PipelineState.HUMAN_GATE

    # Agent cannot transition out of HUMAN_GATE to any state
    for target in PipelineState:
        with pytest.raises(InvalidTransitionError, match="Cannot transition from HUMAN_GATE"):
            sm.transition(target, {}, agent="autonomous_agent")
