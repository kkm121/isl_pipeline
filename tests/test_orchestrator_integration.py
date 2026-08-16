"""Integration tests verifying that PipelineController and State Machine actively gate all agent MCP operations."""

import pytest

from src.orchestrator.pipeline_controller import GateViolationError, PipelineController
from src.orchestrator.state_machine import InvalidTransitionError, PipelineState


@pytest.fixture
def controller(tmp_path):
    """Create a clean pipeline controller rooted in a temporary workspace."""
    return PipelineController(project_root=str(tmp_path))


def test_mcp_write_forbidden_before_implementation(controller):
    """Verify that file modification MCP operations are strictly forbidden in IDLE/SPECIFICATION/TEST_PLAN."""
    assert controller.current_state == PipelineState.IDLE

    # In IDLE: Write must fail
    with pytest.raises(GateViolationError) as exc:
        controller.execute_mcp_operation("write_file", path="src/model.py", content="code")
    assert "forbidden in state 'IDLE'" in str(exc.value)

    # In SPECIFICATION: Write must fail
    controller.start_specification({"objective": "ISL", "requirements": ["MediaPipe"]})
    assert controller.current_state == PipelineState.SPECIFICATION
    with pytest.raises(GateViolationError) as exc:
        controller.execute_mcp_operation("edit_file", path="src/model.py")
    assert "forbidden in state 'SPECIFICATION'" in str(exc.value)

    # In TEST_PLAN: Write must fail
    controller.submit_test_plan({"test_cases": ["test_forward"], "acceptance_criteria": "pass 100%"})
    assert controller.current_state == PipelineState.TEST_PLAN
    with pytest.raises(GateViolationError) as exc:
        controller.execute_mcp_operation("replace_content", path="src/model.py")
    assert "forbidden in state 'TEST_PLAN'" in str(exc.value)

    # In IMPLEMENTATION: Write is allowed
    controller.begin_implementation({"feature": "model"})
    assert controller.current_state == PipelineState.IMPLEMENTATION
    res = controller.execute_mcp_operation("write_file", path="src/model.py", content="class ISLModel: pass")
    assert res["status"] == "ok"


def test_mcp_static_and_dynamic_verification_gating(controller):
    """Verify that verification MCP tools can only execute during their designated verification states."""
    controller.start_specification({"objective": "ISL", "requirements": ["Test"]})
    controller.submit_test_plan({"test_cases": ["tc1"], "acceptance_criteria": "crit"})
    controller.begin_implementation({"feature": "code"})

    # In IMPLEMENTATION: Static and dynamic tools cannot run yet
    with pytest.raises(GateViolationError) as exc:
        controller.execute_mcp_operation("run_mypy")
    assert "Must be in STATIC_VERIFY" in str(exc.value)

    with pytest.raises(GateViolationError) as exc:
        controller.execute_mcp_operation("run_pytest")
    assert "Must be in DYNAMIC_VERIFY" in str(exc.value)

    # Transition to STATIC_VERIFY
    controller.run_static_verification()
    assert controller.current_state == PipelineState.STATIC_VERIFY

    # Now static analysis runs
    res_static = controller.execute_mcp_operation("run_mypy")
    assert res_static["status"] == "ok"

    # Pytest is still blocked until DYNAMIC_VERIFY
    with pytest.raises(GateViolationError):
        controller.execute_mcp_operation("run_pytest")

    # Transition to DYNAMIC_VERIFY
    controller.run_dynamic_verification()
    assert controller.current_state == PipelineState.DYNAMIC_VERIFY

    # Now pytest runs
    res_dynamic = controller.execute_mcp_operation("run_pytest")
    assert res_dynamic["status"] == "ok"


def test_mcp_deployment_and_merge_gating(controller):
    """Verify that merging and deployment operations are forbidden until ACCEPT or COMPLETE."""
    controller.start_specification({"objective": "ISL", "requirements": ["Test"]})
    controller.submit_test_plan({"test_cases": ["tc1"], "acceptance_criteria": "crit"})
    controller.begin_implementation({"feature": "code"})
    controller.run_static_verification()
    controller.run_dynamic_verification()

    # In DYNAMIC_VERIFY: Merge is forbidden
    with pytest.raises(GateViolationError):
        controller.execute_mcp_operation("git_merge_pr")


def test_full_mocked_agent_lifecycle_progression(controller, monkeypatch):
    """Run an entire simulated autonomous turn through the controller and verify every gate transitions legally."""
    # 1. Specification
    controller.start_specification(
        {"objective": "Implement ISL landmark encoder", "requirements": ["Transformer backbone", "MediaPipe 21 pts"]},
        agent="principal-engineer",
    )
    assert controller.current_state == PipelineState.SPECIFICATION

    # 2. Test Plan
    controller.submit_test_plan(
        {"test_cases": ["test_encoder_forward", "test_nan_prevention"], "acceptance_criteria": "100% test pass"},
        agent="test-engineer",
    )
    assert controller.current_state == PipelineState.TEST_PLAN

    # 3. Implementation
    controller.begin_implementation({"feature": "ISL landmark encoder"}, agent="code-writer")
    assert controller.current_state == PipelineState.IMPLEMENTATION

    # Simulate code writing in IMPLEMENTATION state
    controller.execute_mcp_operation("write_file", path="src/encoder.py", content="# encoder code")

    # 4. Static Verification
    def mock_static():
        return True, {"mypy": "passed", "ruff": "passed"}

    ok_static = controller.run_static_verification(static_verifier=mock_static, agent="test-engineer")
    assert ok_static is True
    assert controller.current_state == PipelineState.STATIC_VERIFY

    # 5. Dynamic Verification
    def mock_tests():
        return True, {"passed": 56, "failed": 0}

    ok_tests = controller.run_dynamic_verification(test_runner=mock_tests, agent="test-engineer")
    assert ok_tests is True
    assert controller.current_state == PipelineState.DYNAMIC_VERIFY

    # 6. Diff Verification (mock git diff to verify non-zero changes)
    monkeypatch.setattr(
        controller.state_machine,
        "verify_git_diff",
        lambda expected_modifications=True: (True, {"has_changes": True, "diff_stat": "1 file changed"}),
    )
    ok_diff = controller.run_diff_verification(expected_changes=True, agent="principal-engineer")
    assert ok_diff is True
    assert controller.current_state == PipelineState.DIFF_VERIFY

    # 7. Independent Review
    controller.submit_independent_review(
        {"approved": True, "checklist": ["security_ok", "tests_pass", "clean_architecture"]},
        agent="independent-reviewer",
    )
    assert controller.current_state == PipelineState.INDEPENDENT_REVIEW

    # 8. Accept & Complete
    controller.approve_and_accept({"decision": "merge_approved"}, agent="principal-engineer")
    assert controller.current_state == PipelineState.ACCEPT

    controller.complete_task({"summary": "Feature successfully implemented and verified"}, agent="principal-engineer")
    assert controller.current_state == PipelineState.COMPLETE


def test_illegal_shortcut_bypassing_specification_raises_error(controller):
    """Verify that an agent attempting to jump directly from IDLE to IMPLEMENTATION is blocked."""
    with pytest.raises(InvalidTransitionError):
        controller.begin_implementation({"feature": "direct_hack"})
