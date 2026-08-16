"""End-to-End integration tests proving that actual MCP tools route through the authoritative PipelineGate."""

import json
import subprocess
from unittest.mock import patch

import pytest

from mcp_servers import filesystem_mcp, github_mcp, kaggle_manager_mcp, linter_test_mcp
from src.orchestrator.gatekeeper import GateViolation, PipelineGate
from src.orchestrator.state_machine import PipelineState


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Set up isolated project workspace and point all MCP servers to the shared PipelineGate."""
    gate = PipelineGate(str(tmp_path))

    # Point all MCP server modules to the isolated workspace gate
    monkeypatch.setattr(filesystem_mcp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(filesystem_mcp, "gate", gate)

    monkeypatch.setattr(linter_test_mcp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(linter_test_mcp, "gate", gate)

    monkeypatch.setattr(github_mcp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(github_mcp, "gate", gate)

    monkeypatch.setattr(kaggle_manager_mcp, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(kaggle_manager_mcp, "gate", gate)

    return {"gate": gate, "root": tmp_path}


def test_filesystem_mcp_write_rejected_before_implementation(env):
    """Prove that filesystem_mcp.write_file is physically rejected before IMPLEMENTATION state."""
    gate = env["gate"]
    assert gate.current_state() == PipelineState.IDLE

    # 1. In IDLE: source file write must be rejected by gatekeeper
    with pytest.raises(GateViolation) as exc:
        filesystem_mcp.write_file("src/models/encoder.py", "class Encoder: pass")
    assert "filesystem.write_source" in str(exc.value)

    # 2. Advance to SPECIFICATION: source file write is still rejected
    gate.transition(PipelineState.SPECIFICATION, {"spec": "ISL"}, "principal-engineer")
    with pytest.raises(GateViolation):
        filesystem_mcp.write_file("src/models/encoder.py", "class Encoder: pass")

    # 3. Advance to TEST_PLAN: test files allowed, but source files are still rejected
    gate.transition(PipelineState.TEST_PLAN, {"plan": "TDD"}, "test-engineer")
    test_res = json.loads(filesystem_mcp.write_file("tests/test_encoder.py", "def test_f(): pass"))
    assert test_res["status"] == "success"

    with pytest.raises(GateViolation):
        filesystem_mcp.write_file("src/models/encoder.py", "class Encoder: pass")

    # 4. Advance to IMPLEMENTATION: source files are now allowed
    gate.transition(PipelineState.IMPLEMENTATION, {"code": "start"}, "code-writer")
    src_res = json.loads(filesystem_mcp.write_file("src/models/encoder.py", "class Encoder: pass"))
    assert src_res["status"] == "success"
    assert (env["root"] / "src/models/encoder.py").exists()


def test_linter_test_mcp_rejected_outside_designated_states(env):
    """Prove that test/linter MCP operations only execute in their designated verification states."""
    gate = env["gate"]
    gate.transition(PipelineState.SPECIFICATION, {"spec": "data"}, "principal-engineer")
    gate.transition(PipelineState.TEST_PLAN, {"plan": "tests"}, "test-engineer")
    gate.transition(PipelineState.IMPLEMENTATION, {"code": "write"}, "code-writer")

    # During IMPLEMENTATION: static analysis and test tools are forbidden
    with pytest.raises(GateViolation) as exc:
        linter_test_mcp.run_mypy()
    assert "verification.mypy" in str(exc.value)

    with pytest.raises(GateViolation) as exc:
        linter_test_mcp.run_pytest()
    assert "verification.pytest" in str(exc.value)

    # Advance to STATIC_VERIFY
    gate.transition(PipelineState.STATIC_VERIFY, {}, "test-engineer")

    # Static analysis tools now pass gate (mock sandbox runner)
    with patch.object(linter_test_mcp, "run_in_docker_sandbox", return_value={"exit_code": 0, "stdout": "ok"}):
        mypy_res = json.loads(linter_test_mcp.run_mypy())
        assert mypy_res["exit_code"] == 0

        ruff_res = json.loads(linter_test_mcp.run_ruff_check())
        assert ruff_res["exit_code"] == 0

    # Pytest is still strictly forbidden during STATIC_VERIFY
    with pytest.raises(GateViolation):
        linter_test_mcp.run_pytest()

    # Advance to DYNAMIC_VERIFY
    gate.transition(PipelineState.DYNAMIC_VERIFY, {}, "test-engineer")

    # Pytest now passes gate
    with patch.object(linter_test_mcp, "run_in_docker_sandbox", return_value={"exit_code": 0, "stdout": "1 passed"}):
        pytest_res = json.loads(linter_test_mcp.run_pytest())
        assert pytest_res["exit_code"] == 0


def test_github_mcp_gating_commit_and_pr(env):
    """Prove that git commit and PR creation are rejected until reaching ACCEPT."""
    gate = env["gate"]
    gate.transition(PipelineState.SPECIFICATION, {"spec": "data"}, "principal-engineer")
    gate.transition(PipelineState.TEST_PLAN, {"plan": "tests"}, "test-engineer")
    gate.transition(PipelineState.IMPLEMENTATION, {"code": "write"}, "code-writer")

    # In IMPLEMENTATION: branch creation is permitted
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
        branch_res = json.loads(github_mcp.create_branch("feature/isl-model"))
        assert "exit_code" in branch_res

    # In IMPLEMENTATION: commit and PR creation are forbidden
    with pytest.raises(GateViolation) as exc:
        github_mcp.commit("Premature commit")
    assert "github.commit" in str(exc.value)

    with pytest.raises(GateViolation) as exc:
        github_mcp.create_pr("Premature PR", "body")
    assert "github.create_pr" in str(exc.value)


def test_kaggle_mcp_submit_rejected_before_accept(env):
    """Prove that Kaggle kernel submission is rejected when pipeline is in an unapproved state."""
    gate = env["gate"]
    gate.transition(PipelineState.SPECIFICATION, {"spec": "data"}, "principal-engineer")

    # In SPECIFICATION: Kaggle submit must be blocked
    with pytest.raises(GateViolation) as exc:
        kaggle_manager_mcp.submit_kernel("test-kernel", "train.py")
    assert "kaggle.submit" in str(exc.value)


def test_mcp_advance_pipeline_full_valid_sequence(env):
    """Prove that the advance_pipeline MCP tool successfully advances the pipeline through all mandatory gates."""
    # 1. Advance to SPECIFICATION
    res1 = json.loads(github_mcp.advance_pipeline("SPECIFICATION", {"spec": "ISL recognition"}, "principal-engineer"))
    assert res1["status"] == "success"
    assert res1["state"] == "SPECIFICATION"

    # 2. Illegal jump to ACCEPT must be rejected
    illegal_res = json.loads(github_mcp.advance_pipeline("ACCEPT", {}, "principal-engineer"))
    assert illegal_res["status"] == "rejected"
    assert illegal_res["current_state"] == "SPECIFICATION"

    # 3. Advance through complete legal sequence
    res2 = json.loads(github_mcp.advance_pipeline("TEST_PLAN", {"tests": ["t1"]}, "test-engineer"))
    assert res2["status"] == "success"

    res3 = json.loads(github_mcp.advance_pipeline("IMPLEMENTATION", {"code": "impl"}, "code-writer"))
    assert res3["status"] == "success"

    res4 = json.loads(github_mcp.advance_pipeline("STATIC_VERIFY", {}, "test-engineer"))
    assert res4["status"] == "success"

    res5 = json.loads(github_mcp.advance_pipeline("DYNAMIC_VERIFY", {}, "test-engineer"))
    assert res5["status"] == "success"

    # Mock diff verification for DIFF_VERIFY step
    with patch("src.orchestrator.state_machine.PipelineStateMachine.verify_git_diff", return_value=(True, {})):
        res6 = json.loads(github_mcp.advance_pipeline("DIFF_VERIFY", {"expected_changes": True}, "principal-engineer"))
        assert res6["status"] == "success"

    res7 = json.loads(
        github_mcp.advance_pipeline(
            "INDEPENDENT_REVIEW", {"approved": True, "checklist": ["ok"]}, "independent-reviewer"
        )
    )
    assert res7["status"] == "success"

    res8 = json.loads(github_mcp.advance_pipeline("ACCEPT", {"approval": True}, "principal-engineer"))
    assert res8["status"] == "success"

    # In ACCEPT state: commit and PR creation now succeed
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0, stdout="f4b3e2", stderr="")
        commit_res = json.loads(github_mcp.commit("Verified commit"))
        assert "exit_code" in commit_res

    res9 = json.loads(github_mcp.advance_pipeline("COMPLETE", {"done": True}, "principal-engineer"))
    assert res9["status"] == "success"
    assert res9["state"] == "COMPLETE"
