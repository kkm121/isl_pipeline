import json
from pathlib import Path
import subprocess

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

from src.orchestrator.gatekeeper import PipelineGate
from src.orchestrator.state_machine import (
    InvalidTransitionError,
    PipelineState,
)

mcp = FastMCP("isl-github")
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
gate = PipelineGate(str(PROJECT_ROOT))


def run_cmd(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        return {"exit_code": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except Exception as e:
        return {"exit_code": -1, "stdout": "", "stderr": str(e)}


@mcp.tool()
def advance_pipeline(target_state: str, evidence: dict, agent: str) -> str:
    """Request a deterministic state transition in the authoritative pipeline gatekeeper.

    The state machine enforces all transition invariants, captures baselines, and rejects illegal jumps.
    """
    try:
        target = PipelineState[target_state]
        new_state = gate.transition(target=target, evidence=evidence, agent=agent)
        return json.dumps({"status": "success", "state": new_state.name})
    except KeyError:
        return json.dumps({"status": "rejected", "error": f"Unknown state name '{target_state}'"})
    except InvalidTransitionError as exc:
        return json.dumps(
            {
                "status": "rejected",
                "error": str(exc),
                "current_state": gate.current_state().name,
            }
        )


@mcp.tool()
def get_pipeline_status() -> str:
    """Get the current authoritative pipeline state snapshot from SQLite durable store."""
    return json.dumps(gate.get_state_snapshot())


@mcp.tool()
def create_issue(title: str, body: str, labels: list[str] = None) -> str:
    """Create a GitHub issue."""
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    return json.dumps(run_cmd(cmd))


@mcp.tool()
def list_issues(state: str = "open", labels: str = None) -> str:
    """List issues."""
    cmd = ["gh", "issue", "list", "--state", state, "--json", "number,title,state,url"]
    if labels:
        cmd.extend(["--label", labels])
    res = run_cmd(cmd)
    try:
        if res["exit_code"] == 0:
            res["stdout"] = json.loads(res["stdout"])
    except json.JSONDecodeError:
        pass
    return json.dumps(res)


@mcp.tool()
def create_branch(branch_name: str, base: str = "main") -> str:
    """Create and checkout a new branch (Allowed in IMPLEMENTATION and RETRY states)."""
    gate.require_state(
        "github.create_branch",
        [PipelineState.IMPLEMENTATION, PipelineState.RETRY],
    )
    subprocess.run(["git", "checkout", base], cwd=str(PROJECT_ROOT))
    subprocess.run(["git", "pull"], cwd=str(PROJECT_ROOT))
    cmd = ["git", "checkout", "-b", branch_name]
    return json.dumps(run_cmd(cmd))


@mcp.tool()
def commit(message: str, files: list[str] = None) -> str:
    """Stage files and commit.

    Gated: Commits are strictly forbidden until code passes independent review and reaches ACCEPT.
    """
    gate.require_state("github.commit", [PipelineState.ACCEPT, PipelineState.COMPLETE])

    if files is None:
        subprocess.run(["git", "add", "."], cwd=str(PROJECT_ROOT))
    else:
        for f in files:
            subprocess.run(["git", "add", f], cwd=str(PROJECT_ROOT))
    cmd = ["git", "commit", "-m", message]
    res = run_cmd(cmd)

    if res["exit_code"] == 0:
        hash_res = run_cmd(["git", "rev-parse", "HEAD"])
        res["commit_hash"] = hash_res["stdout"]

    return json.dumps(res)


@mcp.tool()
def create_pr(title: str, body: str, base: str = "main") -> str:
    """Create a pull request.

    Gated: PR creation is strictly forbidden until independent review approval (ACCEPT or COMPLETE).
    """
    gate.require_state("github.create_pr", [PipelineState.ACCEPT, PipelineState.COMPLETE])
    cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--base", base]
    return json.dumps(run_cmd(cmd))


@mcp.tool()
def get_diff(base: str = "HEAD~1", target: str = "HEAD") -> str:
    """Get git diff."""
    cmd = ["git", "diff", f"{base}..{target}"]
    return json.dumps(run_cmd(cmd))


if __name__ == "__main__":
    mcp.run()
