import subprocess
import json
import os
from pathlib import Path
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

mcp = FastMCP("isl-linter-test")
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

SANDBOX_IMAGE = os.environ.get("ISL_SANDBOX_IMAGE", "isl-sandbox")

def run_in_docker_sandbox(cmd: list[str]) -> dict:
    """Execute command strictly inside the sealed ephemeral Docker sandbox.
    
    Security requirements:
    - --network=none (Strictly isolated from network)
    - --read-only (Read-only root filesystem)
    - --tmpfs /tmp (Volatile temporary memory only)
    - --security-opt no-new-privileges (Prevent privilege escalation)
    - Host project directories mounted read-only (:ro)
    """
    docker_cmd = [
        "docker", "run", "--rm",
        "--network=none",
        "--read-only",
        "--tmpfs", "/tmp",
        "--security-opt", "no-new-privileges",
        "-v", f"{PROJECT_ROOT / 'src'}:/workspace/src:ro",
        "-v", f"{PROJECT_ROOT / 'tests'}:/workspace/tests:ro",
        "-v", f"{PROJECT_ROOT / 'configs'}:/workspace/configs:ro",
        SANDBOX_IMAGE
    ] + cmd

    try:
        result = subprocess.run(
            docker_cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300
        )
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": "Completed successfully" if result.returncode == 0 else "Completed with errors",
            "execution_boundary": "docker-sealed-sandbox"
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Timeout expired in sealed sandbox",
            "summary": "Timeout",
            "execution_boundary": "docker-sealed-sandbox"
        }
    except FileNotFoundError:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": "Docker executable not found on host. The execution plane requires Docker for isolation.",
            "summary": "Docker execution boundary failure",
            "execution_boundary": "docker-sealed-sandbox"
        }
    except Exception as e:
        return {
            "exit_code": 1,
            "stdout": "",
            "stderr": f"Error launching sealed Docker sandbox: {str(e)}",
            "summary": "Sandbox execution error",
            "execution_boundary": "docker-sealed-sandbox"
        }

@mcp.tool()
def run_pytest(test_path: str = 'tests/', markers: str = '', verbose: bool = True) -> str:
    """Run pytest inside the sealed ephemeral Docker sandbox (--network=none)."""
    cmd = ["pytest", test_path]
    if markers:
        cmd.extend(["-m", markers])
    if verbose:
        cmd.append("-v")
    res = run_in_docker_sandbox(cmd)
    
    summary = "Tests ran in sealed sandbox."
    if res.get("exit_code") != 0:
        summary = "Tests failed in sealed sandbox."
    elif "passed" in res.get("stdout", ""):
        summary = "All tests passed in sealed sandbox."
    res["summary"] = summary
    return json.dumps(res)

@mcp.tool()
def run_mypy(target: str = 'src/') -> str:
    """Run mypy type checker inside the sealed ephemeral Docker sandbox."""
    cmd = ["mypy", "--ignore-missing-imports"] + target.split()
    return json.dumps(run_in_docker_sandbox(cmd))

@mcp.tool()
def run_ruff_check(target: str = 'src/ tests/') -> str:
    """Run ruff linter inside the sealed ephemeral Docker sandbox."""
    cmd = ["ruff", "check"] + target.split()
    return json.dumps(run_in_docker_sandbox(cmd))

@mcp.tool()
def run_ruff_format(target: str = 'src/ tests/', check_only: bool = False) -> str:
    """Run ruff format checker inside the sealed ephemeral Docker sandbox."""
    cmd = ["ruff", "format"] + target.split()
    if check_only:
        cmd.append("--check")
    return json.dumps(run_in_docker_sandbox(cmd))

if __name__ == "__main__":
    mcp.run()
