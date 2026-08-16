import subprocess
import json
from pathlib import Path
from fastmcp import FastMCP

mcp = FastMCP("isl-linter-test")
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def run_cmd(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=300)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "summary": "Completed successfully" if result.returncode == 0 else "Completed with errors"
        }
    except subprocess.TimeoutExpired:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": "Timeout expired",
            "summary": "Timeout"
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e),
            "summary": "Error running command"
        }

@mcp.tool()
def run_pytest(test_path: str = 'tests/', markers: str = '', verbose: bool = True) -> str:
    """Run pytest with subprocess."""
    cmd = ["pytest", test_path]
    if markers:
        cmd.extend(["-m", markers])
    if verbose:
        cmd.append("-v")
    res = run_cmd(cmd)
    
    summary = "Tests ran."
    if "failed" in res["stdout"]:
        summary = "Some tests failed."
    elif "passed" in res["stdout"]:
        summary = "All tests passed."
    res["summary"] = summary
    return json.dumps(res)

@mcp.tool()
def run_mypy(target: str = 'src/') -> str:
    """Run mypy with --ignore-missing-imports."""
    cmd = ["mypy", "--ignore-missing-imports"] + target.split()
    return json.dumps(run_cmd(cmd))

@mcp.tool()
def run_ruff_check(target: str = 'src/ tests/') -> str:
    """Run ruff check."""
    cmd = ["ruff", "check"] + target.split()
    return json.dumps(run_cmd(cmd))

@mcp.tool()
def run_ruff_format(target: str = 'src/ tests/', check_only: bool = False) -> str:
    """Run ruff format (or --check)."""
    cmd = ["ruff", "format"] + target.split()
    if check_only:
        cmd.append("--check")
    return json.dumps(run_cmd(cmd))

if __name__ == "__main__":
    mcp.run()
