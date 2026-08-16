import subprocess
import json
from pathlib import Path
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

mcp = FastMCP("isl-github")
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def run_cmd(cmd: list[str]) -> dict:
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True)
        return {
            "exit_code": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip()
        }
    except Exception as e:
        return {
            "exit_code": -1,
            "stdout": "",
            "stderr": str(e)
        }

@mcp.tool()
def create_issue(title: str, body: str, labels: list[str] = None) -> str:
    """Create a GitHub issue."""
    cmd = ["gh", "issue", "create", "--title", title, "--body", body]
    if labels:
        cmd.extend(["--label", ",".join(labels)])
    return json.dumps(run_cmd(cmd))

@mcp.tool()
def list_issues(state: str = 'open', labels: str = None) -> str:
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
def create_branch(branch_name: str, base: str = 'main') -> str:
    """Create and checkout a new branch."""
    subprocess.run(["git", "checkout", base], cwd=str(PROJECT_ROOT))
    subprocess.run(["git", "pull"], cwd=str(PROJECT_ROOT))
    cmd = ["git", "checkout", "-b", branch_name]
    return json.dumps(run_cmd(cmd))

@mcp.tool()
def commit(message: str, files: list[str] = None) -> str:
    """Stage files and commit."""
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
def create_pr(title: str, body: str, base: str = 'main') -> str:
    """Create a pull request."""
    cmd = ["gh", "pr", "create", "--title", title, "--body", body, "--base", base]
    return json.dumps(run_cmd(cmd))

@mcp.tool()
def get_diff(base: str = 'HEAD~1', target: str = 'HEAD') -> str:
    """Get git diff."""
    cmd = ["git", "diff", f"{base}..{target}"]
    return json.dumps(run_cmd(cmd))

if __name__ == "__main__":
    mcp.run()
