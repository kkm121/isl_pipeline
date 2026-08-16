import os
import json
from pathlib import Path
try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

mcp = FastMCP("isl-filesystem")

PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def validate_path(path: str) -> Path:
    target_path = Path(path)
    if not target_path.is_absolute():
        target_path = (PROJECT_ROOT / target_path).resolve()
    else:
        target_path = target_path.resolve()
        
    try:
        target_path.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Path traversal detected: {path}")
    return target_path

@mcp.tool()
def read_file(path: str) -> str:
    """Read file content."""
    safe_path = validate_path(path)
    if not safe_path.is_file():
        return f"Error: File not found {path}"
    with open(safe_path, 'r', encoding='utf-8') as f:
        return f.read()

@mcp.tool()
def write_file(path: str, content: str) -> str:
    """Write file."""
    safe_path = validate_path(path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(safe_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Success: Wrote to {path}"

@mcp.tool()
def list_directory(path: str) -> str:
    """List directory contents."""
    safe_path = validate_path(path)
    if not safe_path.is_dir():
        return json.dumps({"error": f"Directory not found {path}"})
    items = []
    for item in safe_path.iterdir():
        items.append({
            "name": item.name,
            "type": "directory" if item.is_dir() else "file",
            "size": item.stat().st_size if item.is_file() else 0
        })
    return json.dumps({"path": str(path), "items": items})

@mcp.tool()
def read_logs(log_type: str, n_lines: int = 100) -> str:
    """Read from logs/{log_type}/."""
    if log_type not in ["local", "kaggle", "agents"]:
        return "Error: Invalid log_type. Must be local, kaggle, or agents."
    log_dir = PROJECT_ROOT / "logs" / log_type
    if not log_dir.is_dir():
        return "Error: Log directory not found."
    
    logs = []
    for log_file in log_dir.glob("*.log"):
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            logs.append({"file": log_file.name, "lines": lines[-n_lines:]})
    return json.dumps(logs)

@mcp.tool()
def read_metrics(experiment_id: str) -> str:
    """Read metrics from metrics/ or experiments/{experiment_id}/."""
    metrics_path = PROJECT_ROOT / "metrics" / f"{experiment_id}.json"
    exp_path = PROJECT_ROOT / "experiments" / experiment_id / "metrics.json"
    
    if metrics_path.is_file():
        with open(metrics_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif exp_path.is_file():
        with open(exp_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        return json.dumps({"error": "Metrics not found"})

if __name__ == "__main__":
    mcp.run()
