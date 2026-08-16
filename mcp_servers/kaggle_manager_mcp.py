import json
from pathlib import Path
import re
from typing import List, Optional

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    from fastmcp import FastMCP

from kaggle.account_broker import AccountBroker
from kaggle.kernel_manager import KernelManager
from src.orchestrator.gatekeeper import PipelineGate
from src.orchestrator.state_machine import PipelineState

mcp = FastMCP("isl-kaggle-manager")
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
gate = PipelineGate(str(PROJECT_ROOT))
broker = AccountBroker(credentials_dir=str(PROJECT_ROOT / "credentials"))
manager = KernelManager(broker=broker, project_root=str(PROJECT_ROOT))


def sanitize(text: str) -> str:
    # Redact sensitive values
    text = re.sub(r"(?i)(key|secret|password|token)[=:]\s*[\w-]+", r"\1=[REDACTED]", text)
    text = re.sub(r"[a-zA-Z0-9_-]{32,}", "***", text)
    return text


@mcp.tool()
def submit_kernel(
    kernel_name: str,
    script_path: str,
    dataset_sources: Optional[List[str]] = None,
    competition: Optional[str] = None,
    gpu: bool = True,
    internet: bool = False,
) -> str:
    """Submit a Kaggle kernel for remote GPU execution.

    Gated: Submissions only permitted in ACCEPT (approved experiment) or RETRY (diagnostic run).
    """
    gate.require_state("kaggle.submit", [PipelineState.ACCEPT, PipelineState.RETRY])

    try:
        submission = manager.submit(
            kernel_name=kernel_name,
            script_path=script_path,
            dataset_sources=dataset_sources,
            competition=competition,
            gpu=gpu,
            internet=internet,
        )
        return json.dumps(
            {
                "status": "submitted",
                "kernel_ref": submission.kernel_ref,
                "account_id": submission.account_id,
                "submit_time": submission.submit_time,
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_status(kernel_ref: str) -> str:
    """Get kernel execution status and release account if in terminal state."""
    try:
        status_info = manager.get_status(kernel_ref)
        return json.dumps(status_info)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_logs(kernel_ref: str, tail: int = 100) -> str:
    """Get sanitized kernel output logs."""
    try:
        logs = manager.get_logs(kernel_ref, tail=tail)
        lines = logs.split("\n")[-tail:]
        return sanitize("\n".join(lines))
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def get_output(kernel_ref: str, output_dir: Optional[str] = None) -> str:
    """Download kernel output files and checkpoints."""
    try:
        files = manager.get_output(kernel_ref, output_dir=output_dir)
        return json.dumps({"status": "success", "downloaded_files": files})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def cancel(kernel_ref: str) -> str:
    """Cancel a running kernel and guarantee account release."""
    gate.require_state("kaggle.cancel", [PipelineState.DYNAMIC_VERIFY, PipelineState.RETRY, PipelineState.HUMAN_GATE])

    try:
        success = manager.cancel(kernel_ref)
        return json.dumps({"status": "success", "cancelled": success})
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def select_account(account_id: Optional[str] = None) -> str:
    """Select or auto-select a Kaggle account (does not expose credentials)."""
    try:
        acc = broker.select_account(account_id)
        return json.dumps(
            {
                "status": "selected",
                "account_id": acc.account_id,
                "username": acc.username,
                "account_status": acc.status.value,
                "active_kernels": acc.active_kernels,
            }
        )
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def account_status() -> str:
    """Report status of all accounts from SQLite durable store."""
    try:
        status_info = broker.get_status()
        return json.dumps(status_info)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})


if __name__ == "__main__":
    mcp.run()
