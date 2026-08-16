import json
import re
from typing import Optional
from fastmcp import FastMCP
from kaggle.account_broker import AccountBroker
from kaggle.kernel_manager import KernelManager

mcp = FastMCP("isl-kaggle-manager")
broker = AccountBroker()
manager = KernelManager()

def sanitize(text: str) -> str:
    return re.sub(r'(?i)(key|secret|password|token)[=:]\s*\w+', r'\1=[REDACTED]', text)

@mcp.tool()
def submit_kernel(kernel_name: str, script_path: str, dataset_sources: list[str] = None, competition: str = None, gpu: bool = True, internet: bool = False) -> str:
    """Submit a Kaggle kernel."""
    acc = broker.get_account()
    ref = manager.push(acc, kernel_name, script_path, dataset_sources, competition, gpu, internet)
    return json.dumps({"kernel_ref": ref})

@mcp.tool()
def get_status(kernel_ref: str) -> str:
    """Get kernel execution status."""
    return json.dumps(manager.status(kernel_ref))

@mcp.tool()
def get_logs(kernel_ref: str, tail: int = 100) -> str:
    """Get kernel output logs."""
    logs = manager.logs(kernel_ref)
    lines = logs.split("\n")[-tail:]
    return sanitize("\n".join(lines))

@mcp.tool()
def get_output(kernel_ref: str, output_dir: str = None) -> str:
    """Download kernel output files."""
    files = manager.download(kernel_ref, output_dir)
    return json.dumps({"downloaded_files": files})

@mcp.tool()
def cancel(kernel_ref: str) -> str:
    """Cancel a running kernel."""
    res = manager.cancel(kernel_ref)
    return json.dumps({"cancelled": res})

@mcp.tool()
def select_account(account_id: str = None) -> str:
    """Select Kaggle account."""
    acc = broker.get_account(account_id)
    # Assuming acc object has an identifier attribute
    return json.dumps({"selected_account": getattr(acc, 'identifier', str(acc))})

@mcp.tool()
def account_status() -> str:
    """Report status of all accounts."""
    return json.dumps(broker.status())

if __name__ == "__main__":
    mcp.run()
