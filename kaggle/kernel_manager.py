import json
import subprocess
import os
import time
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging
import re

from kaggle.account_broker import AccountBroker, AccountStatus

logger = logging.getLogger(__name__)

@dataclass
class KernelSubmission:
    kernel_ref: str
    account_id: str
    submit_time: float
    status: str = 'queued'
    output_dir: Optional[str] = None

class KernelManager:
    def __init__(self, broker: AccountBroker, project_root: str):
        self.broker = broker
        self.project_root = Path(project_root)

    def prepare_metadata(self, kernel_name: str, script_path: str, dataset_sources: List[str] = None, competition: str = None, gpu: bool = True, internet: bool = False) -> Dict:
        return {
            "id": f"USERNAME/{kernel_name}",
            "title": kernel_name,
            "code_file": script_path,
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": gpu,
            "enable_tpu": False,
            "enable_internet": internet,
            "dataset_sources": dataset_sources or [],
            "competition_sources": [competition] if competition else [],
            "kernel_sources": []
        }

    def submit(self, kernel_name: str, script_path: str, **kwargs) -> KernelSubmission:
        account = self.broker.select_account()
        metadata = self.prepare_metadata(kernel_name, script_path, **kwargs)
        metadata["id"] = f"{account.username}/{kernel_name}"
        
        kernel_dir = self.project_root / "temp_kernel"
        kernel_dir.mkdir(exist_ok=True)
        
        with open(kernel_dir / "kernel-metadata.json", "w") as f:
            json.dump(metadata, f)
            
        script_dest = kernel_dir / Path(script_path).name
        import shutil
        shutil.copy(script_path, script_dest)
        
        self._run_kaggle_cmd(['kaggle', 'kernels', 'push', '-p', str(kernel_dir)], account.account_id)
        
        return KernelSubmission(
            kernel_ref=metadata["id"],
            account_id=account.account_id,
            submit_time=time.time()
        )

    def get_status(self, kernel_ref: str) -> Dict:
        account_username = kernel_ref.split('/')[0]
        account_id = next(a.account_id for a in self.broker.accounts.values() if a.username == account_username)
        proc = self._run_kaggle_cmd(['kaggle', 'kernels', 'status', kernel_ref], account_id)
        return {"status": proc.stdout.strip()}

    def get_logs(self, kernel_ref: str, tail: int = 100) -> str:
        account_username = kernel_ref.split('/')[0]
        account_id = next(a.account_id for a in self.broker.accounts.values() if a.username == account_username)
        proc = self._run_kaggle_cmd(['kaggle', 'kernels', 'output', kernel_ref], account_id)
        return self._sanitize_output(proc.stdout)

    def get_output(self, kernel_ref: str, output_dir: str = None) -> List[str]:
        account_username = kernel_ref.split('/')[0]
        account_id = next(a.account_id for a in self.broker.accounts.values() if a.username == account_username)
        if not output_dir:
            output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        self._run_kaggle_cmd(['kaggle', 'kernels', 'output', kernel_ref, '-p', output_dir], account_id)
        return [str(p) for p in Path(output_dir).rglob('*') if p.is_file()]

    def cancel(self, kernel_ref: str) -> bool:
        return True # Mock implementation

    def _sanitize_output(self, text: str) -> str:
        text = re.sub(r'[a-zA-Z0-9]{32}', '***', text)
        return text

    def _run_kaggle_cmd(self, args: List[str], account_id: str) -> subprocess.CompletedProcess:
        self.broker._set_kaggle_env(account_id)
        return subprocess.run(args, capture_output=True, text=True, check=False)
