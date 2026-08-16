"""Kaggle Kernel Lifecycle Manager.

Manages kernel metadata preparation, submission, status polling, log retrieval,
and guaranteed account release on terminal execution states.
"""

import json
import logging
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional

from kaggle.account_broker import AccountBroker

logger = logging.getLogger(__name__)


class KernelSubmission:
    def __init__(self, kernel_ref: str, account_id: str, submit_time: float, status: str = "queued", output_dir: Optional[str] = None):
        self.kernel_ref = kernel_ref
        self.account_id = account_id
        self.submit_time = submit_time
        self.status = status
        self.output_dir = output_dir


class KernelManager:
    def __init__(self, broker: AccountBroker, project_root: str):
        self.broker = broker
        self.project_root = Path(project_root).resolve()

    def prepare_metadata(
        self,
        kernel_name: str,
        script_path: str,
        dataset_sources: Optional[List[str]] = None,
        competition: Optional[str] = None,
        gpu: bool = True,
        internet: bool = False,
    ) -> Dict[str, Any]:
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
            "kernel_sources": [],
        }

    def submit(self, kernel_name: str, script_path: str, **kwargs: Any) -> KernelSubmission:
        """Submit kernel to Kaggle with SUBMITTING -> QUEUED lifecycle transitions."""
        account = self.broker.select_account()
        metadata = self.prepare_metadata(kernel_name, script_path, **kwargs)
        kernel_ref = f"{account.username}/{kernel_name}"
        metadata["id"] = kernel_ref

        kernel_dir = self.project_root / "temp_kernel"
        kernel_dir.mkdir(exist_ok=True)

        with open(kernel_dir / "kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        script_dest = kernel_dir / Path(script_path).name
        target_script = Path(script_path)
        if not target_script.is_absolute():
            target_script = self.project_root / script_path

        if target_script.exists():
            shutil.copy2(str(target_script), str(script_dest))

        # 1. Record SUBMITTING state before remote push
        self.broker.state_store.reserve_and_record_submitting(
            kernel_ref=kernel_ref,
            account_id=account.account_id,
        )

        try:
            proc = self._run_kaggle_cmd(["kaggle", "kernels", "push", "-p", str(kernel_dir)], account.account_id)
            if proc.returncode != 0 and "successfully" not in proc.stdout.lower():
                logger.error("Kaggle push failed: %s (stderr: %s)", proc.stdout, proc.stderr)
                self.broker.state_store.update_job_status(kernel_ref, "error")
                raise RuntimeError(f"Kaggle push failed: {proc.stderr or proc.stdout}")

            # 2. Transition to QUEUED upon successful push
            self.broker.state_store.mark_job_queued(kernel_ref)
        except Exception as e:
            logger.error("Failed to push kernel: %s", str(e))
            self.broker.state_store.update_job_status(kernel_ref, "error")
            raise

        return KernelSubmission(
            kernel_ref=kernel_ref,
            account_id=account.account_id,
            submit_time=time.time(),
            status="queued",
        )

    def get_status(self, kernel_ref: str) -> Dict[str, Any]:
        """Query kernel execution status and release account if terminal."""
        job = self.broker.state_store.get_job(kernel_ref)
        if not job:
            account_username = kernel_ref.split("/")[0]
            account_match = next((a for a in self.broker.accounts.values() if a.username == account_username), None)
            if not account_match:
                return {"status": "unknown", "error": "Account not found"}
            account_id = account_match.account_id
        else:
            account_id = job["account_id"]

        proc = self._run_kaggle_cmd(["kaggle", "kernels", "status", kernel_ref], account_id)
        raw_status = proc.stdout.strip()

        status = "running"
        lower_output = raw_status.lower()
        if "complete" in lower_output:
            status = "complete"
        elif "error" in lower_output or "failed" in lower_output:
            status = "error"
        elif "cancel" in lower_output:
            status = "cancelled"
        elif "queued" in lower_output:
            status = "queued"

        # Update SQLite store - guarantees account release on terminal states
        self.broker.state_store.update_job_status(kernel_ref, status)

        return {
            "kernel_ref": kernel_ref,
            "status": status,
            "raw_output": raw_status,
            "account_id": account_id,
        }

    def get_logs(self, kernel_ref: str, tail: int = 100) -> str:
        job = self.broker.state_store.get_job(kernel_ref)
        account_id = job["account_id"] if job else None
        if not account_id:
            account_username = kernel_ref.split("/")[0]
            acc = next((a for a in self.broker.accounts.values() if a.username == account_username), None)
            account_id = acc.account_id if acc else list(self.broker.accounts.keys())[0]

        proc = self._run_kaggle_cmd(["kaggle", "kernels", "output", kernel_ref], account_id)
        return self._sanitize_output(proc.stdout)

    def get_output(self, kernel_ref: str, output_dir: Optional[str] = None) -> List[str]:
        job = self.broker.state_store.get_job(kernel_ref)
        account_id = job["account_id"] if job else None
        if not account_id:
            account_username = kernel_ref.split("/")[0]
            acc = next((a for a in self.broker.accounts.values() if a.username == account_username), None)
            account_id = acc.account_id if acc else list(self.broker.accounts.keys())[0]

        out_path = Path(output_dir) if output_dir else self.project_root / "results" / "kaggle_output"
        out_path.mkdir(parents=True, exist_ok=True)

        self._run_kaggle_cmd(["kaggle", "kernels", "output", kernel_ref, "-p", str(out_path)], account_id)
        return [str(p) for p in out_path.rglob("*") if p.is_file()]

    def cancel(self, kernel_ref: str) -> Dict[str, Any]:
        """Perform real remote Kaggle cancellation and release account upon confirmation."""
        job = self.broker.state_store.get_job(kernel_ref)
        account_id = job["account_id"] if job else None
        if not account_id:
            account_username = kernel_ref.split("/")[0]
            acc = next((a for a in self.broker.accounts.values() if a.username == account_username), None)
            account_id = acc.account_id if acc else list(self.broker.accounts.keys())[0]

        proc = self._run_kaggle_cmd(["kaggle", "kernels", "cancel", kernel_ref], account_id)
        
        # Verify if remote cancellation succeeded
        output = proc.stdout.strip()
        err = proc.stderr.strip()
        lower_combined = (output + " " + err).lower()

        if proc.returncode == 0 or "cancelled" in lower_combined or "canceled" in lower_combined:
            self.broker.state_store.update_job_status(kernel_ref, "cancelled")
            logger.info("Successfully cancelled kernel %s remotely and released account %s", kernel_ref, account_id)
            return {"status": "cancelled", "remote_success": True, "output": output}
        else:
            logger.warning("Remote cancel command failed for %s: %s (err: %s)", kernel_ref, output, err)
            return {"status": "error", "remote_success": False, "error": err or output}

    def _sanitize_output(self, text: str) -> str:
        text = re.sub(r"[a-zA-Z0-9_-]{32,}", "***", text)
        return text

    def _run_kaggle_cmd(self, args: List[str], account_id: str) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        try:
            auth_env = self.broker.setup_auth_env(account_id)
            env.update(auth_env)
        except Exception as e:
            logger.warning("Could not set up Kaggle auth env: %s", str(e))

        result = subprocess.run(args, capture_output=True, text=True, check=False, env=env)
        return result
