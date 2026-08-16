"""Kaggle Account Broker — Policy-Aware Credential Management with SQLite Persistence.

Critical security component. Raw credentials NEVER appear in:
- Agent context or prompts
- Docker containers (injected via KAGGLE_CONFIG_DIR at runtime)
- Git repository
- Logs or generated artifacts
- MCP tool responses
"""

import atexit
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import tempfile
import time
from typing import Any, Dict, Generator, List, Optional

from kaggle.state_store import KaggleStateStore

logger = logging.getLogger(__name__)


class AccountStatus(Enum):
    AVAILABLE = "available"
    IN_USE = "in_use"
    QUOTA_EXCEEDED = "quota_exceeded"
    ERROR = "error"
    ROTATION_BLOCKED = "rotation_blocked"


@dataclass
class KaggleAccount:
    """Account metadata. API key is NEVER stored here."""

    account_id: str
    username: str
    status: AccountStatus = AccountStatus.AVAILABLE
    last_used: float = 0.0
    active_kernels: int = 0
    max_concurrent: int = 2
    total_submissions: int = 0


@dataclass
class RotationPolicy:
    """Policy gate for account rotation."""

    rotation_permitted: bool = True
    max_rotations_per_task: int = 3
    current_rotations: int = 0
    require_human_approval_after: int = 2

    def can_rotate(self) -> bool:
        if not self.rotation_permitted:
            return False
        if self.current_rotations >= self.max_rotations_per_task:
            return False
        return True

    def needs_human_approval(self) -> bool:
        return self.current_rotations >= self.require_human_approval_after


class AccountBroker:
    """Manages Kaggle account credentials with SQLite durable state and policy-aware rotation."""

    def __init__(
        self,
        credentials_dir: str = "credentials/",
        rotation_policy: Optional[RotationPolicy] = None,
        db_path: Optional[str] = None,
    ):
        self.credentials_dir = Path(credentials_dir)
        db_file = db_path or str(self.credentials_dir / "kaggle_state.db")
        self.state_store = KaggleStateStore(db_path=db_file)
        self.rotation_policy = rotation_policy or RotationPolicy()
        self._active_config_dirs: Dict[str, str] = {}  # account_id -> temp dir path

        # Register cleanup hooks
        self._register_lifecycle_cleanups()

        # Startup cleanup: purge stale temporary auth directories from previous aborted runs
        self.cleanup_stale_temp_dirs()

        # Sync accounts & recover orphans
        self._load_and_sync_accounts()
        self.state_store.recover_orphaned_jobs()

    def _register_lifecycle_cleanups(self) -> None:
        """Register atexit and signal handlers to guarantee temporary directory deletion."""
        atexit.register(self.cleanup_all)

        # Handle termination signals where available
        for sig_name in ("SIGINT", "SIGTERM"):
            if hasattr(signal, sig_name):
                try:
                    sig = getattr(signal, sig_name)
                    prev_handler = signal.getsignal(sig)

                    def make_handler(old_h):
                        def _handler(signum, frame):
                            self.cleanup_all()
                            if callable(old_h):
                                old_h(signum, frame)

                        return _handler

                    signal.signal(sig, make_handler(prev_handler))
                except (ValueError, OSError):
                    # Signal registration can fail if not in main thread; ignore safely
                    pass

    def cleanup_stale_temp_dirs(self) -> int:
        """Remove any leftover kaggle_* temporary authentication directories from prior runs."""
        cleaned = 0
        tmp_dir = Path(tempfile.gettempdir())
        try:
            for p in tmp_dir.glob("kaggle_*"):
                if p.is_dir():
                    try:
                        shutil.rmtree(str(p), ignore_errors=True)
                        cleaned += 1
                    except Exception:
                        pass
        except Exception:
            pass
        if cleaned > 0:
            logger.info("Purged %d stale Kaggle temporary auth directories on startup", cleaned)
        return cleaned

    def _load_and_sync_accounts(self) -> None:
        """Load account metadata from config and synchronize with SQLite store."""
        accounts_file = self.credentials_dir / "kaggle_accounts.json"
        if not accounts_file.exists():
            logger.warning("No kaggle_accounts.json found at %s", accounts_file)
            return

        try:
            with open(accounts_file) as f:
                data = json.load(f)

            raw_accounts = data.get("accounts", [])
            self.state_store.sync_accounts(raw_accounts)

            # Sync rotation policy from SQLite
            rot_data = self.state_store.get_rotation_policy()
            if rot_data:
                self.rotation_policy.current_rotations = rot_data.get("current_rotations", 0)
                self.rotation_policy.max_rotations_per_task = rot_data.get("max_rotations", 3)
                self.rotation_policy.rotation_permitted = bool(rot_data.get("rotation_permitted", 1))
        except Exception as e:
            logger.error("Failed to load accounts: %s", str(e))

    @property
    def accounts(self) -> Dict[str, KaggleAccount]:
        """Expose current account state directly from SQLite."""
        db_accounts = self.state_store.get_accounts()
        result = {}
        for acc in db_accounts:
            result[acc["account_id"]] = KaggleAccount(
                account_id=acc["account_id"],
                username=acc["username"],
                status=AccountStatus(acc["status"])
                if acc["status"] in [s.value for s in AccountStatus]
                else AccountStatus.AVAILABLE,
                last_used=acc["last_used"],
                active_kernels=acc["active_kernels"],
                max_concurrent=acc["max_concurrent"],
                total_submissions=acc["total_submissions"],
            )
        return result

    def select_account(self, account_id: Optional[str] = None) -> KaggleAccount:
        """Select an account with least-recently-used prioritization and persist state."""
        current_accounts = self.accounts
        if not current_accounts:
            raise RuntimeError("No Kaggle accounts configured. Run scripts/setup_credentials.sh")

        if account_id:
            if account_id not in current_accounts:
                raise ValueError(f"Unknown account: {account_id}")
            account = current_accounts[account_id]
            if account.status not in (AccountStatus.AVAILABLE, AccountStatus.IN_USE):
                raise RuntimeError(f"Account {account_id} is {account.status.value}")
        else:
            available = [
                a
                for a in current_accounts.values()
                if a.status in (AccountStatus.AVAILABLE, AccountStatus.IN_USE) and a.active_kernels < a.max_concurrent
            ]
            if not available:
                raise RuntimeError("No available Kaggle accounts with capacity")
            account = min(available, key=lambda a: a.last_used)

        # Update SQLite store
        self.state_store.update_account_usage(
            account_id=account.account_id, delta_kernels=1, delta_submissions=1, status=AccountStatus.IN_USE.value
        )
        logger.info("Selected Kaggle account: %s", account.account_id)
        return self.accounts[account.account_id]

    def rotate_account(self, failed_account_id: str) -> Optional[KaggleAccount]:
        """Policy-aware account rotation with durable rotation tracking."""
        if not self.rotation_policy.can_rotate():
            logger.warning(
                "Account rotation BLOCKED by policy. Rotations: %d/%d",
                self.rotation_policy.current_rotations,
                self.rotation_policy.max_rotations_per_task,
            )
            return None

        if self.rotation_policy.needs_human_approval():
            logger.warning(
                "Account rotation requires human approval after %d rotations",
                self.rotation_policy.require_human_approval_after,
            )
            return None

        self.mark_quota_exceeded(failed_account_id)

        try:
            next_account = self.select_account()
            self.rotation_policy.current_rotations += 1
            self.state_store.update_rotation_count(self.rotation_policy.current_rotations)
            logger.info(
                "Rotated from %s to %s (rotation %d/%d)",
                failed_account_id,
                next_account.account_id,
                self.rotation_policy.current_rotations,
                self.rotation_policy.max_rotations_per_task,
            )
            return next_account
        except RuntimeError:
            logger.error("No accounts available for rotation")
            return None

    def release_account(self, account_id: str) -> None:
        """Release account back to available pool and update SQLite store."""
        self.state_store.update_account_usage(account_id=account_id, delta_kernels=-1, delta_submissions=0)
        # Check if active is 0 to mark available
        acc = self.accounts.get(account_id)
        if acc and acc.active_kernels == 0 and acc.status == AccountStatus.IN_USE:
            self.state_store.set_account_status(account_id, AccountStatus.AVAILABLE.value)
        self._cleanup_config_dir(account_id)
        logger.info("Released account: %s", account_id)

    def mark_quota_exceeded(self, account_id: str) -> None:
        self.state_store.set_account_status(account_id, AccountStatus.QUOTA_EXCEEDED.value)
        self._cleanup_config_dir(account_id)
        logger.warning("Account %s marked as quota exceeded in SQLite store", account_id)

    def setup_auth_env(self, account_id: str) -> Dict[str, str]:
        """Set up temporary credential directory for Kaggle CLI authentication."""
        cred_file = self.credentials_dir / f"{account_id}.json"
        if not cred_file.exists():
            raise FileNotFoundError(f"Credential file not found for {account_id}")

        config_dir = tempfile.mkdtemp(prefix=f"kaggle_{account_id}_")
        target_file = Path(config_dir) / "kaggle.json"

        shutil.copy2(str(cred_file), str(target_file))
        os.chmod(str(target_file), 0o600)

        self._active_config_dirs[account_id] = config_dir
        return {"KAGGLE_CONFIG_DIR": config_dir}

    @contextmanager
    def authenticated_session(self, account_id: str) -> Generator[Dict[str, str], None, None]:
        """Context manager guaranteeing temporary directory cleanup upon exit or exception."""
        auth_env = self.setup_auth_env(account_id)
        try:
            yield auth_env
        finally:
            self._cleanup_config_dir(account_id)

    def _cleanup_config_dir(self, account_id: str) -> None:
        if account_id in self._active_config_dirs:
            config_dir = self._active_config_dirs.pop(account_id)
            try:
                shutil.rmtree(config_dir, ignore_errors=True)
                logger.debug("Cleaned up config dir for %s", account_id)
            except OSError:
                pass

    def get_status(self) -> Dict[str, Any]:
        """Return status of all accounts without exposing credentials or internal paths."""
        current_accounts = self.accounts.values()
        return {
            "accounts": [
                {
                    "account_id": a.account_id,
                    "username": a.username,
                    "status": a.status.value,
                    "active_kernels": a.active_kernels,
                    "total_submissions": a.total_submissions,
                }
                for a in current_accounts
            ],
            "rotation_policy": {
                "rotation_permitted": self.rotation_policy.rotation_permitted,
                "current_rotations": self.rotation_policy.current_rotations,
                "max_rotations": self.rotation_policy.max_rotations_per_task,
            },
        }

    def cleanup_all(self) -> None:
        """Purge all active temporary config directories."""
        for account_id in list(self._active_config_dirs.keys()):
            self._cleanup_config_dir(account_id)
