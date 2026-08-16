"""Kaggle Account Broker — Policy-Aware Credential Management.

Critical security component. Raw credentials NEVER appear in:
- Agent context or prompts
- Docker containers (injected via KAGGLE_CONFIG_DIR at runtime)
- Git repository
- Logs or generated artifacts
- MCP tool responses
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

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
    """Policy gate for account rotation (correction #7)."""
    rotation_permitted: bool = True
    max_rotations_per_task: int = 3
    current_rotations: int = 0
    require_human_approval_after: int = 2  # after N rotations, ask human
    
    def can_rotate(self) -> bool:
        if not self.rotation_permitted:
            return False
        if self.current_rotations >= self.max_rotations_per_task:
            return False
        return True
    
    def needs_human_approval(self) -> bool:
        return self.current_rotations >= self.require_human_approval_after


class AccountBroker:
    """Manages Kaggle account credentials with policy-aware rotation.
    
    Credentials are stored in the credentials/ directory (gitignored).
    The broker NEVER exposes raw API keys in any return value, log, or error message.
    """
    
    def __init__(self, credentials_dir: str = "credentials/", rotation_policy: Optional[RotationPolicy] = None):
        self.credentials_dir = Path(credentials_dir)
        self.rotation_policy = rotation_policy or RotationPolicy()
        self.accounts: Dict[str, KaggleAccount] = {}
        self._active_config_dirs: Dict[str, str] = {}  # account_id -> temp dir path
        self._load_accounts()
    
    def _load_accounts(self) -> None:
        """Load account metadata (NOT credentials) from config."""
        accounts_file = self.credentials_dir / "kaggle_accounts.json"
        if not accounts_file.exists():
            logger.warning("No kaggle_accounts.json found at %s", accounts_file)
            return
        
        with open(accounts_file) as f:
            data = json.load(f)
        
        for entry in data.get("accounts", []):
            account_id = entry["account_id"]
            self.accounts[account_id] = KaggleAccount(
                account_id=account_id,
                username=entry["username"],
            )
            logger.info("Loaded account: %s (username: %s)", account_id, entry["username"])
    
    def select_account(self, account_id: Optional[str] = None) -> KaggleAccount:
        """Select an account. Auto-selects least-recently-used if no ID given."""
        if not self.accounts:
            raise RuntimeError("No Kaggle accounts configured. Run scripts/setup_credentials.sh")
        
        if account_id:
            if account_id not in self.accounts:
                raise ValueError(f"Unknown account: {account_id}")
            account = self.accounts[account_id]
            if account.status not in (AccountStatus.AVAILABLE, AccountStatus.IN_USE):
                raise RuntimeError(f"Account {account_id} is {account.status.value}")
        else:
            available = [
                a for a in self.accounts.values()
                if a.status == AccountStatus.AVAILABLE and a.active_kernels < a.max_concurrent
            ]
            if not available:
                raise RuntimeError("No available Kaggle accounts")
            account = min(available, key=lambda a: a.last_used)
        
        account.status = AccountStatus.IN_USE
        account.last_used = time.time()
        account.active_kernels += 1
        account.total_submissions += 1
        
        logger.info("Selected account: %s", account.account_id)
        return account
    
    def rotate_account(self, failed_account_id: str) -> Optional[KaggleAccount]:
        """Policy-aware account rotation (correction #7).
        
        Returns None if rotation is blocked by policy → caller should trigger HUMAN_GATE.
        """
        if not self.rotation_policy.can_rotate():
            logger.warning("Account rotation BLOCKED by policy. Rotations: %d/%d",
                          self.rotation_policy.current_rotations,
                          self.rotation_policy.max_rotations_per_task)
            return None
        
        if self.rotation_policy.needs_human_approval():
            logger.warning("Account rotation requires human approval after %d rotations",
                          self.rotation_policy.require_human_approval_after)
            return None
        
        # Mark failed account
        self.mark_quota_exceeded(failed_account_id)
        
        # Try to select next account
        try:
            next_account = self.select_account()
            self.rotation_policy.current_rotations += 1
            logger.info("Rotated from %s to %s (rotation %d/%d)",
                       failed_account_id, next_account.account_id,
                       self.rotation_policy.current_rotations,
                       self.rotation_policy.max_rotations_per_task)
            return next_account
        except RuntimeError:
            logger.error("No accounts available for rotation")
            return None
    
    def release_account(self, account_id: str) -> None:
        """Release account back to available pool."""
        if account_id in self.accounts:
            account = self.accounts[account_id]
            account.active_kernels = max(0, account.active_kernels - 1)
            if account.active_kernels == 0:
                account.status = AccountStatus.AVAILABLE
            self._cleanup_config_dir(account_id)
            logger.info("Released account: %s", account_id)
    
    def mark_quota_exceeded(self, account_id: str) -> None:
        if account_id in self.accounts:
            self.accounts[account_id].status = AccountStatus.QUOTA_EXCEEDED
            self.accounts[account_id].active_kernels = 0
            self._cleanup_config_dir(account_id)
            logger.warning("Account %s marked as quota exceeded", account_id)
    
    def setup_auth_env(self, account_id: str) -> Dict[str, str]:
        """Set up authentication environment for a Kaggle CLI call.
        
        Uses KAGGLE_CONFIG_DIR approach (modern method).
        Creates a temporary directory with the kaggle.json for this account.
        Returns environment variables to set — NEVER returns raw credentials.
        """
        cred_file = self.credentials_dir / f"{account_id}.json"
        if not cred_file.exists():
            raise FileNotFoundError(f"Credential file not found for {account_id}")
        
        # Create temp config directory
        config_dir = tempfile.mkdtemp(prefix=f"kaggle_{account_id}_")
        target_file = Path(config_dir) / "kaggle.json"
        
        # Copy credentials to temp dir (never reads content into memory)
        shutil.copy2(str(cred_file), str(target_file))
        os.chmod(str(target_file), 0o600)
        
        self._active_config_dirs[account_id] = config_dir
        
        return {"KAGGLE_CONFIG_DIR": config_dir}
    
    def _cleanup_config_dir(self, account_id: str) -> None:
        """Remove temporary credential directory."""
        if account_id in self._active_config_dirs:
            config_dir = self._active_config_dirs.pop(account_id)
            try:
                shutil.rmtree(config_dir)
                logger.debug("Cleaned up config dir for %s", account_id)
            except OSError:
                pass
    
    def get_status(self) -> Dict:
        """Return status of all accounts. NEVER includes API keys."""
        return {
            "accounts": [
                {
                    "account_id": a.account_id,
                    "username": a.username,
                    "status": a.status.value,
                    "active_kernels": a.active_kernels,
                    "total_submissions": a.total_submissions,
                }
                for a in self.accounts.values()
            ],
            "rotation_policy": {
                "rotation_permitted": self.rotation_policy.rotation_permitted,
                "current_rotations": self.rotation_policy.current_rotations,
                "max_rotations": self.rotation_policy.max_rotations_per_task,
            }
        }
    
    def cleanup_all(self) -> None:
        """Clean up all temporary credential directories."""
        for account_id in list(self._active_config_dirs.keys()):
            self._cleanup_config_dir(account_id)
