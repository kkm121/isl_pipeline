import json
import os
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from enum import Enum
import logging
import time

logger = logging.getLogger(__name__)

class AccountStatus(Enum):
    AVAILABLE = "AVAILABLE"
    IN_USE = "IN_USE"
    QUOTA_EXCEEDED = "QUOTA_EXCEEDED"
    ERROR = "ERROR"

@dataclass
class KaggleAccount:
    account_id: str
    username: str
    status: AccountStatus = AccountStatus.AVAILABLE
    last_used: float = 0.0
    active_kernels: int = 0
    max_concurrent: int = 2

class AccountBroker:
    def __init__(self, credentials_dir: str = 'credentials/'):
        self.credentials_dir = Path(credentials_dir)
        self.accounts: Dict[str, KaggleAccount] = {}
        self._load_accounts()

    def _load_accounts(self):
        config_path = self.credentials_dir / 'kaggle_accounts.json'
        if not config_path.exists():
            return
        with open(config_path, 'r') as f:
            data = json.load(f)
        for acc in data.get('accounts', []):
            account_id = acc.get('account_id')
            self.accounts[account_id] = KaggleAccount(
                account_id=account_id,
                username=acc.get('username')
            )

    def select_account(self, account_id: Optional[str] = None) -> KaggleAccount:
        if account_id:
            if account_id not in self.accounts:
                raise RuntimeError(f"Account {account_id} not found.")
            acc = self.accounts[account_id]
            if acc.status == AccountStatus.AVAILABLE and acc.active_kernels < acc.max_concurrent:
                acc.active_kernels += 1
                acc.last_used = time.time()
                if acc.active_kernels >= acc.max_concurrent:
                    acc.status = AccountStatus.IN_USE
                self._set_kaggle_env(account_id)
                return acc
            raise RuntimeError(f"Account {account_id} is not available.")

        available = [
            acc for acc in self.accounts.values()
            if acc.status == AccountStatus.AVAILABLE and acc.active_kernels < acc.max_concurrent
        ]
        if not available:
            raise RuntimeError("No accounts available.")
        
        available.sort(key=lambda a: a.last_used)
        selected = available[0]
        selected.active_kernels += 1
        selected.last_used = time.time()
        if selected.active_kernels >= selected.max_concurrent:
            selected.status = AccountStatus.IN_USE
        
        self._set_kaggle_env(selected.account_id)
        return selected

    def release_account(self, account_id: str):
        if account_id in self.accounts:
            acc = self.accounts[account_id]
            if acc.active_kernels > 0:
                acc.active_kernels -= 1
            if acc.status == AccountStatus.IN_USE and acc.active_kernels < acc.max_concurrent:
                acc.status = AccountStatus.AVAILABLE

    def mark_quota_exceeded(self, account_id: str):
        if account_id in self.accounts:
            self.accounts[account_id].status = AccountStatus.QUOTA_EXCEEDED

    def get_credentials_path(self, account_id: str) -> str:
        return str(self.credentials_dir / f"{account_id}.json")

    def get_status(self) -> Dict:
        return {
            acc_id: {
                "id": acc.account_id,
                "username": acc.username,
                "status": acc.status.value,
                "active_kernels": acc.active_kernels
            }
            for acc_id, acc in self.accounts.items()
        }

    def _set_kaggle_env(self, account_id: str):
        os.environ['KAGGLE_CONFIG_DIR'] = str(self.credentials_dir.absolute())
