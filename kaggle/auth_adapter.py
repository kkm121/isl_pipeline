import atexit
from enum import Enum
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    KAGGLE_JSON = 1
    API_TOKEN = 2
    ENV_VARS = 3


class KaggleAuthAdapter:
    def __init__(self, credentials_dir: str = "credentials/"):
        self.credentials_dir = Path(credentials_dir)
        self.active_temp_dir: Optional[str] = None
        self.old_env_vars: Dict[str, Optional[str]] = {}
        atexit.register(self.teardown_env)

    def get_auth_method(self, account_id: str) -> AuthMethod:
        if (self.credentials_dir / f"{account_id}.json").exists():
            return AuthMethod.KAGGLE_JSON
        if "KAGGLE_API_TOKEN" in os.environ:
            return AuthMethod.API_TOKEN
        return AuthMethod.ENV_VARS

    def authenticate(self, account_id: str) -> Dict[str, str]:
        method = self.get_auth_method(account_id)

        if method == AuthMethod.KAGGLE_JSON:
            cred_path = self.credentials_dir / f"{account_id}.json"
            with open(cred_path, "r") as f:
                creds = json.load(f)

            temp_dir = tempfile.mkdtemp(prefix=f"kaggle_auth_{account_id}_")
            temp_json = Path(temp_dir) / "kaggle.json"
            with open(temp_json, "w") as f:
                json.dump(creds, f)
            os.chmod(temp_json, 0o600)

            self.active_temp_dir = temp_dir
            return {"KAGGLE_CONFIG_DIR": temp_dir}

        elif method == AuthMethod.API_TOKEN:
            token = os.environ["KAGGLE_API_TOKEN"]
            try:
                creds = json.loads(token)
                temp_dir = tempfile.mkdtemp(prefix="kaggle_auth_token_")
                temp_json = Path(temp_dir) / "kaggle.json"
                with open(temp_json, "w") as f:
                    json.dump(creds, f)
                os.chmod(temp_json, 0o600)
                self.active_temp_dir = temp_dir
                return {"KAGGLE_CONFIG_DIR": temp_dir}
            except json.JSONDecodeError:
                return {}

        elif method == AuthMethod.ENV_VARS:
            return {
                "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
                "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", ""),
            }

        return {}

    def setup_env(self, account_id: str) -> None:
        env_vars = self.authenticate(account_id)
        for k, v in env_vars.items():
            self.old_env_vars[k] = os.environ.get(k)
            os.environ[k] = v

    def teardown_env(self) -> None:
        for k, v in self.old_env_vars.items():
            if v is None:
                if k in os.environ:
                    del os.environ[k]
            else:
                os.environ[k] = v
        self.old_env_vars.clear()

        if self.active_temp_dir and os.path.exists(self.active_temp_dir):
            try:
                shutil.rmtree(self.active_temp_dir, ignore_errors=True)
            except OSError:
                pass
            self.active_temp_dir = None

    def validate_credentials(self, account_id: str) -> bool:
        self.setup_env(account_id)
        try:
            res = subprocess.run(
                ["kaggle", "competitions", "list", "--page-size", "1"],
                capture_output=True,
                text=True,
                check=True,
            )
            return "ref" in res.stdout
        except Exception as e:
            logger.error("Credential validation failed: %s", str(e))
            return False
        finally:
            self.teardown_env()
