import os
import json
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, Dict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class AuthMethod(Enum):
    KAGGLE_JSON = 1
    API_TOKEN = 2
    ENV_VARS = 3

class KaggleAuthAdapter:
    def __init__(self, credentials_dir: str = 'credentials/'):
        self.credentials_dir = Path(credentials_dir)
        self.active_temp_dir: Optional[str] = None
        self.old_env_vars: Dict[str, Optional[str]] = {}

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
            with open(cred_path, 'r') as f:
                creds = json.load(f)
            
            temp_dir = tempfile.mkdtemp(prefix=f"kaggle_auth_{account_id}_")
            temp_json = Path(temp_dir) / "kaggle.json"
            with open(temp_json, 'w') as f:
                json.dump(creds, f)
            os.chmod(temp_json, 0o600)
            
            self.active_temp_dir = temp_dir
            return {"KAGGLE_CONFIG_DIR": temp_dir}
            
        elif method == AuthMethod.API_TOKEN:
            # We assume KAGGLE_API_TOKEN is a JSON string or valid base64 token logic
            # For simplicity, if they have KAGGLE_API_TOKEN, we might need to write it to kaggle.json
            token = os.environ["KAGGLE_API_TOKEN"]
            try:
                creds = json.loads(token)
                temp_dir = tempfile.mkdtemp(prefix="kaggle_auth_token_")
                temp_json = Path(temp_dir) / "kaggle.json"
                with open(temp_json, 'w') as f:
                    json.dump(creds, f)
                os.chmod(temp_json, 0o600)
                self.active_temp_dir = temp_dir
                return {"KAGGLE_CONFIG_DIR": temp_dir}
            except json.JSONDecodeError:
                # If it's not JSON, might be a direct env var setup needed, fallback
                return {}
                
        elif method == AuthMethod.ENV_VARS:
            return {
                "KAGGLE_USERNAME": os.environ.get("KAGGLE_USERNAME", ""),
                "KAGGLE_KEY": os.environ.get("KAGGLE_KEY", "")
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
            shutil.rmtree(self.active_temp_dir)
            self.active_temp_dir = None

    def validate_credentials(self, account_id: str) -> bool:
        self.setup_env(account_id)
        try:
            res = subprocess.run(["kaggle", "competitions", "list", "--page-size", "1"], capture_output=True, text=True, check=True)
            return "ref" in res.stdout
        except Exception as e:
            logger.error(f"Credential validation failed: {e}")
            return False
        finally:
            self.teardown_env()
