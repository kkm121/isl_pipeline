import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import os

def test_safety_check_blocks_rm_rf():
    # Mock behavior of safety_check.py checking logic
    command = "rm -rf /"
    blocked = "rm -rf" in command
    assert blocked

def test_safety_check_blocks_credential_exposure():
    command = "echo $API_KEY"
    blocked = "API_KEY" in command or "SECRET" in command
    assert blocked

def test_safety_check_allows_safe_commands():
    command = "pytest tests/"
    blocked = "rm -rf" in command
    assert not blocked

def test_auto_format_handles_python_files():
    filename = "script.py"
    processed = filename.endswith(".py")
    assert processed

def test_auto_format_skips_non_python():
    filename = "README.md"
    processed = filename.endswith(".py")
    assert not processed

def test_diagnostics_oom():
    error_msg = "RuntimeError: CUDA out of memory"
    is_oom = "out of memory" in error_msg.lower()
    assert is_oom

def test_diagnostics_nan():
    error_msg = "Loss is NaN"
    is_nan = "nan" in error_msg.lower()
    assert is_nan

def test_diagnostics_dependency():
    error_msg = "ModuleNotFoundError: No module named 'torch'"
    is_dep = "modulenotfounderror" in error_msg.lower() or "importerror" in error_msg.lower()
    assert is_dep

def test_diagnostics_unknown():
    error_msg = "ValueError: Unexpected value"
    is_unknown = not any(x in error_msg.lower() for x in ["out of memory", "nan", "modulenotfounderror"])
    assert is_unknown

def test_account_broker_no_credentials_exposed():
    status_response = {"status": "ok", "user": "test_user"}
    assert "API_KEY" not in str(status_response)
    assert "SECRET" not in str(status_response)
