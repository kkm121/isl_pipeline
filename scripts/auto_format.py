#!/usr/bin/env python3
"""Auto-format hook — runs ruff format after file edits.

Used as a PostToolUse hook for edit/write tools.
Validates and canonicalizes the target path, rejecting path traversals and symlink escapes.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).parent.parent.resolve()


def validate_canonical_path(file_path: str) -> Path:
    """Validate and canonicalize path, strictly prohibiting traversal or symlink escapes outside PROJECT_ROOT."""
    target = Path(file_path)
    if not target.is_absolute():
        target = (PROJECT_ROOT / target).resolve()
    else:
        target = target.resolve()

    # Verify no symlink or traversal escapes PROJECT_ROOT
    try:
        target.relative_to(PROJECT_ROOT)
    except ValueError:
        raise ValueError(f"Path traversal or outside boundary detected: {file_path}")

    # If file exists, ensure resolved real path is also inside PROJECT_ROOT
    if target.exists() and not target.resolve().is_relative_to(PROJECT_ROOT):
        raise ValueError(f"Symlink escape detected: {file_path}")

    return target


def format_file(file_path: str) -> dict:
    if not file_path.endswith(".py"):
        return {"status": "skipped", "reason": "not_python", "file": file_path}

    try:
        safe_path = validate_canonical_path(file_path)
    except ValueError as e:
        return {"status": "rejected", "error": str(e), "file": file_path}

    try:
        result = subprocess.run(
            ["ruff", "format", str(safe_path)],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        return {
            "status": "formatted" if result.returncode == 0 else "error",
            "file": str(safe_path.relative_to(PROJECT_ROOT)),
            "exit_code": result.returncode,
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "file": file_path}


def main():
    try:
        raw_input = sys.stdin.read().strip()
        if not raw_input:
            json.dump({"status": "no_input"}, sys.stdout)
            return

        payload = json.loads(raw_input)
        tool_args = payload.get("toolArgs", {})
        target_file = tool_args.get("TargetFile", "")

        if target_file:
            result = format_file(target_file)
        else:
            result = {"status": "no_target_file"}

        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({"status": "hook_error", "error": str(e)}, sys.stdout)


if __name__ == "__main__":
    main()
