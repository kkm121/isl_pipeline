#!/usr/bin/env python3
"""Auto-format hook — runs ruff format after file edits.

Used as a PostToolUse hook for edit/write tools.
Reads tool call JSON from stdin, runs formatter, outputs result.
"""
import sys
import json
import subprocess
from pathlib import Path

def format_file(file_path: str) -> dict:
    if not file_path.endswith('.py'):
        return {}
    
    try:
        result = subprocess.run(
            ['python', '-m', 'ruff', 'format', file_path],
            capture_output=True, text=True, timeout=30
        )
        return {
            'formatted': result.returncode == 0,
            'file': file_path
        }
    except Exception as e:
        return {'error': str(e)}

def main():
    try:
        payload = json.load(sys.stdin)
        tool_args = payload.get('toolArgs', {})
        target_file = tool_args.get('TargetFile', '')
        
        if target_file:
            result = format_file(target_file)
        else:
            result = {}
        
        json.dump(result, sys.stdout)
    except Exception:
        json.dump({}, sys.stdout)

if __name__ == '__main__':
    main()
