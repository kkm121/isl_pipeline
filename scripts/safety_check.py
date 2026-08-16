#!/usr/bin/env python3
"""Safety check hook — validates commands before execution.

Used as a PreToolUse hook to block dangerous operations.
Reads tool call JSON from stdin, outputs decision JSON to stdout.
"""
import sys
import json
import re

BLOCKED_PATTERNS = [
    r'rm\s+-rf\s+/',          # recursive delete from root
    r'rm\s+-rf\s+~',          # recursive delete home
    r'rm\s+-rf\s+\*',         # recursive delete all
    r'curl.*\|.*sh',          # pipe curl to shell
    r'wget.*\|.*sh',          # pipe wget to shell
    r'chmod\s+777',           # world-writable
    r'eval\s*\(',             # eval injection
    r'KAGGLE_KEY',            # credential exposure
    r'KAGGLE_USERNAME',       # credential exposure
    r'API_KEY',               # credential exposure
    r'SECRET',                # credential exposure
    r'PASSWORD',              # credential exposure
    r'TOKEN=',                # credential exposure
]

def check_command(command: str) -> dict:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command, re.IGNORECASE):
            return {
                'decision': 'deny',
                'reason': f'Blocked: command matches dangerous pattern "{pattern}"'
            }
    return {'decision': 'allow'}

def main():
    try:
        payload = json.load(sys.stdin)
        tool_name = payload.get('toolName', '')
        tool_args = payload.get('toolArgs', {})
        
        if tool_name == 'run_command':
            command = tool_args.get('CommandLine', '')
            result = check_command(command)
        else:
            result = {'decision': 'allow'}
        
        json.dump(result, sys.stdout)
    except Exception as e:
        json.dump({'decision': 'allow', 'warning': str(e)}, sys.stdout)

if __name__ == '__main__':
    main()
