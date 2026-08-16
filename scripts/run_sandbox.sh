#!/bin/bash
# Run sealed sandbox — NO INTERNET
# Usage: ./scripts/run_sandbox.sh [pytest args]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[SANDBOX] Building sealed sandbox container..."
docker build -t isl-sandbox -f "$PROJECT_ROOT/docker/Dockerfile.sandbox" "$PROJECT_ROOT"

echo "[SANDBOX] Running with --network=none (no internet)..."
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  -v "$PROJECT_ROOT/src:/workspace/src:ro" \
  -v "$PROJECT_ROOT/tests:/workspace/tests:ro" \
  -v "$PROJECT_ROOT/configs:/workspace/configs:ro" \
  isl-sandbox \
  pytest tests/ -v "${@}"

echo "[SANDBOX] Complete."
