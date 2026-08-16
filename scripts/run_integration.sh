#!/bin/bash
# Run integration runner — standard bridge network for remote tool operations
# Usage: ./scripts/run_integration.sh <command>
# Note: Raw credentials are NEVER mounted into execution containers.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "[INTEGRATION] Building integration runner..."
docker build -t isl-integration -f "$PROJECT_ROOT/docker/Dockerfile.integration" "$PROJECT_ROOT"

echo "[INTEGRATION] Running in isolated container (no credentials mounted)..."
docker run --rm \
  --security-opt no-new-privileges \
  -v "$PROJECT_ROOT/src:/workspace/src:ro" \
  -v "$PROJECT_ROOT/kaggle:/workspace/kaggle:ro" \
  -v "$PROJECT_ROOT/scripts:/workspace/scripts:ro" \
  isl-integration \
  -c "${@}"

echo "[INTEGRATION] Complete."
