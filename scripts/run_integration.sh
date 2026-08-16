#!/bin/bash
# Run controlled integration runner — allowlisted network
# Usage: ./scripts/run_integration.sh <command>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CREDENTIALS_DIR="$PROJECT_ROOT/credentials"

if [ ! -d "$CREDENTIALS_DIR" ]; then
  echo "[ERROR] Credentials directory not found: $CREDENTIALS_DIR"
  exit 1
fi

echo "[INTEGRATION] Building integration runner..."
docker build -t isl-integration -f "$PROJECT_ROOT/docker/Dockerfile.integration" "$PROJECT_ROOT"

echo "[INTEGRATION] Running with allowlisted network..."
docker run --rm \
  -v "$PROJECT_ROOT/src:/workspace/src" \
  -v "$PROJECT_ROOT/kaggle:/workspace/kaggle" \
  -v "$PROJECT_ROOT/scripts:/workspace/scripts" \
  -v "$CREDENTIALS_DIR:/workspace/credentials:ro" \
  isl-integration \
  -c "${@}"

echo "[INTEGRATION] Complete."
