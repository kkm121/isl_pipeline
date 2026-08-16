#!/bin/bash
# Cleanup ephemeral ISL Pipeline containers
set -euo pipefail

echo "[CLEANUP] Removing stopped ISL Pipeline containers..."
docker container prune -f --filter label=purpose=sealed-sandbox
docker container prune -f --filter label=purpose=controlled-integration

echo "[CLEANUP] Removing dangling images..."
docker image prune -f --filter label=purpose=sealed-sandbox
docker image prune -f --filter label=purpose=controlled-integration

echo "[CLEANUP] Done."
