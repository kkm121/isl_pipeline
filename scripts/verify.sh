#!/bin/bash
# ISL Pipeline — 7-Step Containerized Verification Pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PASS=0
FAIL=0
REQUIRE_DIFF=0

for arg in "$@"; do
  case $arg in
    --require-diff)
      REQUIRE_DIFF=1
      shift
      ;;
  esac
done

step() {
  echo ""
  echo "====================================="
  echo "STEP $1: $2"
  echo "====================================="
}

record() {
  if [ $1 -eq 0 ]; then
    echo "  ✅ PASS"
    PASS=$((PASS + 1))
  else
    echo "  ❌ FAIL"
    FAIL=$((FAIL + 1))
  fi
}

SANDBOX_IMAGE="isl-sandbox"

# Build sealed sandbox container if not present
echo "[VERIFY] Ensuring sealed sandbox image is ready..."
if ! docker image inspect "$SANDBOX_IMAGE" >/dev/null 2>&1; then
  echo "[VERIFY] Building sealed sandbox container: $SANDBOX_IMAGE"
  docker build -t "$SANDBOX_IMAGE" -f "$PROJECT_ROOT/docker/Dockerfile.sandbox" "$PROJECT_ROOT"
fi

# Step 1: Specification verification
step 1 "Specification Verification"
if [ -f "configs/default.yaml" ]; then
  echo "  Config exists: configs/default.yaml"
  record 0
else
  echo "  No config found."
  record 1
fi

# Step 2: Static analysis (mypy) inside sealed container
step 2 "Static Analysis (mypy via Docker Sealed Sandbox)"
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  -v "$PROJECT_ROOT/src:/workspace/src:ro" \
  -v "$PROJECT_ROOT/tests:/workspace/tests:ro" \
  -v "$PROJECT_ROOT/configs:/workspace/configs:ro" \
  -v "$PROJECT_ROOT/kaggle:/workspace/kaggle:ro" \
  -v "$PROJECT_ROOT/mcp_servers:/workspace/mcp_servers:ro" \
  -v "$PROJECT_ROOT/scripts:/workspace/scripts:ro" \
  "$SANDBOX_IMAGE" \
  mypy src/ --ignore-missing-imports --cache-dir /tmp/.mypy_cache
record $?

# Step 3: Lint check (ruff) inside sealed container
step 3 "Lint Check (ruff via Docker Sealed Sandbox)"
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  -v "$PROJECT_ROOT/src:/workspace/src:ro" \
  -v "$PROJECT_ROOT/tests:/workspace/tests:ro" \
  -v "$PROJECT_ROOT/configs:/workspace/configs:ro" \
  -v "$PROJECT_ROOT/kaggle:/workspace/kaggle:ro" \
  -v "$PROJECT_ROOT/mcp_servers:/workspace/mcp_servers:ro" \
  -v "$PROJECT_ROOT/scripts:/workspace/scripts:ro" \
  "$SANDBOX_IMAGE" \
  sh -c "ruff check src/ tests/ --cache-dir /tmp/.ruff_cache && ruff format --check src/ tests/"
record $?

# Step 4: Unit / Integration tests inside sealed container
step 4 "Unit & Integration Tests (pytest via Docker Sealed Sandbox)"
docker run --rm \
  --network=none \
  --read-only \
  --tmpfs /tmp \
  --security-opt no-new-privileges \
  -v "$PROJECT_ROOT/src:/workspace/src:ro" \
  -v "$PROJECT_ROOT/tests:/workspace/tests:ro" \
  -v "$PROJECT_ROOT/configs:/workspace/configs:ro" \
  -v "$PROJECT_ROOT/kaggle:/workspace/kaggle:ro" \
  -v "$PROJECT_ROOT/mcp_servers:/workspace/mcp_servers:ro" \
  -v "$PROJECT_ROOT/scripts:/workspace/scripts:ro" \
  "$SANDBOX_IMAGE" \
  pytest tests/ -v --tb=short -o cache_dir=/tmp/.pytest_cache
record $?

# Step 5: Real Mandatory Git Tree Diff Verification
step 5 "Real Git Tree Diff Verification (Mandatory Agent Baseline)"
BASELINE_FILE=".state/tree_baseline.sha"

if [ ! -f "$BASELINE_FILE" ]; then
  echo "  ❌ REJECTED: Mandatory turn baseline missing at $BASELINE_FILE"
  echo "  The state machine must capture and persist a baseline before implementation/verification."
  record 1
else
  BASE_SHA=$(cat "$BASELINE_FILE" | tr -d '[:space:]')
  echo "  Comparing working tree against mandatory turn baseline: $BASE_SHA"
  DIFF_COUNT=$(git diff --name-only "$BASE_SHA" 2>/dev/null | wc -l || echo "0")
  STATUS_COUNT=$(git status --porcelain | wc -l)
  TOTAL_CHANGES=$((DIFF_COUNT + STATUS_COUNT))
  echo "  Files changed since turn baseline: $DIFF_COUNT (dirty working tree: $STATUS_COUNT)"
  if [ "$TOTAL_CHANGES" -eq 0 ]; then
    echo "  ❌ REJECTED: Expected agent modifications produced ZERO diff against baseline ($BASE_SHA)"
    record 1
  else
    echo "  ✅ Real tree diff verified ($TOTAL_CHANGES modifications detected)"
    record 0
  fi
fi

# Step 6: Artifact verification
step 6 "Artifact Verification"
echo "  Checking required architecture and source files..."
REQUIRED_FILES=(
  "src/models/classifier.py"
  "src/data/dataset.py"
  "src/training/trainer.py"
  "src/orchestrator/state_machine.py"
  "src/orchestrator/resource_budgets.py"
  "src/orchestrator/experiment_registry.py"
  "kaggle/account_broker.py"
  "kaggle/kernel_manager.py"
  "kaggle/diagnostics.py"
  "mcp_servers/linter_test_mcp.py"
  "mcp_servers/filesystem_mcp.py"
  "mcp_servers/github_mcp.py"
  "mcp_servers/kaggle_manager_mcp.py"
  "docker/Dockerfile.sandbox"
  "docker/Dockerfile.integration"
)
ALL_EXIST=0
for f in "${REQUIRED_FILES[@]}"; do
  if [ ! -f "$f" ]; then
    echo "  MISSING: $f"
    ALL_EXIST=1
  fi
done
record $ALL_EXIST

# Step 7: Summary
step 7 "Final Verification Decision"
echo ""
echo "  Results: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
  echo "  🎉 ALL CHECKS PASSED (Sealed Container + Architecture Gates Verified)"
  exit 0
else
  echo "  ⚠️ SOME CHECKS FAILED — review required"
  exit 1
fi
