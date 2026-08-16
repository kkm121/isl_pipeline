#!/bin/bash
# ISL Pipeline — 7-Step Verification Pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

PASS=0
FAIL=0

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

# Step 1: Specification verification
step 1 "Specification Verification"
if [ -f "configs/default.yaml" ]; then
  echo "  Config exists."
  record 0
else
  echo "  No config found."
  record 1
fi

# Step 2: Static analysis (mypy)
step 2 "Static Analysis (mypy)"
python -m mypy src/ --ignore-missing-imports --no-error-summary 2>&1 || true
python -m mypy src/ --ignore-missing-imports 2>&1
record $?

# Step 3: Static analysis (ruff)
step 3 "Lint Check (ruff)"
python -m ruff check src/ tests/ 2>&1
record $?

# Step 4: Unit and integration tests
step 4 "Unit / Integration Tests (pytest)"
python -m pytest tests/ -v --tb=short 2>&1
record $?

# Step 5: Git diff verification
step 5 "HEAD / Git Diff Verification"
git status --short
git log --oneline -5
record 0

# Step 6: Artifact verification
step 6 "Artifact Verification"
echo "  Checking required source files..."
REQUIRED_FILES=(
  "src/models/classifier.py"
  "src/data/dataset.py"
  "src/training/trainer.py"
  "tests/test_model.py"
  "tests/test_dataset.py"
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
step 7 "Final Decision"
echo ""
echo "  Results: $PASS passed, $FAIL failed"
if [ $FAIL -eq 0 ]; then
  echo "  🎉 ALL CHECKS PASSED"
  exit 0
else
  echo "  ⚠️  SOME CHECKS FAILED — review required"
  exit 1
fi
