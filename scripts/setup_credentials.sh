#!/bin/bash
# Setup credential store for ISL Pipeline
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CREDENTIALS_DIR="$PROJECT_ROOT/credentials"

mkdir -p "$CREDENTIALS_DIR"
chmod 700 "$CREDENTIALS_DIR"

echo "ISL Pipeline Credential Setup"
echo "============================="
echo ""
echo "This script initializes the credential store."
echo "You will need your Kaggle API credentials."
echo ""

# Create account mapping
cat > "$CREDENTIALS_DIR/kaggle_accounts.json" << 'EOF'
{
  "accounts": [
    {"account_id": "kaggle_1", "username": "REPLACE_WITH_USERNAME_1", "credential_file": "kaggle_1.json"},
    {"account_id": "kaggle_2", "username": "REPLACE_WITH_USERNAME_2", "credential_file": "kaggle_2.json"},
    {"account_id": "kaggle_3", "username": "REPLACE_WITH_USERNAME_3", "credential_file": "kaggle_3.json"}
  ]
}
EOF

# Create placeholder credential files
for i in 1 2 3; do
  if [ ! -f "$CREDENTIALS_DIR/kaggle_$i.json" ]; then
    cat > "$CREDENTIALS_DIR/kaggle_$i.json" << EOF
{"username": "REPLACE", "key": "REPLACE"}
EOF
    chmod 600 "$CREDENTIALS_DIR/kaggle_$i.json"
    echo "Created: credentials/kaggle_$i.json (needs real credentials)"
  fi
done

echo ""
echo "✅ Credential store initialized at: $CREDENTIALS_DIR"
echo "⚠️  Edit the kaggle_*.json files with your actual API credentials."
echo "⚠️  These files are gitignored and will NOT be committed."
