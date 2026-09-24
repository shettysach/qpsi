#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/baseline_config.json"

PSI_REPO="$(jq -r '.psi_repo' "$CONFIG_FILE")"
CHECKPOINT="$(jq -r '.checkpoint' "$CONFIG_FILE")"
HF_REPO="USC-PSI-Lab/psi-model"
CHECKPOINT_CACHE="$PSI_REPO/cache/checkpoints"

if [[ ! -f "$PSI_REPO/src/psi/models/psi0.py" ]]; then
  echo "Set psi_repo in $CONFIG_FILE to the official Psi0 checkout." >&2
  exit 1
fi

# If Hugging Face asks for authentication, log in through the CLI and paste your token:
# uvx hf auth login

echo "Downloading $HF_REPO/$CHECKPOINT into $CHECKPOINT_CACHE"
uvx hf download "$HF_REPO" \
  --include "$CHECKPOINT/**" \
  --local-dir "$CHECKPOINT_CACHE"

echo "Checkpoint setup complete."
