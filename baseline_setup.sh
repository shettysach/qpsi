#!/usr/bin/env bash
set -euo pipefail

PSI_REPO="/home/sach/Desktop/Psi0"
CHECKPOINT="psi0/sonic-checkpoints/multi-task.psi-dream.2609092156"
HF_REPO="USC-PSI-Lab/psi-model"
CHECKPOINT_REVISION="4c6f9776fc5b18d87945254175e38bb74b9d7748"
CHECKPOINT_CACHE="$PSI_REPO/cache/checkpoints"
DATASET_ROOT="$PSI_REPO/.data"
DATASET_REVISION="e78fb93cc28912a3031a10b8656d32d7f0a2b867"

if [[ ! -f "$PSI_REPO/src/psi/models/psi0.py" ]]; then
  echo "Psi0 checkout missing at $PSI_REPO" >&2
  exit 1
fi

# If Hugging Face asks for authentication, log in through the CLI and paste your token:
# uvx hf auth login

echo "Downloading $HF_REPO/$CHECKPOINT into $CHECKPOINT_CACHE"
uvx hf download "$HF_REPO" \
  --include "$CHECKPOINT/**" \
  --revision "$CHECKPOINT_REVISION" \
  --local-dir "$CHECKPOINT_CACHE"

echo "Downloading the pinned UniFolM SONIC validation archive"
uvx hf download USC-PSI-Lab/psi-data \
  sonic/unifolm_sonic_lerobot_val.zip \
  --repo-type dataset \
  --revision "$DATASET_REVISION" \
  --local-dir "$DATASET_ROOT"
if [[ ! -f "$DATASET_ROOT/unifolm_sonic_lerobot_val/meta/info.json" ]]; then
  unzip "$DATASET_ROOT/sonic/unifolm_sonic_lerobot_val.zip" -d "$DATASET_ROOT"
fi

echo "Checkpoint and validation data setup complete."
