#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$SCRIPT_DIR/.venv"
PYTHON="$ENV_DIR/bin/python"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
  exit 1
fi

# uv sync owns this project's single .venv and removes packages outside the
# project dependency set. It does not read the Psi0 checkout's broken uv.lock.
uv sync --project "$SCRIPT_DIR" --python 3.11

"$PYTHON" - <<'PY'
import sys
import torch

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__} ({torch.__file__})")
print(f"PyTorch CUDA: {torch.version.cuda}")
if not torch.__version__.endswith("+cu128") or torch.version.cuda != "12.8":
    raise SystemExit("Expected the PyTorch CUDA 12.8 wheel")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Check nvidia-smi and the NVIDIA driver.")
name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
arch = f"sm_{major}{minor}"
supported = torch.cuda.get_arch_list()
print(f"GPU: {name} ({arch})")
print(f"PyTorch architectures: {supported}")
if arch not in supported:
    raise SystemExit(f"PyTorch wheel does not include {arch}")
try:
    x = torch.zeros(1, device="cuda")
    print(f"CUDA allocation and kernel: {x.item()}")
except RuntimeError as exc:
    raise SystemExit(f"CUDA operation failed: {exc}") from exc
PY

echo "Baseline environment ready: $PYTHON"
