#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

run_variant() {
  local vlm="$1"
  local act="$2"
  local result="results/vlm_${vlm}_act_${act}"

  if [[ -f "$result/summary.json" ]]; then
    echo "Already complete: $result"
    return
  fi

  echo "Evaluating VLM=$vlm action expert=$act"
  uv run evaluate.py --vlm "$vlm" --act "$act"
}

run_variant bf16 fp32
run_variant bf16 bf16
run_variant bf16 fp8
run_variant fp8 fp32
run_variant fp8 fp8

uv run compare_results.py
uv run plot_results.py
