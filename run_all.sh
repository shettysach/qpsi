#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
failed=0

run_variant() {
  local vlm="$1"
  local act="$2"
  local result="results/vlm_${vlm}_act_${act}"

  if [[ -f "$result/summary.json" ]]; then
    echo "Already complete: $result"
    return
  fi
  if [[ -f "$result/failure.json" ]]; then
    echo "Numerically failed: $result (remove failure.json to retry)"
    failed=1
    return
  fi

  echo "Evaluating VLM=$vlm action expert=$act"
  if uv run evaluate.py --vlm "$vlm" --act "$act"; then
    return
  else
    local status=$?
    if [[ "$status" -eq 3 ]]; then
      echo "Numerically failed: $result"
      failed=1
      return
    fi
    return "$status"
  fi
}

run_variant bf16 fp32
run_variant bf16 bf16
run_variant bf16 fp16
run_variant fp16 fp32
run_variant fp16 bf16
run_variant fp16 fp16
run_variant bf16 fp8
run_variant fp8 fp32
run_variant fp8 fp8

uv run compare_results.py
uv run plot_results.py
if [[ "$failed" -ne 0 ]]; then
  echo "Some precision variants produced nonfinite outputs; inspect results/*/failure.json" >&2
  exit 1
fi
