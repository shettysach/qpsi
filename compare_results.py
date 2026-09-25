#!/usr/bin/env python3
"""Print and save a table of all Ψ₀ validation runs against the released reference."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/vlm_bf16_act_fp32"
RUN_NAME = re.compile(r"vlm_(bf16|fp8)_act_(fp32|bf16|fp8)")
PAIR_KEYS = ("sample_id", "episode_index", "frame_index")
ACTION_KEYS = PAIR_KEYS + ("seed", "instruction")
SUMMARY_KEYS = (
    "checkpoint", "checkpoint_step", "checkpoint_revision", "dataset",
    "dataset_revision", "dataset_sha256", "samples", "seed", "inference_steps",
    "action_shape", "torch_version", "torch_cuda_version", "cuda_device",
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path) -> dict:
    if not path.is_dir():
        raise FileNotFoundError(f"Results directory missing: {path}")
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    if "vlm_dtype" in summary and summary.get("variant") != path.name:
        raise ValueError(f"Summary variant does not match directory name: {path}")
    return {
        "path": path,
        "summary": summary,
        "actions": read_jsonl(path / "actions.jsonl"),
        "losses": read_jsonl(path / "flow_losses.jsonl"),
    }


def discover_runs() -> list[Path]:
    root = REFERENCE.parent
    if not (REFERENCE / "summary.json").is_file():
        raise FileNotFoundError(f"Released reference missing: {REFERENCE}")
    paths = sorted(path for path in root.iterdir()
                   if path.is_dir() and RUN_NAME.fullmatch(path.name)
                   and (path / "summary.json").is_file())
    paths.remove(REFERENCE)
    return [REFERENCE, *paths]


def check_pairs(left: list[dict], right: list[dict], keys: tuple[str, ...]) -> None:
    if not left or len(left) != len(right):
        raise ValueError("Runs have empty data or different sample counts")
    for index, (a, b) in enumerate(zip(left, right)):
        if a.get("sample_id") != index or any(a.get(key) != b.get(key) for key in keys):
            raise ValueError(f"Samples differ at row {index} on {keys}")


def compare_run(reference: dict, variant: dict) -> dict:
    base, other = reference["summary"], variant["summary"]
    if (base.get("vlm_dtype", "bf16"),
        base.get("action_expert_dtype", base.get("action_head_dtype", "fp32"))) != ("bf16", "fp32"):
        raise ValueError("Reference directory does not contain the released BF16-VLM/FP32-expert run")
    for key in SUMMARY_KEYS:
        if base.get(key) != other.get(key):
            raise ValueError(f"Run settings differ at {key}: {base.get(key)!r} vs {other.get(key)!r}")
    for filename in ("flow_noise.npy", "flow_sigmas.npy"):
        if sha256(reference["path"] / filename) != sha256(variant["path"] / filename):
            raise ValueError(f"Runs used different {filename}")
    check_pairs(reference["actions"], variant["actions"], ACTION_KEYS)
    check_pairs(reference["losses"], variant["losses"], PAIR_KEYS)
    check_pairs(reference["actions"], reference["losses"], PAIR_KEYS)
    check_pairs(variant["actions"], variant["losses"], PAIR_KEYS)

    original = np.asarray([row["action"] for row in reference["actions"]], dtype=np.float64)
    current = np.asarray([row["action"] for row in variant["actions"]], dtype=np.float64)
    shape = (len(reference["actions"]), *base["action_shape"])
    if original.shape != shape or current.shape != shape:
        raise ValueError(f"Expected action arrays of shape {shape}")
    base_flow = np.asarray([row["shared78_mse"] for row in reference["losses"]], dtype=np.float64)
    other_flow = np.asarray([row["shared78_mse"] for row in variant["losses"]], dtype=np.float64)
    if not all(np.isfinite(values).all() for values in (original, current, base_flow, other_flow)):
        raise ValueError("Saved actions or losses contain nonfinite values")

    error = current - original
    flat_base = original.reshape(len(original), -1)
    flat_other = current.reshape(len(current), -1)
    denom = np.linalg.norm(flat_base, axis=1) * np.linalg.norm(flat_other, axis=1)
    cosine = np.divide(np.sum(flat_base * flat_other, axis=1), denom,
                       out=np.zeros(len(original)), where=denom > 0)
    cosine[(denom == 0) & np.all(flat_base == flat_other, axis=1)] = 1.0
    flow_delta = other_flow - base_flow
    return {
        "reference": reference["path"].name,
        "variant": variant["path"].name,
        "samples": len(original),
        "flow_loss_shared78_reference": float(base_flow.mean()),
        "flow_loss_shared78_variant": float(other_flow.mean()),
        "flow_loss_shared78_delta": float(flow_delta.mean()),
        "flow_loss_shared78_relative_change_percent": (
            float(100 * flow_delta.mean() / base_flow.mean()) if base_flow.mean() else None
        ),
        "action_mae": float(np.abs(error).mean()),
        "action_mse": float(np.square(error).mean()),
        "mean_action_cosine_similarity": float(cosine.mean()),
        "mean_latency_ms_reference": float(base["mean_latency_seconds"] * 1000),
        "mean_latency_ms_variant": float(other["mean_latency_seconds"] * 1000),
        "peak_vram_gib_reference": float(base["peak_inference_vram_bytes"] / 2**30),
        "peak_vram_gib_variant": float(other["peak_inference_vram_bytes"] / 2**30),
    }


def main() -> None:
    paths = discover_runs()
    reference = load_run(REFERENCE)
    columns = (
        ("Run", "run"),
        ("Flow MSE", "flow_mse"),
        ("Δ flow", "flow_delta"),
        ("Δ %", "flow_delta_percent"),
        ("Output MAE", "output_mae"),
        ("Output MSE", "output_mse"),
        ("Cosine", "output_cosine"),
        ("Latency ms", "latency_ms"),
        ("Peak GiB", "peak_vram_gib"),
    )
    records = []
    for path in paths:
        report = compare_run(reference, load_run(path))
        records.append({
            "run": path.name,
            "flow_mse": report["flow_loss_shared78_variant"],
            "flow_delta": report["flow_loss_shared78_delta"],
            "flow_delta_percent": report["flow_loss_shared78_relative_change_percent"],
            "output_mae": report["action_mae"],
            "output_mse": report["action_mse"],
            "output_cosine": report["mean_action_cosine_similarity"],
            "latency_ms": report["mean_latency_ms_variant"],
            "peak_vram_gib": report["peak_vram_gib_variant"],
        })

    root = REFERENCE.parent
    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for _, key in columns])
        writer.writeheader()
        writer.writerows(records)

    def display(key: str, value) -> str:
        if value is None:
            return "n/a"
        if key == "run":
            return value
        if key == "output_cosine":
            return f"{value:.6f}"
        if key in ("latency_ms", "peak_vram_gib"):
            return f"{value:.2f}"
        if key == "flow_delta_percent":
            return f"{value:+.3f}%"
        if key == "flow_delta":
            return f"{value:+.6g}"
        return f"{value:.6g}"

    headers = [title for title, _ in columns]
    cells = [[display(key, record[key]) for _, key in columns] for record in records]
    widths = [max(len(header), *(len(row[i]) for row in cells)) for i, header in enumerate(headers)]
    lines = ["| " + " | ".join(header.ljust(widths[i]) for i, header in enumerate(headers)) + " |",
             "| " + " | ".join("-" * width for width in widths) + " |"]
    lines.extend("| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(row)) + " |"
                 for row in cells)
    table = "\n".join(lines)
    markdown_path = root / "comparison.md"
    markdown_path.write_text(
        "# Ψ₀ validation comparison\n\n"
        "Reference: `vlm_bf16_act_fp32`. Flow MSE uses the shared 78 dimensions; "
        "output errors compare generated 30×80 action chunks with the released reference.\n\n"
        + table + "\n", encoding="utf-8"
    )
    print(table)
    print(f"Saved {markdown_path} and {csv_path}")


if __name__ == "__main__":
    main()
