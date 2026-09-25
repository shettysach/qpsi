#!/usr/bin/env python3
"""Compare one saved Ψ₀ validation run with the released reference."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "results/vlm_bf16_act_fp32"
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


def check_pairs(left: list[dict], right: list[dict], keys: tuple[str, ...]) -> None:
    if not left or len(left) != len(right):
        raise ValueError("Runs have empty data or different sample counts")
    for index, (a, b) in enumerate(zip(left, right)):
        if a.get("sample_id") != index or any(a.get(key) != b.get(key) for key in keys):
            raise ValueError(f"Samples differ at row {index} on {keys}")


def compare_run(reference: dict, variant: dict) -> tuple[dict, list[dict]]:
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
    episode_delta = {}
    for episode in sorted({int(row["episode_index"]) for row in reference["losses"]}):
        positions = [i for i, row in enumerate(reference["losses"])
                     if int(row["episode_index"]) == episode]
        episode_delta[str(episode)] = float(flow_delta[positions].mean())

    report = {
        "reference": reference["path"].name,
        "variant": variant["path"].name,
        "samples": len(original),
        "flow_loss_shared78_reference": float(base_flow.mean()),
        "flow_loss_shared78_variant": float(other_flow.mean()),
        "flow_loss_shared78_delta": float(flow_delta.mean()),
        "flow_loss_shared78_relative_change_percent": (
            float(100 * flow_delta.mean() / base_flow.mean()) if base_flow.mean() else None
        ),
        "flow_loss_delta_by_episode": episode_delta,
        "action_mae": float(np.abs(error).mean()),
        "action_mse": float(np.square(error).mean()),
        "mean_action_cosine_similarity": float(cosine.mean()),
        "action_mae_body64": float(np.abs(error[:, :, :64]).mean()),
        "action_mae_hands14": float(np.abs(error[:, :, 64:78]).mean()),
        "action_mae_neck2": float(np.abs(error[:, :, 78:]).mean()),
        "per_action_dimension_mae": np.abs(error).mean(axis=(0, 1)).tolist(),
        "mean_latency_ms_reference": float(base["mean_latency_seconds"] * 1000),
        "mean_latency_ms_variant": float(other["mean_latency_seconds"] * 1000),
        "peak_vram_gib_reference": float(base["peak_inference_vram_bytes"] / 2**30),
        "peak_vram_gib_variant": float(other["peak_inference_vram_bytes"] / 2**30),
    }
    rows = [
        {
            "sample_id": i,
            "episode_index": row["episode_index"],
            "frame_index": row["frame_index"],
            "action_mae": float(np.abs(error[i]).mean()),
            "action_cosine": float(cosine[i]),
            "flow_loss_reference": float(base_flow[i]),
            "flow_loss_variant": float(other_flow[i]),
            "flow_loss_delta": float(flow_delta[i]),
        }
        for i, row in enumerate(reference["actions"])
    ]
    return report, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path, help="Saved variant directory under results/")
    args = parser.parse_args()
    variant_path = args.results_dir.expanduser().resolve()
    reference = load_run(REFERENCE)
    variant = load_run(variant_path)
    report, rows = compare_run(reference, variant)
    (variant_path / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (variant_path / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{report['variant']}: flow Δ {report['flow_loss_shared78_delta']:+.6g}, "
          f"action MAE {report['action_mae']:.6g}, "
          f"latency {report['mean_latency_ms_variant']:.2f} ms, "
          f"VRAM {report['peak_vram_gib_variant']:.2f} GiB")
    print(f"Saved {variant_path / 'comparison.json'} and {variant_path / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
