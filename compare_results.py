#!/usr/bin/env python3
"""Compare the released and BF16-action-head validation runs; no model load needed."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "baseline_results"
BF16 = ROOT / "bf16_action_head_results"
OUT = ROOT / "comparison_results"
PAIR_KEYS = ("sample_id", "episode_index", "frame_index")
ACTION_KEYS = PAIR_KEYS + ("seed", "instruction")
SUMMARY_KEYS = (
    "checkpoint", "checkpoint_step", "checkpoint_revision", "dataset",
    "dataset_revision", "dataset_sha256", "samples", "seed", "inference_steps",
    "action_shape", "torch_version", "torch_cuda_version", "cuda_device",
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def paired(base: list[dict], other: list[dict], keys: tuple[str, ...], label: str) -> None:
    if not base or len(base) != len(other):
        raise ValueError(f"{label}: empty data or differing sample counts")
    for index, (left, right) in enumerate(zip(base, other)):
        if any(left.get(key) != right.get(key) for key in keys):
            raise ValueError(f"{label}: sample {index} is not paired on {keys}")
        if left["sample_id"] != index:
            raise ValueError(f"{label}: sample IDs are not sequential at row {index}")


def bar_pair(axis, values: list[float], title: str, unit: str, color: str) -> None:
    bars = axis.bar(["Released", "BF16 head"], values, color=["#64748b", color], width=0.58)
    axis.set_title(title)
    axis.set_ylabel(unit)
    axis.set_ylim(0, max(values) * 1.22 if max(values) > 0 else 1)
    axis.grid(axis="y", alpha=0.25)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, values):
        axis.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.3f}",
                  ha="center", va="bottom", fontsize=9)


def main() -> None:
    base = read_json(BASE / "summary.json")
    bf16 = read_json(BF16 / "summary.json")
    if bf16.get("action_head_dtype") != "bf16":
        raise ValueError("bf16_action_head_results is not a BF16 action-head run")
    if base.get("action_head_dtype", "fp32") != "fp32":
        raise ValueError("baseline_results is not the released FP32 action-head run")
    for key in SUMMARY_KEYS:
        if base.get(key) != bf16.get(key):
            raise ValueError(f"Run settings differ at {key}: {base.get(key)!r} vs {bf16.get(key)!r}")
    for filename in ("flow_noise.npy", "flow_sigmas.npy"):
        if digest(BASE / filename) != digest(BF16 / filename):
            raise ValueError(f"The paired runs used different {filename}")

    base_actions = read_jsonl(BASE / "actions.jsonl")
    bf16_actions = read_jsonl(BF16 / "actions.jsonl")
    base_losses = read_jsonl(BASE / "flow_losses.jsonl")
    bf16_losses = read_jsonl(BF16 / "flow_losses.jsonl")
    paired(base_actions, bf16_actions, ACTION_KEYS, "actions")
    paired(base_losses, bf16_losses, PAIR_KEYS, "flow losses")
    paired(base_actions, base_losses, PAIR_KEYS, "released actions and losses")
    paired(bf16_actions, bf16_losses, PAIR_KEYS, "BF16 actions and losses")

    released = np.asarray([row["action"] for row in base_actions], dtype=np.float32)
    converted = np.asarray([row["action"] for row in bf16_actions], dtype=np.float32)
    if released.shape != (len(base_actions), 30, 80) or converted.shape != released.shape:
        raise ValueError("Action arrays must have shape [samples, 30, 80]")
    if not np.isfinite(released).all() or not np.isfinite(converted).all():
        raise ValueError("Action arrays contain nonfinite values")
    difference = converted - released
    per_sample_mae = np.abs(difference).mean(axis=(1, 2))
    per_dim_mae = np.abs(difference).mean(axis=(0, 1))
    flat_released = released.reshape(len(released), -1)
    flat_converted = converted.reshape(len(converted), -1)
    numerator = np.sum(flat_released * flat_converted, axis=1, dtype=np.float64)
    denominator = np.linalg.norm(flat_released, axis=1) * np.linalg.norm(flat_converted, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)
    both_zero = (np.linalg.norm(flat_released, axis=1) == 0) & (np.linalg.norm(flat_converted, axis=1) == 0)
    cosine[both_zero] = 1.0

    base_flow = np.asarray([row["shared78_mse"] for row in base_losses], dtype=np.float64)
    bf16_flow = np.asarray([row["shared78_mse"] for row in bf16_losses], dtype=np.float64)
    if not np.isfinite(base_flow).all() or not np.isfinite(bf16_flow).all():
        raise ValueError("Flow loss contains nonfinite values")
    flow_delta = bf16_flow - base_flow
    episode_deltas = defaultdict(list)
    for row, delta in zip(base_losses, flow_delta):
        episode_deltas[int(row["episode_index"])].append(float(delta))
    episode_mean_delta = {str(key): float(np.mean(values)) for key, values in sorted(episode_deltas.items())}
    base_latency = float(base["mean_latency_seconds"])
    bf16_latency = float(bf16["mean_latency_seconds"])
    base_vram = int(base["peak_inference_vram_bytes"])
    bf16_vram = int(bf16["peak_inference_vram_bytes"])
    result = {
        "reference": "baseline_results",
        "variant": "bf16_action_head_results",
        "samples": len(base_actions),
        "action_mae": float(np.abs(difference).mean()),
        "action_mse": float(np.square(difference).mean()),
        "mean_action_cosine_similarity": float(np.mean(cosine)),
        "action_mae_body64": float(np.abs(difference[:, :, :64]).mean()),
        "action_mae_hands14": float(np.abs(difference[:, :, 64:78]).mean()),
        "action_mae_neck2": float(np.abs(difference[:, :, 78:]).mean()),
        "per_action_dimension_mae": per_dim_mae.astype(float).tolist(),
        "flow_loss_shared78_released": float(base_flow.mean()),
        "flow_loss_shared78_bf16": float(bf16_flow.mean()),
        "flow_loss_shared78_delta": float(flow_delta.mean()),
        "flow_loss_shared78_relative_change_percent": float(100 * flow_delta.mean() / base_flow.mean()) if base_flow.mean() else None,
        "flow_loss_delta_by_episode": episode_mean_delta,
        "mean_latency_ms_released": base_latency * 1000,
        "mean_latency_ms_bf16": bf16_latency * 1000,
        "latency_speedup_released_over_bf16": base_latency / bf16_latency,
        "peak_vram_gib_released": base_vram / 2**30,
        "peak_vram_gib_bf16": bf16_vram / 2**30,
        "peak_vram_saved_gib": (base_vram - bf16_vram) / 2**30,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (OUT / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "sample_id", "episode_index", "frame_index", "action_mae", "action_cosine",
            "flow_loss_released", "flow_loss_bf16", "flow_loss_delta",
        ])
        writer.writeheader()
        for index, row in enumerate(base_actions):
            writer.writerow({
                "sample_id": index, "episode_index": row["episode_index"],
                "frame_index": row["frame_index"], "action_mae": float(per_sample_mae[index]),
                "action_cosine": float(cosine[index]),
                "flow_loss_released": float(base_flow[index]),
                "flow_loss_bf16": float(bf16_flow[index]),
                "flow_loss_delta": float(flow_delta[index]),
            })

    fig, axes = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    fig.suptitle("Ψ₀/SONIC: released head vs BF16 action head", fontsize=16, fontweight="bold")
    scatter = axes[0, 0]
    scatter.scatter(base_flow, bf16_flow, s=18, alpha=0.65, color="#2563eb")
    lower = float(min(base_flow.min(), bf16_flow.min()))
    upper = float(max(base_flow.max(), bf16_flow.max()))
    margin = max((upper - lower) * 0.05, 1e-6)
    scatter.plot([lower - margin, upper + margin], [lower - margin, upper + margin],
                 color="#64748b", linestyle="--", linewidth=1)
    scatter.set_xlim(lower - margin, upper + margin)
    scatter.set_ylim(lower - margin, upper + margin)
    scatter.set_title("Paired flow loss: each point is one frame")
    scatter.set_xlabel("Released, shared 78 MSE")
    scatter.set_ylabel("BF16 head, shared 78 MSE")
    scatter.grid(alpha=0.2)

    episode_axis = axes[0, 1]
    episode_ids = list(episode_mean_delta)
    episode_values = [episode_mean_delta[key] for key in episode_ids]
    episode_axis.bar(episode_ids, episode_values,
                     color=["#dc2626" if value > 0 else "#059669" for value in episode_values])
    episode_axis.axhline(0, color="#334155", linewidth=1)
    episode_axis.set_title("Mean flow-loss change by episode")
    episode_axis.set_xlabel("Episode index")
    episode_axis.set_ylabel("BF16 − released MSE")
    episode_axis.grid(axis="y", alpha=0.2)
    episode_axis.set_axisbelow(True)

    dim_axis = axes[0, 2]
    dim_axis.plot(np.arange(80), per_dim_mae, linewidth=1.5, color="#7c3aed")
    dim_axis.axvspan(0, 63.5, color="#bfdbfe", alpha=0.25, label="Body 64")
    dim_axis.axvspan(63.5, 77.5, color="#fde68a", alpha=0.3, label="Hands 14")
    dim_axis.axvspan(77.5, 79.5, color="#fecaca", alpha=0.4, label="Neck 2")
    dim_axis.set_title("Predicted-action error by dimension")
    dim_axis.set_xlabel("Action dimension")
    dim_axis.set_ylabel("MAE vs released output")
    dim_axis.legend(loc="upper right", fontsize=8)
    dim_axis.grid(alpha=0.2)

    bar_pair(axes[1, 0], [float(base_flow.mean()), float(bf16_flow.mean())],
             "Mean shared-78 flow loss", "Velocity MSE ↓", "#2563eb")
    bar_pair(axes[1, 1], [base_latency * 1000, bf16_latency * 1000],
             "Mean action-chunk latency", "Milliseconds ↓", "#2563eb")
    bar_pair(axes[1, 2], [base_vram / 2**30, bf16_vram / 2**30],
             "Peak inference GPU allocation", "GiB ↓", "#2563eb")
    fig.text(0.5, 0.005,
             f"{len(base_actions)} paired UniFolM validation frames · neck excluded from flow loss · "
             f"action MAE {result['action_mae']:.6g} · cosine {result['mean_action_cosine_similarity']:.6f}",
             ha="center", fontsize=10)
    fig.savefig(OUT / "comparison.png", dpi=160)
    plt.close(fig)

    print(f"Action MAE: {result['action_mae']:.6g}; cosine: {result['mean_action_cosine_similarity']:.6f}")
    print(f"Shared-78 flow loss: {result['flow_loss_shared78_released']:.6g} → "
          f"{result['flow_loss_shared78_bf16']:.6g} (Δ {result['flow_loss_shared78_delta']:+.6g})")
    print(f"Latency: {base_latency * 1000:.2f} → {bf16_latency * 1000:.2f} ms; "
          f"peak VRAM: {base_vram / 2**30:.2f} → {bf16_vram / 2**30:.2f} GiB")
    print(f"Saved {OUT / 'comparison.png'}, {OUT / 'comparison.json'}, and {OUT / 'per_sample.csv'}")


if __name__ == "__main__":
    main()
