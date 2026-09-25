#!/usr/bin/env python3
"""Plot every saved Ψ₀ precision variant in results/, one panel per run."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_results import REFERENCE, compare_run, discover_runs, load_run


def main() -> None:
    paths = discover_runs()
    root = REFERENCE.parent
    reference = load_run(REFERENCE)
    runs = []
    for path in paths:
        run = load_run(path)
        runs.append((path, run, compare_run(reference, run)))
    base_flow = np.asarray([row["shared78_mse"] for row in reference["losses"]])
    columns = min(3, len(runs))
    rows_count = math.ceil(len(runs) / columns)
    fig = plt.figure(figsize=(5.3 * columns, 4.7 * rows_count), constrained_layout=True)
    fig.suptitle("Ψ₀/SONIC validation · paired with released BF16 VLM / FP32 expert",
                 fontsize=15, fontweight="bold")

    for index, (path, run, report) in enumerate(runs, start=1):
        axis = fig.add_subplot(rows_count, columns, index)
        if path == REFERENCE:
            axis.hist(base_flow, bins=min(20, max(5, len(base_flow) // 5)), color="#64748b")
            axis.set_xlabel("Shared-78 flow MSE")
            axis.set_ylabel("Frames")
        else:
            variant_flow = np.asarray([row["shared78_mse"] for row in run["losses"]])
            flow_delta = variant_flow - base_flow
            axis.hist(flow_delta, bins=min(20, max(5, len(flow_delta) // 5)), color="#2563eb")
            axis.axvline(0, linestyle="--", linewidth=1, color="#64748b")
            axis.axvline(flow_delta.mean(), linewidth=1.5, color="#dc2626")
            axis.set_xlabel("Flow MSE change vs released reference")
            axis.set_ylabel("Frames")
        axis.set_title(path.name.replace("_", " "), fontsize=11)
        axis.grid(alpha=0.2)
        if path == REFERENCE:
            details = f"Mean flow MSE {report['flow_loss_shared78_reference']:.4g}"
        else:
            percent = report["flow_loss_shared78_relative_change_percent"]
            percent_text = "n/a" if percent is None else f"{percent:+.2f}%"
            details = (
                f"Mean flow Δ {report['flow_loss_shared78_delta']:+.4g} "
                f"({percent_text})\n"
                f"Output MAE {report['action_mae']:.4g}  "
                f"MSE {report['action_mse']:.4g}  "
                f"cosine {report['mean_action_cosine_similarity']:.4f}\n"
                f"Latency {report['mean_latency_ms_variant']:.1f} ms  "
                f"Peak VRAM {report['peak_vram_gib_variant']:.2f} GiB"
            )
        axis.text(0.02, 0.98, details, transform=axis.transAxes, va="top", fontsize=8,
                  bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9,
                        "edgecolor": "#cbd5e1"})

    output = root / "overview.png"
    fig.savefig(output, dpi=160)
    plt.close(fig)
    print(f"Plotted {len(runs)} runs in {output}")


if __name__ == "__main__":
    main()
