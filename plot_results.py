#!/usr/bin/env python3
"""Plot every saved Ψ₀ precision variant in results/, one panel per run."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from compare_results import REFERENCE, load_run, compare_run


def main() -> None:
    root = REFERENCE.parent
    if not REFERENCE.is_dir():
        raise FileNotFoundError(f"Released reference missing: {REFERENCE}")
    paths = sorted(path for path in root.iterdir()
                   if path.is_dir() and path.name.startswith("vlm_")
                   and (path / "summary.json").is_file())
    paths.remove(REFERENCE)
    paths.insert(0, REFERENCE)

    reference = load_run(REFERENCE)
    runs = [(path, *compare_run(reference, load_run(path))) for path in paths]
    flow_arrays = [np.asarray([row["flow_loss_variant"] for row in rows])
                   for _, _, rows in runs]
    lower = min(float(values.min()) for values in flow_arrays)
    upper = max(float(values.max()) for values in flow_arrays)
    margin = max((upper - lower) * 0.05, 1e-6)

    columns = min(3, len(runs))
    rows_count = math.ceil(len(runs) / columns)
    fig = plt.figure(figsize=(5.3 * columns, 4.7 * rows_count), constrained_layout=True)
    fig.suptitle("Ψ₀/SONIC validation · paired with released BF16 VLM / FP32 expert",
                 fontsize=15, fontweight="bold")

    for index, (path, report, rows) in enumerate(runs, start=1):
        axis = fig.add_subplot(rows_count, columns, index)
        base_flow = np.asarray([row["flow_loss_reference"] for row in rows])
        variant_flow = np.asarray([row["flow_loss_variant"] for row in rows])
        if path == REFERENCE:
            axis.hist(base_flow, bins=min(20, max(5, len(base_flow) // 5)), color="#64748b")
            axis.set_xlabel("Shared-78 flow MSE")
            axis.set_ylabel("Frames")
        else:
            axis.scatter(base_flow, variant_flow, s=17, alpha=0.65, color="#2563eb")
            axis.plot([lower - margin, upper + margin], [lower - margin, upper + margin],
                      linestyle="--", linewidth=1, color="#64748b")
            axis.set_xlim(lower - margin, upper + margin)
            axis.set_ylim(lower - margin, upper + margin)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel("Reference flow MSE")
            axis.set_ylabel("Variant flow MSE")
        axis.set_title(path.name.replace("_", " "), fontsize=11)
        axis.grid(alpha=0.2)
        details = (
            f"Flow Δ {report['flow_loss_shared78_delta']:+.4g}   "
            f"Action MAE {report['action_mae']:.4g}\n"
            f"Latency {report['mean_latency_ms_variant']:.1f} ms   "
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
