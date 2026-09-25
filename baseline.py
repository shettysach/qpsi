#!/usr/bin/env python3
"""Save repeatable BF16 outputs from the released Ψ₀ + SONIC checkpoint.

Edit baseline_config.json, create the uv baseline environment, then run:
    /path/to/qpsi/.venv/bin/python /path/to/qpsi/baseline.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image


CONFIG_FILE = Path(__file__).with_name("baseline_config.json")
CHECKPOINT_STEP = 40000
RESULTS_DIR = Path(__file__).with_name("baseline_results")


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in module.parameters():
        counts[str(parameter.dtype)] += parameter.numel()
    return dict(sorted(counts.items()))


def main() -> None:
    settings = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    repo = Path(settings["psi_repo"]).expanduser().resolve()
    if not (repo / "src" / "psi" / "models" / "psi0.py").is_file():
        raise FileNotFoundError(f"Set psi_repo in {CONFIG_FILE} to the official Psi0 checkout")
    # Use the configured checkout directly; the baseline environment contains
    # its inference dependencies, without installing the full training project.
    sys.path.insert(0, str(repo / "src"))

    checkpoint_id = settings["checkpoint"]
    cache_dir = repo / "cache" / "checkpoints"
    checkpoint_dir = cache_dir / checkpoint_id
    checkpoint_step = CHECKPOINT_STEP
    checkpoint_file = checkpoint_dir / "checkpoints" / f"ckpt_{checkpoint_step}" / "model.safetensors"
    pooled_cache = checkpoint_dir / "clip_pooled_cache.pt"
    required = [checkpoint_file, checkpoint_dir / "argv.txt", checkpoint_dir / "run_config.json", pooled_cache]
    output_dir = RESULTS_DIR
    seed = int(settings["seed"])
    samples = int(settings["samples"])
    inference_steps = int(settings["inference_steps"])
    if samples < 1 or inference_steps < 1:
        raise ValueError("samples and inference_steps must be positive")

    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Checkpoint files are missing. Run baseline_setup.sh first. Missing: "
            + ", ".join(missing)
        )
    if checkpoint_file.stat().st_size < 1_000_000:
        raise ValueError(f"Checkpoint contains a pointer or incomplete weights: {checkpoint_file}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this baseline")
    capability = torch.cuda.get_device_capability(0)
    architecture = f"sm_{capability[0]}{capability[1]}"
    supported = torch.cuda.get_arch_list()
    if architecture not in supported:
        raise RuntimeError(
            f"This PyTorch wheel does not support {architecture} "
            f"({torch.cuda.get_device_name(0)}). Installed wheel: {torch.__version__}, "
            f"CUDA {torch.version.cuda}; supported architectures: {supported}. "
            "Recreate the uv baseline environment with baseline_env.sh."
        )

    from psi.models.psi0 import Psi0Model
    from psi.utils import apply_legacy_model_config_defaults, parse_args_to_tyro_config

    template = parse_args_to_tyro_config(checkpoint_dir / "argv.txt")
    run_config = json.loads((checkpoint_dir / "run_config.json").read_text(encoding="utf-8"))
    launch_config = template.model_validate(apply_legacy_model_config_defaults(run_config))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = "cuda:0"
    model = Psi0Model.from_pretrained(checkpoint_dir, checkpoint_step, launch_config, device=device)
    model.to(device)
    model.eval()

    # SONIC's combined time/instruction embedding requires a CLIP vector.
    # The release includes the cached instruction vectors used by deployment.
    pooled = torch.load(pooled_cache, map_location="cpu", weights_only=True)
    if not isinstance(pooled, dict) or not pooled:
        raise ValueError(f"No instruction embeddings in {pooled_cache}")
    instruction = sorted(pooled)[0]
    projection = pooled[instruction].to(device).unsqueeze(0)
    expected_dim = int(launch_config.model.pooled_projection_dim)
    if projection.shape != (1, expected_dim):
        raise ValueError(f"Expected CLIP projection (1, {expected_dim}), got {tuple(projection.shape)}")

    # A fixed input isolates quantization's numerical effect. It cannot measure
    # task accuracy or validation loss.
    image = Image.new("RGB", (480, 270), color=(128, 128, 128))
    states = torch.zeros(
        (1, int(launch_config.model.observation_horizon), int(launch_config.model.odim)),
        device=device,
    )
    predict = dict(
        observations=[[image]],
        states=states,
        instructions=[instruction],
        num_inference_steps=inference_steps,
        traj2ds=None,
        pooled_projections=projection,
    )

    dtypes = {
        "vlm": parameter_counts(model.vlm_model),
        "action_expert": parameter_counts(model.action_header),
    }
    print("CUDA:", torch.cuda.get_device_name(device))
    print("Parameter counts by dtype:", json.dumps(dtypes, indent=2))
    print("Fixture instruction:", instruction)

    # Warm inference before timing. The measured calls reseed afterward.
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model.predict_action(**predict)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    actions_path = output_dir / "actions.jsonl"
    latencies = []
    with actions_path.open("w", encoding="utf-8") as output:
        for sample_index in range(samples):
            sample_seed = seed + sample_index
            torch.manual_seed(sample_seed)
            torch.cuda.manual_seed_all(sample_seed)
            torch.cuda.synchronize()
            start = time.perf_counter()
            actions = model.predict_action(**predict)
            torch.cuda.synchronize()
            latency = time.perf_counter() - start
            values = actions.detach().float().cpu().numpy()
            if values.shape != (1, model.action_horizon, model.action_dim):
                raise ValueError(f"Unexpected action shape: {values.shape}")
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite actions at sample {sample_index}")
            latencies.append(latency)
            output.write(json.dumps({
                "sample_id": sample_index,
                "seed": sample_seed,
                "action": values[0].tolist(),
            }) + "\n")
            print(f"Sample {sample_index + 1}/{samples}: {latency:.3f} s", flush=True)

    summary = {
        "checkpoint": checkpoint_id,
        "checkpoint_step": checkpoint_step,
        "cuda_device": torch.cuda.get_device_name(device),
        "parameter_counts_by_dtype": dtypes,
        "fixture": {
            "image": "uniform RGB 128, 480x270",
            "states": f"zeros, {tuple(states.shape)}",
            "instruction": instruction,
            "seed": seed,
            "samples": samples,
            "inference_steps": inference_steps,
            "actions_are_normalized": True,
        },
        "action_shape": [model.action_horizon, model.action_dim],
        "mean_latency_seconds": float(np.mean(latencies)),
        "throughput_samples_per_second": float(samples / sum(latencies)),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated(device)),
        "actions_file": str(actions_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {actions_path} and {summary_path}")


if __name__ == "__main__":
    main()
