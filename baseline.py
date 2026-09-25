#!/usr/bin/env python3
"""Save the unquantized Ψ₀/SONIC reference on fixed UniFolM validation frames.

All user settings are in baseline_config.json. The public pack has body and hand
targets, but no neck targets, so flow loss is measured on the shared 78 values.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "baseline_config.json"
RESULTS = ROOT / "baseline_results"
STEP = 40000
CHECKPOINT_REVISION = "4c6f9776fc5b18d87945254175e38bb74b9d7748"
DATA_REVISION = "e78fb93cc28912a3031a10b8656d32d7f0a2b867"
DATA_SHA256 = "3d264d6454d59e83be5dadaa17f122b732a8c6cb86a2429c33c1f4fc912fc4b7"
IMAGE_KEY = "observation.images.egocentric"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dtype_counts(module) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for parameter in module.parameters():
        counts[str(parameter.dtype)] += parameter.numel()
    return dict(sorted(counts.items()))


def load_examples(dataset: Path, samples: int, horizon: int) -> list[dict]:
    info = json.loads((dataset / "meta/info.json").read_text())
    for key, width in (("observation.state", 43), ("action", 36), ("action.body_token", 64)):
        if info["features"].get(key, {}).get("shape") != [width]:
            raise ValueError(f"Unexpected {key} shape in {dataset}")
    if IMAGE_KEY not in info["features"]:
        raise ValueError(f"Missing {IMAGE_KEY}")
    episodes = [json.loads(line) for line in (dataset / "meta/episodes.jsonl").read_text().splitlines() if line]
    if not episodes:
        raise ValueError("Validation pack has no episodes")
    allocations = [samples // len(episodes) + (i < samples % len(episodes)) for i in range(len(episodes))]
    selected = []
    for episode, amount in zip(episodes, allocations):
        if amount == 0:
            continue
        number = int(episode["episode_index"])
        args = {"episode_index": number, "episode_chunk": number // int(info["chunks_size"]), "video_key": IMAGE_KEY}
        parquet = dataset / info["data_path"].format(**args)
        video = dataset / info["video_path"].format(**args)
        table = pq.read_table(parquet, columns=[
            "frame_index", "observation.state", "action", "action.body_token", "task_description"
        ])
        last = table.num_rows - horizon
        if last < 0 or amount > last + 1:
            raise ValueError(f"Episode {number} has too few full action windows")
        positions = np.linspace(0, last, amount, dtype=int).tolist()
        frame_to_position = {int(table["frame_index"][pos].as_py()): pos for pos in positions}
        frames = {}
        with av.open(str(video)) as container:
            for frame_number, frame in enumerate(container.decode(video=0)):
                if frame_number in frame_to_position:
                    frames[frame_number] = frame.to_image().convert("RGB")
                if len(frames) == len(frame_to_position):
                    break
        if len(frames) != len(frame_to_position):
            raise ValueError(f"Selected frames missing in {video}")
        for frame_number, pos in sorted(frame_to_position.items()):
            body = np.asarray(table["action.body_token"].slice(pos, horizon).to_pylist(), dtype=np.float32)
            hands = np.asarray(table["action"].slice(pos, horizon).to_pylist(), dtype=np.float32)[:, :14]
            state = np.asarray(table["observation.state"][pos].as_py(), dtype=np.float32)
            instruction = str(table["task_description"][pos].as_py()).lower()
            actions = np.concatenate((body, hands), axis=-1)
            if actions.shape != (horizon, 78) or state.shape != (43,) or not instruction:
                raise ValueError(f"Invalid episode {number} frame {frame_number}")
            selected.append(dict(episode_index=number, frame_index=frame_number,
                                 image=frames[frame_number], state=state,
                                 actions=actions, instruction=instruction))
    if len(selected) != samples:
        raise ValueError(f"Selected {len(selected)} frames, expected {samples}")
    return selected


def get_projections(instructions: list[str], checkpoint: Path, config, device: str):
    cached = torch.load(checkpoint / "clip_pooled_cache.pt", map_location="cpu", weights_only=True)
    if not isinstance(cached, dict):
        raise ValueError("Checkpoint CLIP cache is malformed")
    vectors = {key: value.detach().cpu() for key, value in cached.items() if key in instructions}
    missing = sorted(set(instructions) - vectors.keys())
    if missing:
        from transformers import CLIPTextModelWithProjection, CLIPTokenizer

        tokenizer = CLIPTokenizer.from_pretrained(config.pooled_text_encoder_path)
        encoder = CLIPTextModelWithProjection.from_pretrained(
            config.pooled_text_encoder_path, torch_dtype=torch.bfloat16
        ).to(device).eval()
        if encoder.config.projection_dim != config.pooled_projection_dim:
            raise ValueError("CLIP projection dimension differs from checkpoint")
        with torch.inference_mode():
            for instruction in missing:
                tokens = tokenizer([instruction], padding=True, truncation=True,
                                   max_length=tokenizer.model_max_length, return_tensors="pt").to(device)
                vectors[instruction] = encoder(**tokens).text_embeds[0].detach().cpu()
        del encoder
        torch.cuda.empty_cache()
    for key, vector in vectors.items():
        if tuple(vector.shape) != (config.pooled_projection_dim,):
            raise ValueError(f"Wrong CLIP vector size for {key!r}")
    return vectors


def normalize_actions(actions78: np.ndarray, field) -> np.ndarray:
    if field.action_norm_type != "bounds" or field.use_norm_mask:
        raise ValueError("Expected checkpoint bounds normalization without a norm mask")
    low = np.asarray(field.action_min, dtype=np.float32)
    high = np.asarray(field.action_max, dtype=np.float32)
    if low.shape != (80,) or high.shape != (80,):
        raise ValueError("Expected 80 checkpoint action bounds")
    raw = np.pad(actions78, ((0, 0), (0, 2)))
    ill = np.abs(high - low) < 1e-4 * (np.abs(high) + np.abs(low) + 1e-8)
    safe_high = np.where(ill, 1.0, high)
    values = np.where(ill, raw, (raw - low) / (safe_high - low) * 2 - 1)
    values = np.clip(values, -1, 1).astype(np.float32)
    values[:, 78:] = 0  # Unknown neck targets; those dimensions are excluded from loss.
    return values


def evaluate_flow(model, image, state, instruction, projection, target, sigma, noise, train_steps):
    from qwen_vl_utils import process_vision_info

    messages = [[{"role": "user", "content": [
        {"type": "image", "image": image}, {"type": "text", "text": instruction}
    ]}]]
    prompt = model.vlm_processor.apply_chat_template(messages[0], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    inputs = model.vlm_processor(text=[prompt], images=image_inputs, videos=video_inputs,
                                 padding=True, return_tensors="pt").to(model.device)
    clean = torch.from_numpy(target).unsqueeze(0).to(model.device)
    epsilon = torch.from_numpy(noise).unsqueeze(0).to(model.device)
    timestep = torch.tensor([sigma * train_steps], device=model.device, dtype=torch.float32)
    noisy = (1 - sigma) * clean + sigma * epsilon
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        prediction = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"],
            action_samples=noisy, states=state, timestep=timestep,
            traj2ds=None, pooled_projections=projection,
        ).action
    squared = (prediction.float() - (epsilon - clean)).square()[0, :, :78]
    if not torch.isfinite(squared).all():
        raise ValueError("Nonfinite flow loss")
    return [float(squared[:, a:b].mean().item()) for a, b in ((0, 78), (0, 64), (64, 78))]


def main() -> None:
    settings = json.loads(CONFIG.read_text())
    repo = Path(settings["psi_repo"]).expanduser().resolve()
    dataset = Path(settings["validation_dataset"]).expanduser()
    dataset = (dataset if dataset.is_absolute() else repo / dataset).resolve()
    if not (repo / "src/psi/models/psi0.py").is_file():
        raise FileNotFoundError(f"Set psi_repo in {CONFIG} to the official Psi0 checkout")
    if not (dataset / "meta/info.json").is_file():
        raise FileNotFoundError(f"Validation dataset missing at {dataset}; run baseline_setup.sh")
    archive = dataset.parent / "sonic/unifolm_sonic_lerobot_val.zip"
    if not archive.is_file() or file_hash(archive) != DATA_SHA256:
        raise ValueError(f"Expected pinned validation archive at {archive}, SHA256 {DATA_SHA256}")
    checkpoint = repo / "cache/checkpoints" / settings["checkpoint"]
    weights = checkpoint / f"checkpoints/ckpt_{STEP}/model.safetensors"
    for path in (weights, checkpoint / "argv.txt", checkpoint / "run_config.json", checkpoint / "clip_pooled_cache.pt"):
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file missing: {path}; run baseline_setup.sh")
    if weights.stat().st_size < 1_000_000:
        raise ValueError(f"Checkpoint weights are incomplete: {weights}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability(0)
    if f"sm_{major}{minor}" not in torch.cuda.get_arch_list():
        raise RuntimeError(f"PyTorch {torch.__version__} does not support this GPU")
    sys.path.insert(0, str(repo / "src"))
    from psi.models.psi0 import Psi0Model
    from psi.utils import apply_legacy_model_config_defaults, parse_args_to_tyro_config

    template = parse_args_to_tyro_config(checkpoint / "argv.txt")
    saved = json.loads((checkpoint / "run_config.json").read_text())
    launch = template.model_validate(apply_legacy_model_config_defaults(saved))
    if (launch.model.action_dim, launch.model.action_chunk_size, launch.model.odim) != (80, 30, 45):
        raise ValueError("Checkpoint dimensions are not the expected SONIC 80/30/45")
    if launch.model.noise_scheduler != "flow" or launch.model.pooled_text_encoder != "clip":
        raise ValueError("Expected the CLIP-conditioned SONIC flow checkpoint")
    field = launch.data.transform.field
    if not field.normalize_state or field.state_min is None or field.action_min is None:
        raise ValueError("Checkpoint run_config lacks embedded normalization statistics")
    seed, samples, steps = (int(settings[key]) for key in ("seed", "samples", "inference_steps"))
    if samples < 1 or steps < 1:
        raise ValueError("samples and inference_steps must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    examples = load_examples(dataset, samples, 30)
    device = "cuda:0"
    model = Psi0Model.from_pretrained(checkpoint, STEP, launch, device=device).to(device).eval()
    vectors = get_projections([row["instruction"] for row in examples], checkpoint, launch.model, device)
    resize = launch.data.transform.model.resize()
    crop = launch.data.transform.model.center_crop()
    for row in examples:
        row["image"] = crop(resize(row["image"]))
        padded = np.pad(row["state"], (0, 2))
        row["state"] = torch.from_numpy(field.normalize_state_func(padded)).reshape(1, 1, 45).to(device)
        row["target"] = normalize_actions(row["actions"], field)
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.save(vectors, RESULTS / "pooled_projections.pt")
    rng = torch.Generator(device="cpu").manual_seed(seed)
    sigmas = torch.rand(samples, generator=rng).numpy().astype(np.float32)
    noise = torch.randn((samples, 30, 80), generator=rng).numpy().astype(np.float32)
    np.save(RESULTS / "flow_sigmas.npy", sigmas)
    np.save(RESULTS / "flow_noise.npy", noise)

    def predict(row):
        projection = vectors[row["instruction"]].unsqueeze(0).to(device)
        with torch.inference_mode():
            return model.predict_action(
                observations=[[row["image"]]], states=row["state"],
                instructions=[row["instruction"]], num_inference_steps=steps,
                traj2ds=None, pooled_projections=projection,
            )

    predict(examples[0])  # Warmup on a real input.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    actions, latencies = [], []
    for index, row in enumerate(examples):
        sample_seed = seed + index
        torch.manual_seed(sample_seed)
        torch.cuda.manual_seed_all(sample_seed)
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = predict(row)
        torch.cuda.synchronize()
        latency = time.perf_counter() - start
        values = output.detach().float().cpu().numpy()
        if values.shape != (1, 30, 80) or not np.isfinite(values).all():
            raise ValueError(f"Invalid prediction for sample {index}: {values.shape}")
        latencies.append(latency)
        actions.append(dict(sample_id=index, seed=sample_seed,
                            episode_index=row["episode_index"], frame_index=row["frame_index"],
                            instruction=row["instruction"], action=values[0].tolist()))
        print(f"Inference {index + 1}/{samples}: {latency:.3f} s", flush=True)
    inference_peak = int(torch.cuda.max_memory_allocated(device))

    losses = []
    for index, row in enumerate(examples):
        projection = vectors[row["instruction"]].unsqueeze(0).to(device)
        shared, body, hands = evaluate_flow(
            model, row["image"], row["state"], row["instruction"], projection,
            row["target"], float(sigmas[index]), noise[index],
            int(launch.model.train_diffusion_steps),
        )
        losses.append(dict(sample_id=index, episode_index=row["episode_index"],
                           frame_index=row["frame_index"], shared78_mse=shared,
                           body64_mse=body, hands14_mse=hands))
        print(f"Flow loss {index + 1}/{samples}: {shared:.5f}", flush=True)

    for name, rows in (("actions.jsonl", actions), ("flow_losses.jsonl", losses)):
        with (RESULTS / name).open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
    summary = {
        "reference": "unquantized_psi0_sonic",
        "checkpoint": settings["checkpoint"], "checkpoint_step": STEP,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "dataset": "USC-PSI-Lab/psi-data/sonic/unifolm_sonic_lerobot_val.zip",
        "dataset_revision": DATA_REVISION, "dataset_sha256": DATA_SHA256,
        "selection": "equal per episode, evenly spaced full 30-step windows",
        "samples": samples, "seed": seed, "inference_steps": steps,
        "action_shape": [30, 80], "action_layout": "body64 + hands14 + neck2",
        "flow_loss_scope": "mean velocity-prediction MSE on shared 78 dims; neck masked",
        "flow_loss_shared78_mse": float(np.mean([row["shared78_mse"] for row in losses])),
        "flow_loss_body64_mse": float(np.mean([row["body64_mse"] for row in losses])),
        "flow_loss_hands14_mse": float(np.mean([row["hands14_mse"] for row in losses])),
        "flow_loss_shared78_by_episode": {
            str(episode): float(np.mean([row["shared78_mse"] for row in losses
                                         if row["episode_index"] == episode]))
            for episode in sorted({row["episode_index"] for row in losses})
        },
        "flow_noise_file": "flow_noise.npy", "flow_sigmas_file": "flow_sigmas.npy",
        "pooled_projections_file": "pooled_projections.pt",
        "cuda_device": torch.cuda.get_device_name(device), "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "parameter_counts_by_dtype": {"vlm": dtype_counts(model.vlm_model),
                                      "action_expert": dtype_counts(model.action_header)},
        "inference_autocast_dtype": "torch.bfloat16",
        "mean_latency_seconds": float(np.mean(latencies)),
        "median_latency_seconds": float(np.median(latencies)),
        "p95_latency_seconds": float(np.percentile(latencies, 95)),
        "throughput_samples_per_second": float(samples / sum(latencies)),
        "peak_inference_vram_bytes": inference_peak,
        "actions_file": "actions.jsonl", "flow_losses_file": "flow_losses.jsonl",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved unquantized validation reference in {RESULTS}")


if __name__ == "__main__":
    main()
