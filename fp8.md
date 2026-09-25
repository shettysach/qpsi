# Ψ₀ precision experiments

The released SONIC checkpoint is the reference: its VLM weights are BF16 and its action expert weights are FP32. Inference uses BF16 autocast. The five planned runs are:

| Experiment | VLM | Action expert | Command arguments |
| --- | --- | --- | --- |
| Released reference | BF16 | FP32 | `--vlm bf16 --act fp32` |
| Action expert BF16 | BF16 | BF16 | `--vlm bf16 --act bf16` |
| Action expert FP8 | BF16 | FP8 | `--vlm bf16 --act fp8` |
| VLM FP8 | FP8 | FP32 | `--vlm fp8 --act fp32` |
| Both FP8 | FP8 | FP8 | `--vlm fp8 --act fp8` |

See [baseline.md](baseline.md) for setup and the three commands: [evaluate.py](evaluate.py) writes a validation run, [compare_results.py](compare_results.py) compares one saved run to the reference, and [plot_results.py](plot_results.py) plots all saved runs.

`fp8` means TorchAO dynamic W8A8 E4M3 on eligible `nn.Linear` layers, not every tensor in the component. Linear dimensions must be multiples of 16 and at least 64. The VLM's tied output/embedding weight is excluded. Norms, embeddings, nonlinear operations, and unsupported projections retain their released precision. The action output linear layer is included when eligible. Each run's `summary.json` records the exact converted layer names and parameter count.

Every comparison uses the same selected UniFolM observations, CLIP projections, action seeds, flow noise, and timesteps. Compare shared-78 flow MSE, paired generated-action error, inference latency, and peak allocated VRAM. UniFolM has no neck targets, so its two neck dimensions are excluded from flow loss. These numbers measure offline numerical effects, not robot task success.
