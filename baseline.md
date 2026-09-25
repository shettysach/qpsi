# Ψ₀/SONIC validation

This project measures how precision changes affect the released `multi-task.psi-dream.2609092156` checkpoint on fixed examples from the public [UniFolM SONIC validation archive](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/blob/main/sonic/unifolm_sonic_lerobot_val.zip). The released reference has BF16 VLM weights and FP32 action expert weights. Inference uses BF16 autocast. It is not an all-FP32 computation.

The archive supplies 64 body and 14 hand action targets. The checkpoint also predicts two neck values, which are excluded from flow loss because the archive has no neck targets. This is an offline, cross-domain loss measurement, not the authors' Psi-Dream validation score or robot task success.

## Environment and data

Keep the [official Ψ₀ checkout](https://github.com/physical-superintelligence-lab/Psi0) at `/home/sach/Desktop/Psi0`. The checkpoint and dataset paths, seed, 100 samples, and 10 inference steps are fixed near the top of [evaluate.py](evaluate.py). On the RTX 5090 machine:

```bash
cd /path/to/qpsi
uv sync --python 3.11
uv run --python 3.11 python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); print(torch.zeros(1, device="cuda"))'
bash baseline_setup.sh
```

`uv sync` uses [pyproject.toml](pyproject.toml) and the CUDA 12.8 PyTorch wheel source. The setup script pins both downloads and checks the validation archive. If Hugging Face asks for authentication, run `uvx hf auth login` and enter your token in the CLI.

## 1. Evaluate one precision variant

```bash
bash run_all.sh
```

The [runner](run_all.sh) evaluates all five variants with `uv run`, then generates the comparison table and plot. It skips completed runs. To run one variant yourself, use `uv run --python 3.11 python evaluate.py --vlm bf16 --act fp32`, replacing the two dtypes as needed. Each invocation writes `results/vlm_<dtype>_act_<dtype>/` and refuses to overwrite an existing run. Other variants reuse the reference's selected frames, CLIP projections, flow noise, and timesteps. Every run saves:

| File | Contents |
| --- | --- |
| `summary.json` | Precision settings, converted FP8 layers, checkpoint and dataset identity, mean flow loss, timing, and peak allocated GPU memory. |
| `actions.jsonl` | Normalized 30×80 predicted action chunk for each frame, with its paired seed. |
| `flow_losses.jsonl` | Shared 78-value, body 64-value, and hand 14-value velocity MSE for each frame. |
| `flow_noise.npy`, `flow_sigmas.npy` | Fixed noise and timesteps for paired flow loss. |
| `pooled_projections.pt` | Frozen CLIP instruction projections used by the run. |

FP8 uses TorchAO dynamic W8A8 E4M3 quantization on eligible linear layers of the selected component. The remaining layers keep their original precision. `summary.json` lists every converted layer and its weight parameter count. `act bf16` casts the entire action expert, including its output projection, to BF16.

## 2. Compare all saved runs with the reference

```bash
uv run --python 3.11 python compare_results.py
```

The script finds every completed run under `results/`, checks checkpoint, dataset, software, GPU, frame IDs, inference seeds, noise, and timesteps, then prints one comparison table. It saves the same table as `results/comparison.md` and `results/comparison.csv`. Columns show shared-78 flow MSE, absolute and relative flow-loss change, generated-output MAE/MSE/cosine, latency, and peak VRAM. A positive flow-loss change means greater velocity-prediction error on these UniFolM frames.

The comparison reference is always `vlm_bf16_act_fp32`, the **released** model. It is not the all-BF16 variant. Action MAE, MSE, and cosine compare two *generated normalized action chunks* on the same seeded observation; they are not prediction error against the recorded action. The cosine measures agreement between 30×80 output vectors, not physical direction or robot success. Recorded actions are used to construct the flow-matching target instead. A single recorded trajectory is a weak target for direct generated-action MSE because the model samples one of multiple plausible action chunks.

## 3. Plot all saved runs

```bash
uv run --python 3.11 python plot_results.py
```

This scans `results/` and writes `results/overview.png` with one panel per run. The reference panel shows its flow-loss distribution. Every other panel shows the distribution of per-frame flow-loss changes; its text gives mean and relative flow-loss change, generated-output MAE/MSE/cosine, mean latency, and peak allocated VRAM.

The experiment uses 100 evenly spaced full 30-step windows, distributed across the archive's nine episodes. Frames within an episode are correlated; inspect the per-episode means in each run's `summary.json` when judging whether a mean change is widespread. Latency and GPU memory are comparable only for runs made on the same GPU and software environment. Peak allocation covers the timed inference loop after warmup; it is not the size of the checkpoint file or CUDA's reserved-memory total.
