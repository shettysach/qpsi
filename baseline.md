# Ψ₀ BF16 reference for FP8 comparisons

The first baseline records Ψ₀'s action outputs, parameter dtypes, latency, and peak GPU memory. Later FP8 runs should use the **same checkpoint, input, sampling steps, and seeds** and compare their actions with `actions.jsonl`. This is a numerical comparison on a fixed synthetic input; it does not measure validation loss or robot task success.

## Set up one uv environment

Keep the [official Ψ₀ checkout](https://github.com/physical-superintelligence-lab/Psi0) beside this project. Edit [baseline_config.json](baseline_config.json) and set `psi_repo` to its absolute path. The checkout supplies model code; its `.venv`, dependency groups, and `uv.lock` are not used by this baseline.

On the RTX 5090 machine, run:

```bash
cd /path/to/qpsi
nvidia-smi
uv sync --python 3.11
./.venv/bin/python -c 'import sys, torch; print("Python:", sys.executable); print("PyTorch:", torch.__version__, torch.__file__); print("CUDA:", torch.version.cuda); print("GPU:", torch.cuda.get_device_name(0)); print("architectures:", torch.cuda.get_arch_list()); print("CUDA kernel:", torch.zeros(1, device="cuda"))'
```

`uv sync` reads this project's [pyproject.toml](pyproject.toml), creates or updates `qpsi/.venv`, and records resolved dependencies in `qpsi/uv.lock`. The project file pins direct inference dependencies and directs Torch and torchvision to the CUDA 12.8 wheel index. No activation is needed. The next command prints the interpreter, wheel, CUDA version, GPU, and supported architectures, then actually runs `torch.zeros` on CUDA. Stop here if that check fails; the checkpoint is not needed to diagnose a Torch kernel error. The Ψ₀ source selects PyTorch SDPA when `flash-attn` is absent, so this first environment does not compile `flash-attn`.

The [Ψ₀ troubleshooting guide](https://github.com/physical-superintelligence-lab/Psi0#troubleshootings) recommends CUDA 12.8 PyTorch for `sm_120`; [uv supports selecting that backend directly](https://docs.astral.sh/uv/guides/integration/pytorch/). CUDA 12.8 requires a sufficiently recent NVIDIA driver; [NVIDIA lists 570.26 or newer for Linux CUDA 12.8 GA](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/). If `nvidia-smi` reports an older driver or the CUDA allocation fails despite the `+cu128` wheel, record the full output before changing packages.

## Run the baseline

After the environment check passes:

```bash
cd /path/to/qpsi
bash baseline_setup.sh
./.venv/bin/python baseline.py
```

Use `./.venv/bin/python` for later BF16 and FP8 measurements. This avoids accidentally running the Python in `Psi0/.venv` or another activated environment.

`baseline_setup.sh` downloads the selected [SONIC checkpoint](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156) into `Psi0/cache/checkpoints/`. The download uses `uvx --from huggingface_hub hf download` and is about 11 GB. The release includes the combined model weights, Ψ₀ configuration, and cached CLIP instruction embeddings. No training run or local dataset is needed for this first reference. `baseline.py` only loads the checkpoint and measures BF16 inference.

The script prints model parameter counts by dtype, warms inference once, then writes these files into `baseline_results/` beside the script:

- `actions.jsonl`: one normalized action chunk per sampling seed.
- `summary.json`: checkpoint identity, exact synthetic input, seeds, action shape, mean latency, throughput, and peak CUDA memory.

The input is a constant gray 480×270 image, zero state vector, and one instruction from the released CLIP cache. This gives reproducible BF16 action outputs for an initial FP8 error measurement. For accuracy on real tasks or validation flow loss, a representative SONIC dataset and a separate evaluation path are still required.

## Read the results

`summary.json` contains the values to place alongside each FP8 run:

| Value | Meaning |
| --- | --- |
| `parameter_counts_by_dtype` | Actual stored parameter types in the VLM and action expert. “BF16 reference” describes the official unquantized inference path; some parameters may be FP32. |
| `mean_latency_seconds` | Mean time for one action chunk after a warmup call. With five samples this is a quick estimate. |
| `throughput_samples_per_second` | Number of action chunks per second, computed from the measured call times. |
| `peak_vram_bytes` | Peak PyTorch CUDA allocation during the measured calls, including loaded model weights. |
| `action_shape` | Expected shape of each saved action chunk. |

`actions.jsonl` contains one record per seed. To compare an FP8 run, pair records by `sample_id` and `seed`, then calculate action MAE, MSE, and cosine similarity between FP8 and BF16 arrays. Compare latency and peak VRAM on the same GPU with the same `samples`, `seed`, `inference_steps`, image, state, and instruction. Lower action error means the quantized output stayed closer to this reference; lower latency or VRAM means a resource improvement.

This first run answers whether the released checkpoint loads, what dtypes it actually uses, and how FP8 changes its outputs and resource use on one fixed input. Five draws of that input are a smoke measurement. They do not establish model accuracy, validation loss, episode reward, or representative performance across tasks; those require real SONIC observations and a separate evaluation set.

## Config fields

| Field | Use |
| --- | --- |
| `psi_repo` | Local Ψ₀ checkout. Edit this first. |
| `checkpoint` | Released SONIC checkpoint; keep fixed across BF16 and FP8. The script uses its step 40000. |
| `seed` / `samples` | Initial seed and number of repeatable sampling runs. |
| `inference_steps` | Flow sampling steps passed to Ψ₀'s `predict_action()`. |
