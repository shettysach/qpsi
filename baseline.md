# Ψ₀/SONIC unquantized validation reference

Run the released `multi-task.psi-dream.2609092156` checkpoint on fixed examples from PSI's public [UniFolM SONIC validation archive](https://huggingface.co/datasets/USC-PSI-Lab/psi-data/blob/main/sonic/unifolm_sonic_lerobot_val.zip). Save its action chunks, flow matching loss, latency, and peak GPU allocation for later paired FP8 comparisons. This is an **unquantized reference**, not an all-BF16 model: the VLM weights are BF16, the action expert weights are FP32, and `predict_action()` uses BF16 autocast.

The public archive has 9 episodes and 4,688 frames. It includes 64 SONIC 1.0 body-token values and 14 hand actions per frame. The fine-tuned checkpoint predicts those 78 values plus 2 neck values. UniFolM has no neck targets, so this evaluation masks the 2 neck values when computing flow loss. Its images also come from a different camera domain than the Psi-Dream fine-tuning data. The result measures quantization degradation on **UniFolM SONIC**, not the authors' unavailable Psi-Dream validation loss or robot task success.

## Set up the single uv environment

Keep the [official Ψ₀ checkout](https://github.com/physical-superintelligence-lab/Psi0) beside this project. In [baseline_config.json](baseline_config.json), set `psi_repo` to its absolute path. The dataset path is relative to that checkout. All baseline settings live in that JSON file; the checkpoint step and published dataset revision are fixed in the scripts.

On the RTX 5090 machine:

```bash
cd /path/to/qpsi
uv sync --python 3.11
./.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), torch.cuda.get_arch_list()); print(torch.zeros(1, device="cuda"))'
bash baseline_setup.sh
./.venv/bin/python baseline.py
```

`uv sync` uses this project's [pyproject.toml](pyproject.toml) and CUDA 12.8 Torch wheel source. `baseline_setup.sh` pins the checkpoint to model commit `4c6f9776fc5b18d87945254175e38bb74b9d7748` and the public UniFolM validation archive to data commit `e78fb93cc28912a3031a10b8656d32d7f0a2b867`, then extracts the archive under `Psi0/.data/`. If Hugging Face requests authentication, run `uvx hf auth login` and paste your token into the CLI. The script verifies the archive's SHA256 before evaluation. The checkpoint is about 11 GB; the validation archive is 33.5 MB. Keep sufficient disk space for the checkpoint, uv environment, and model caches.

The first run may download `openai/clip-vit-large-patch14` to compute frozen CLIP projections for UniFolM instructions absent from the checkpoint's cache. The run saves the exact projections used, so later precision variants can reuse them.

## What the script measures

It selects `samples` full 30-step windows, distributing them as evenly as possible across the nine episodes and spacing frames evenly within each episode. For each selected frame it uses the recorded egocentric image, 43-value state, and lowercased instruction. It applies the checkpoint's saved image resize/crop and **checkpoint normalization bounds**; it pads the state to 45. The recorded action target is ordered as body token 64, hands 14, then two neutral neck inputs. The neck values are excluded from loss. No validation augmentation or random state jitter is applied.

For output fidelity, the script seeds each example with `seed + sample_id`, runs ten inference steps by default, and saves the complete normalized 30×80 predicted action chunk. For quality, it samples one fixed Gaussian noise tensor and flow timestep per example, computes the checkpoint's velocity target `noise − normalized_action`, and evaluates mean squared velocity error on the shared 78 dimensions. It reports body and hand loss separately. The saved noise, timesteps, frame IDs, and CLIP vectors make the next variant's evaluation paired with this reference.

The result is **flow matching loss on UniFolM observations**. It is not action MSE against a single demonstrated trajectory, and its absolute value is not the checkpoint's original fine-tuning validation score. Compare an FP8 variant by the change in this loss under the same saved inputs, noise, and timesteps. Also compare its full 80-value predicted actions against this reference using MAE, MSE, cosine similarity, and per-dimension error. Compare latency and peak allocated VRAM only on the same GPU and software environment.

## Saved files

Everything is written to `baseline_results/` next to `baseline.py`:

| File | Contents |
| --- | --- |
| `summary.json` | Checkpoint and dataset identity, parameter dtypes, shared-action flow loss, per-episode loss, timing, throughput, peak inference allocation, and software/GPU information. |
| `actions.jsonl` | One normalized 30×80 action chunk per selected frame, with episode/frame ID and inference seed. |
| `flow_losses.jsonl` | Shared 78-value, body 64-value, and hand 14-value velocity MSE for each selected frame. |
| `flow_noise.npy`, `flow_sigmas.npy` | Fixed noise and flow timesteps for the paired quantized run. |
| `pooled_projections.pt` | Exact frozen CLIP instruction vectors used by this run. |

The `samples` default is 100, or roughly 11 windows per episode. Windows from one episode are correlated; per-episode means in the summary help reveal when one scene dominates a change. This is an offline numerical comparison. A robot or simulation evaluation would be needed to measure task success.

## Run the BF16 action head variant

After `baseline_results/` exists, change only `action_head_dtype` in `baseline_config.json` from `"fp32"` to `"bf16"`, then run:

```bash
./.venv/bin/python baseline.py
```

The script loads the same released checkpoint and casts `model.action_header` to BF16 **after loading**. The VLM stays as released. This changes the head's stored weights; the released run already used BF16 autocast for inference. The BF16 run saves its own outputs in `bf16_action_head_results/` and does not overwrite `baseline_results/`.

The BF16 run reads the reference's saved CLIP projections, Gaussian noise, and timesteps. It checks the checkpoint, dataset, sample count, seed, inference steps, PyTorch/CUDA versions, and GPU before starting. Its `summary.json` includes `comparison_to_unquantized`: paired action MAE/MSE/cosine similarity, change in shared-78 flow loss, change in mean latency, and change in peak allocated VRAM. A positive flow-loss change means the BF16 head has greater velocity-prediction error on this set.

## View the comparison

Once both result folders exist, install the plotting dependency through the same project environment and generate the figure:

```bash
uv sync --python 3.11
./.venv/bin/python compare_results.py
```

Open `comparison_results/comparison.png`. The top row shows paired flow loss for every frame, mean flow-loss change by episode, and action-output MAE for each of the 80 dimensions. In the scatter plot, points above the diagonal mean BF16 had higher loss on that frame. The bottom row compares mean flow loss, mean action-chunk latency, and peak inference GPU allocation. The script also writes `comparison.json` with exact numbers and `per_sample.csv` for deeper analysis; it checks that the two runs used the same checkpoint, dataset, samples, GPU, noise, and timesteps.

For a quick text view without plotting, run `jq '.comparison_to_unquantized' bf16_action_head_results/summary.json`. Lower flow loss is better on this offline set; lower action MAE means the BF16 head stayed closer to the released output. A small action MAE alone does not establish task success. Latency differences should be read alongside the GPU and software details in both summaries.

## Config fields

| Field | Meaning |
| --- | --- |
| `psi_repo` | Absolute path to the Ψ₀ checkout. |
| `checkpoint` | Released SONIC checkpoint, held fixed across all precision variants. |
| `validation_dataset` | Extracted UniFolM validation directory, relative to `psi_repo` or absolute. |
| `action_head_dtype` | `fp32` for the released reference; `bf16` for the isolated BF16 action head variant. |
| `seed` | Base seed for paired inference and saved flow noise. |
| `samples` | Number of fixed validation frames with complete 30-step action windows. |
| `inference_steps` | Euler flow sampling steps for action generation; separate from the single-step flow loss. |
