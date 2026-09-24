# Ψ₀ BF16 reference for FP8 comparisons

The first baseline records Ψ₀'s action outputs, parameter dtypes, latency, and peak GPU memory. Later FP8 runs should use the **same checkpoint, input, sampling steps, and seeds** and compare their actions with `actions.jsonl`. This is a numerical comparison on a fixed synthetic input; it does not measure validation loss or robot task success.

## Run

1. Clone and set up the [official Ψ₀ repository](https://github.com/physical-superintelligence-lab/Psi0) with its `psi` environment and a CUDA GPU.
2. Edit [baseline_config.json](baseline_config.json). Set `psi_repo` to the checkout path. The other values select the released SONIC fine-tune and sampling settings.
3. From the Ψ₀ checkout, download the checkpoint, then run the baseline:

```bash
cd /path/to/Psi0
bash /path/to/qpsi/baseline_setup.sh
uv run --active --group psi python /path/to/qpsi/baseline.py
```

`baseline_setup.sh` downloads the selected [SONIC checkpoint](https://huggingface.co/USC-PSI-Lab/psi-model/tree/main/psi0/sonic-checkpoints/multi-task.psi-dream.2609092156) into `Psi0/cache/checkpoints/`. The download uses `uvx --from huggingface_hub hf download` and is about 11 GB. The release includes the combined model weights, Ψ₀ configuration, and cached CLIP instruction embeddings. No training run or local dataset is needed for this first reference. `baseline.py` only loads the checkpoint and measures BF16 inference.

The script prints model parameter counts by dtype, warms inference once, then writes these files into `baseline_results/` beside the script:

- `actions.jsonl`: one normalized action chunk per sampling seed.
- `summary.json`: checkpoint identity, exact synthetic input, seeds, action shape, mean latency, throughput, and peak CUDA memory.

The input is a constant gray 480×270 image, zero state vector, and one instruction from the released CLIP cache. This gives reproducible BF16 action outputs for an initial FP8 error measurement. For accuracy on real tasks or validation flow loss, a representative SONIC dataset and a separate evaluation path are still required.

## Config fields

| Field | Use |
| --- | --- |
| `psi_repo` | Local Ψ₀ checkout. Edit this first. |
| `checkpoint` | Released SONIC checkpoint; keep fixed across BF16 and FP8. The script uses its step 40000. |
| `seed` / `samples` | Initial seed and number of repeatable sampling runs. |
| `inference_steps` | Flow sampling steps passed to Ψ₀'s `predict_action()`. |
