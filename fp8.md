# Ψ₀ FP8 comparisons

The reference is the released, unquantized Ψ₀ + SONIC checkpoint. Its Qwen3-VL weights are BF16; its action expert weights are FP32. Inference runs under BF16 autocast. Call it the **unquantized reference**, rather than an all-BF16 baseline.

## Measures for every precision variant

- **Offline quality:** flow matching velocity MSE on the shared 78 body/hand dimensions of the public UniFolM SONIC validation set. Reuse the saved noise, timesteps, frame IDs, CLIP vectors, and checkpoint normalization for every variant. Neck dimensions have no targets and are masked. This is a cross-domain loss measure, not robot task success or the authors' Psi-Dream validation score.
- **Output fidelity:** action-output MAE, MSE, and cosine similarity against the released configuration, with identical observations, inference steps, and paired random seeds.
- **Resources:** stored weight dtypes by component, inference latency and throughput, and peak GPU memory. Measure all variants on the same GPU and software environment.

## First comparison

Quantize eligible large linear layers with W8A8 FP8 E4M3. Keep normalization, softmax, embeddings, and other sensitive operations at their existing precision. Start with the language transformer in FP8, leaving the vision tower at BF16 and the action expert at FP32. Expand to the vision tower, then optionally to the action expert, one change at a time.

Run `baseline.py` first on the pinned UniFolM validation pack and retain `baseline_results/`. Keep the released BF16-VLM/FP32-expert configuration as the reference. Treat a BF16 action expert as a separate precision variant when comparing FP32, BF16, and FP8 in that component.

For the BF16 action expert, change only `action_head_dtype` in `baseline_config.json` to `bf16` and rerun `baseline.py`. The script stores its paired results in `bf16_action_head_results/`.

Add task success or episode reward when an evaluation environment is available.
