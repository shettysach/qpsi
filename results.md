| Run               | Flow MSE | Δ flow       | Δ %     | Output MAE | Output MSE  | Cosine   | Latency ms | Peak GiB |
| ----------------- | -------- | ------------ | ------- | ---------- | ----------- | -------- | ---------- | -------- |
| vlm_bf16_act_fp32 | 0.60377  | +0           | +0.000% | 0          | 0           | 1.000000 | 138.48     | 7.85     |
| vlm_bf16_act_bf16 | 0.604115 | +0.000345204 | +0.057% | 0.00191662 | 7.4501e-06  | 0.999979 | 135.78     | 5.33     |
| vlm_bf16_act_fp8  | 0.659748 | +0.0559774   | +9.271% | 0.0194405  | 0.000675148 | 0.998099 | 340.02     | 4.74     |
| vlm_fp8_act_fp32  | 0.603427 | -0.0003434   | -0.057% | 0.00862696 | 0.00271511  | 0.992796 | 171.12     | 6.17     |
| vlm_fp8_act_fp8   | 0.659203 | +0.0554326   | +9.181% | 0.0224142  | 0.00321562  | 0.991398 | 385.34     | 3.07     |

## FP16 validation on balerion

The FP16 variants did not produce finite predictions on sample 0. They therefore have no valid flow loss, action comparison, or performance result to add to the table.

| Run | First failing stage | Nonfinite values |
| --- | --- | ---: |
| `vlm_bf16_act_fp16` | BF16 VLM features converted to FP16 for the action expert | 24 / 3,710,976 |
| `vlm_fp16_act_fp32` | FP16 VLM features | 3,401,730 / 3,710,976 |
| `vlm_fp16_act_bf16` | FP16 VLM features | 3,401,730 / 3,710,976 |
| `vlm_fp16_act_fp16` | FP16 VLM features | 3,401,730 / 3,710,976 |

The BF16 VLM features are finite before conversion, so the 24 failures in `vlm_bf16_act_fp16` exceed FP16's maximum finite magnitude of 65,504. The action expert is not reached in that run. The FP16 VLM runs fail within the VLM, before action expert precision matters. These outcomes describe the unmodified precision variants; clipping or rescaling would define a separate experiment.

`vlm_bf16_act_fp16_bf16proj` is a follow-up hybrid variant that keeps the VLM feature projection in BF16 and runs the remaining action expert in FP16. It has not been measured on balerion yet.
