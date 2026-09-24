# Psi-0 Quantization tests 

W8A8 FP8-E4M3

Levels
- all BF16 - baseline
- language transformer FP8; vision + action expert BF16
- full Qwen3-VL FP8; action expert BF16
- full Qwen3-VL + action expert FP8
- all eligible large Linear/GEMM layers quantized; norms, softmax, embeddings, and other numerically sensitive ops remain BF16/FP32

Values to measure
- validation / flow-matching loss and Δ vs BF16
- action-output error vs BF16: MSE/MAE, cosine similarity
- task success rate / episode reward
- peak VRAM
- inference latency and throughput
- maximum batch size / number of parallel environments that fit
