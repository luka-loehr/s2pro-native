# s2pro-native

Complete native C serving stack for **Fish Audio S2-Pro** (OpenAudio S2) on
NVIDIA DGX Spark (GB10, sm_121). No Python, no PyTorch, no SGLang at runtime —
C11 host code, CUDA C kernels, cuBLAS BF16 GEMM by default with an opt-in
FP8 path through [fish-scales-ops](https://github.com/fishaudio/fish-scales-ops)
(block-scaled FP8, verified on this GPU at cos 0.9993 vs BF16).

```
text ──► tokenizer + ChatML prompt (C) ──► Slow-AR 36L/2560 (CUDA) ─┐
                                            Fast-AR 4L residuals ────┤ 10 codes/frame
                                                                     ▼
                        HTTP chunked stream ◄── scheduler ◄── DAC vocoder (CUDA) ── 44.1 kHz PCM
```

- `include/s2pro/` — frozen module contracts (see `CONTRACT.md`)
- `src/core` tensors, safetensors mmap loader, config, JSON
- `src/slowar` 36-layer backbone: GQA + qk-norm + RoPE(1e6) + SwiGLU, KV cache,
  two-softmax sampling, lockstep batch decode
- `src/fastar` + `src/text` 4-layer residual decoder, BPE tokenizer, prompt builder
- `src/dac` modded-DAC / Firefly-GAN vocoder + encoder (voice cloning)
- `src/sched` + `src/http` lockstep scheduler, dependency-free HTTP server
- `src/fso` extern-C shim over fish-scales-ops FP8 kernels
- `docs/PORTING.md` — authoritative porting spec (dims, sampling, pitfalls)
- `docs/SPARK.md` — build/run on the box, proven flags, FSO integration

Model dims (verified against checkpoint): slow-AR 36L, dim 2560, 32Q/8KV
heads, head_dim 128, FFN 9728, vocab 155776, qk-norm, tied head; fast-AR 4L,
10 codebooks x 4096, untied head, no qk-norm; frame = 2048 samples @ 44.1 kHz.

Weights are not in git. Tokenizer/config are under `model/`.
