# Performance and optimization record

## 1. Results

Measured on one DGX Spark (GB10, `sm_121`), single stream, streamed
serving over HTTP, sampled (temperature 0.8, top-p 0.8, fixed seed). RTF
is compute ÷ audio; below 1.0 means synthesis outruns playback.

| serving path | wall RTF | TTFA |
| --- | ---: | ---: |
| zero-shot | **0.51** | 0.14 s |
| 60 s voice reference (warm voice cache) | **0.52** | 0.20 s |
| ~143 s chunked long-form | **0.50** | 0.20 s |

Configuration: all-INT4 weight stream (packed group-wise INT4 backbone +
QAT-distilled INT4 fast-AR and lm-head, `S2P_INT4_ALL=1` on a
QAT-patched checkpoint), accepted by critical listening on 2026-08-03.
An unpatched checkpoint serves the INT8-fast-AR configuration at
0.75–0.76. Starting point of the same stack: wall RTF 2.05. The measured
path from 2.05 to 0.51, including the negative results, is recorded in
the technical reports (§3).

BF16, INT8, and the packed group-wise INT4 backbone pass the layer-parity
gate against the PyTorch reference; voice cloning, the multilingual voice
registry, and the HTTP streaming server are exercised end to end on real
hardware.

## 2. Why weight precision dominates

At batch 1 every frame reads ~7.75 B weight parameters (backbone once,
tied lm-head once, nine sequential fast-AR passes), so decode is
bandwidth-bound and 16-bit weights sit above realtime on this memory
system by construction (~220 GB/s sustained). The full analysis,
per-scheme measurements, and ablations are in [QUANT.md](QUANT.md).

Engine-level decode (single stream, 67-token prompt, 60 frames):

| GEMM path | Prefill | Decode per frame | RTF (compute ÷ audio) |
| --- | ---: | ---: | ---: |
| BF16 cuBLAS (default) | 306.6 ms | 93.3 ms | 2.41 |
| INT8 weight-only (`S2P_INT8=1`) | 353.4 ms | **39.7 ms** | **1.27** |

Serving configuration (`S2P_INT8=1 S2P_INT4=1 S2P_INT4_ALL=1`,
QAT-patched checkpoint): packed group-wise INT4 backbone (4.5 bits per
weight, bit-identical to the unpacked container), QAT-distilled INT4
fast-AR and lm-head (untrained 4-bit collapses the fast-AR's argmax
cascade; the distillation is what makes this tier possible), optimized
DAC convolution kernels (2.1 ms/frame, PCM identical by MD5), per-voice
KV-prefix cache (~189 MB per voice, LRU). Result: the wall-RTF table in
§1. Process memory: ~8.5 GB (INT8) vs ~12 GB (BF16); backbone weights
2.04 GB packed.

Numeric fidelity is gated by the layer-parity protocol
([benchmarks/parity](../benchmarks/parity/README.md)): BF16 and INT8 pass
at every stage (backbone cos ≥ 0.99989, prefill/step-1 argmax identical,
native DAC at SNR 65.9 dB); the packed INT4 backbone holds the same
argmax class; FP8 fails (backbone cos collapses to 0.33 over 36 layers)
and is retained for memory-system measurements only.

## 3. Optimization record

Each step shipped only after its gate (parity, MD5 comparison, or the
audio battery); the reports hold the details, including what did not
work.

| step | measured effect |
| --- | --- |
| layer-parity validation vs the PyTorch reference | BF16 PASS, INT8 PASS, FP8 FAIL |
| INT8 per-channel weight-only GEMV | decode 93.3 → 39.7 ms/frame; memory ~12 → ~8.5 GB |
| DAC on a dedicated CUDA stream + batched pushes | wall RTF 2.05 → 1.19 (51 s reference) |
| split-K flash-decode attention | long-context decode 46.9 → 42.7 ms/frame |
| device-side sampling (exact two-softmax port) | removes the 623 KB per-frame logits round-trip |
| CUDA-graph replay of the decode tick | 42.4 → 39.6 ms/frame; graph vs eager byte-identical |
| incremental streaming DAC | streamed PCM matches the whole-buffer decode (MD5) |
| group-wise INT4 backbone (g32 + MSE clip search) | INT8-class parity decisions; fixes naive INT4's audible failure |
| packed nibbles + f16 group scales (4.5 bpw) | zero-shot wall RTF 1.10 → 0.95, identical output to unpacked |
| sentence chunking + join normalization | long-form prosody holds through 130+ s (was flattening at ~40 s) |
| per-voice KV-prefix cache | voice-ref TTFA 1.58 → 0.23 s; every path below realtime |
| optimized DAC conv kernels | DAC 15.1 → 2.1 ms/frame; wall RTF 0.75–0.76 |
| QAT self-distillation of the fast-AR → all-INT4 | weight stream 6.17 → 4.37 GB/frame; wall RTF 0.60–0.63; accepted by listening |
| prequantized-weight sidecar cache | server start 27.7 → 5.1 s, identical weights by construction |
| INT8 KV cache (g32 per head vector) | KV memory/traffic halved; parity holds the INT8 class exactly |
| INT8 embedding lookups (bf16 table dropped) | −0.8 GB; prefill/step-1 argmax unchanged |
| FP16 vocoder weights | codec memory 1.6 → 0.75 GB; 68.1 dB vs the f32 decode |
| fast-AR launch fusion | rejected: graphs already amortize launches ([SERVING.md](SERVING.md)) |

Every module now serves at 4 bits (backbone by construction, fast-AR and
lm-head by self-distillation; runs, gates, and telemetry in
[QAT-RUNS.md](QAT-RUNS.md)), the KV cache and embeddings at 8, the
vocoder at 16. Serving infrastructure and concurrency behavior (4 streams
under the per-stream RTF < 1 rule; the levers for more) are measured in
[SERVING.md](SERVING.md). Possible further work: cross-session DAC
batching, chunked prefill, a norm+GEMV fused kernel, and further
distillation rounds if listening demands them.
