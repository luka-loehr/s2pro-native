# Serving-Path Engineering: Startup, Launch Overhead, and Concurrency

Technical report. Hardware: NVIDIA DGX Spark (GB10, `sm_121`).
Companion to [QUANT.md](QUANT.md) (which owns the weight/KV/embedding
precision experiments); this report owns the serving infrastructure
measurements around them: startup cost, kernel-launch overhead, and
multi-stream behavior.

## 1. Scope

Three questions, each answered by measurement: (a) why did every server
start cost ~27 s and what removes it; (b) does kernel-launch overhead in
the fast-AR still matter under CUDA-graph replay; (c) how many
concurrent streams can one box serve under the rule that every stream's
wall RTF stays below 1.

## 2. Prequantized-weight sidecar cache

Load-time quantization cost BF16 staging of the full checkpoint plus the
INT4 MSE clip search on every start. `src/sched/qcache.c` serializes the
final quantized device tensors once (161 tensors, 2.28 GB for the
all-INT4 configuration) into a fingerprinted sidecar (FNV-1a over the
quantization config and the model directory's safetensors name, size,
mtime); later starts upload the payload directly. Any read error or
fingerprint mismatch silently falls back to live quantization; writes
are atomic (tmp + rename). `S2P_QCACHE=0` disables; `S2P_QCACHE_DIR`
relocates the file for read-only model mounts.

| | cold (live quant) | cache write | cache read |
| --- | ---: | ---: | ---: |
| all-INT4 configuration | 27.7 s | 36.3 s (one-time) | **5.1 s** |
| pure-INT8 configuration | 32.3 s | — | **5.0 s** |

Gate: the deterministic smoke MD5 is identical across no-cache,
cache-write, and cache-read for both configurations — same bytes in GPU
memory by construction.

## 3. Fast-AR launch fusion: a measured negative result

Hypothesis: the per-row RoPE/append/attention launch triple (3 × 4
layers × 10 passes per frame, plus per-row host loops) costs enough
launch latency to matter at batch 1. A fused kernel (one block per
(row, kv-head), the ≤ 10-deep cache in shared memory, the reference
kernel's single-tile softmax body) replaced the triple.

Result: decode 24.4 → 24.3 ms/frame — no meaningful win. CUDA-graph
replay already amortizes exactly the overhead the fusion removes, and
the fast-AR cost is GEMV weight bandwidth, which fusion cannot touch.
The fused kernel is argmax-equivalent on every first-frame residual
step but not bit-identical (cos delta ~1e-5, a rounding-order
difference not yet located), so it also fails the MD5 gate. It stays in
the tree behind `S2P_FA_FUSED=1` (default off) as the natural carrier
for a future norm+GEMV megakernel — where the remaining fast-AR margin
actually lives.

## 4. Measurement battery

Conditions: streamed serving over HTTP, otherwise idle GPU, all-INT4 QAT
checkpoint + INT8 KV + INT8 embedding lookups, prequant cache warm,
2026-08-03.

| serving path | wall RTF | TTFA |
| --- | ---: | ---: |
| zero-shot (~23 s) | **0.60** | 0.14 s |
| voice reference, warm cache (~21 s) | **0.58** | 0.20 s |
| long-form chunked, deeper-male (147.5 s) | **0.58** | — |
| long-form chunked, neutral-female (148.2 s) | **0.59** | — |
| short multilingual, 3 voices (~48 s) | **0.58–0.62** | — |

Audio gates: HF envelope stable over both long takes (content-driven
variation, no collapse); EOS probe vs the INT8 reference 0/24 flags,
mean lengths 3.5 s vs 3.4 s.

Memory (computed allocation deltas at equal function vs the pre-sprint
stack): bf16 embedding table −0.80 GB, per-session KV at ctx 4096
−0.28 GB, four cached voice prefixes −0.38 GB, vocoder weights f16
−0.85 GB — about −2.3 GB total for a single-session server with a warm
voice cache. (Host RSS is not meaningful for device allocations on the
GB10's unified memory; the deltas above are allocation sizes.)

The FP16 vocoder weights ([DAC-KERNELS.md §8](DAC-KERNELS.md)) landed
after this battery; re-measured on the complete stack, the serving
numbers hold: zero-shot 0.60 (TTFA 0.13–0.14 s), warm voice reference
0.58 (TTFA 0.20 s), long-form 0.58/0.59, short takes 0.56–0.61.

Startup after the sprint is dominated by the voice registry's DAC
encode (~33 voices × ~6 s of encode); persisting the encoded voice
codes (~1.5 MB total) is the identified next startup lever.

## 5. Concurrency under the per-stream RTF < 1 rule

The lockstep scheduler shares the weight stream across all decoding
sessions; a stream's marginal cost is its private KV reads, its DAC
decode, and — dominant in practice — its prefill stalls (each new chunk
of a chunked take re-enters prefill, which pauses the lockstep tick for
every stream). Measured with B concurrent voiced streams (~27 s takes,
distinct voices, cold caches — the worst case):

| B | worst per-stream wall RTF | mean |
| ---: | ---: | ---: |
| 2 | 0.69 | 0.69 |
| 4 | 0.99 | 0.98 |
| 6 | 2.49 (queueing: 4 session slots) | 2.37 |

The B = 6 cliff is the configured session/graph cap, not physics:
streams beyond four wait for a free slot. Under the per-stream
RTF < 1 rule the cap is **4 today (no margin) and 3 with margin**; the
identified levers for a higher cap are cross-session DAC batching,
chunked prefill (so a joining stream does not stall the tick), captured
graphs for B > 4, and warm voice caches (the B = 4 worst case includes
cold-voice prefills).
