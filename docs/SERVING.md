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

## 6. The concurrency build-out (chunked prefill, graphs to 16, batched DAC)

All four levers were then built, each individually gated:

1. **Chunked prefill**: a joining request prefills in
   `S2P_PREFILL_CHUNK`-id slices (default 96), at most one slice per
   tick round-robin, VQ runs never split; the cacheable prefix
   snapshots the moment it is resident.
2. **Captured decode graphs for every batch size** up to the session
   cap (`S2P_MAX_SESSIONS` 8 → 16). This removed the measured B = 6
   queueing cliff (2.49 → 1.17). A second wall at B = 10 (worst 4.19)
   turned out to be `S2P_INT8_GEMV_MAX_M`: beyond M = 8 every decode
   linear fell back to the full-weight dequant+GEMM path. Raising the
   cap to 16 with dual kernel instantiation (MM = 8 preserves the
   single-stream register budget exactly; MM = 16 serves 9…16) fixed
   B = 10 to 1.99 with single-stream unchanged.
3. **Pre-warmed voice prefixes**: every registry voice's system block
   prefills once at startup (`s2p_sched_warm_voices`, cache capacity
   raised to the roster size); first requests never pay the reference
   prefill.
4. **Cross-session batched DAC** (`S2P_DAC_BATCH`, default on): one
   stage-walk decodes a frame group for all streams. The weight-bearing
   kernels (convs, transposed convs, depthwise, transformer matmuls)
   and — after review condition C2 — the element-wise/norm stages run
   batched over per-session pointer tables with the session index as
   the fastest-varying grid coordinate; the temporal accumulation
   (8 frames per flush) composes, so each weight tile is read from DRAM
   once per 8 frames for ALL streams: 745 MB × 344 frame-decodes/s =
   256 GB/s unbatched (beyond the ~220 GB/s bus — the original
   saturation) vs ~2 GB/s batched, a 128× reduction. Pointer tables
   upload only on roster change (condition C3). Gate: with two
   concurrent fixed-seed streams, batch-off and batch-on produce
   byte-identical WAVs per stream; single-stream serving is unchanged
   (0.60 / 0.58, TTFA 0.14 / 0.20 s). An external design review
   (batched-inference research pass) approved the architecture —
   block-mapping choice confirmed, L2-persistence hints explicitly
   rejected as counterproductive here — with CUDA-graph capture of the
   DAC stage walk (D1) and an Nsight `dram__bytes` sweep (C4) recorded
   as the formal follow-ups.

Measured per-stream wall RTF after the build-out (~27 s voiced takes,
distinct voices, warm prefixes):

| B | before | after |
| ---: | ---: | ---: |
| 2 | 0.69 | 0.70 |
| 4 | 0.99 | 0.91 |
| 6 | 2.49 | 1.16 |
| 8 | 3.61 | 1.45 |
| 10 | — | 1.99 |
| 12 | — | 2.32 |
| 16 | — | 2.95 |

**Where the wall lives now.** The DAC batching changes none of these
numbers (batch-off and batch-on walls are identical at every B) — the
per-stream marginal cost of ~6.3 ms is not vocoder work. It matches the
backbone's per-stream KV read: ~1450 prefix tokens × 74 KB (INT8 KV)
× 16 streams ≈ 1.7 GB per tick ≈ 7.7 ms at bus speed. Concurrency is
now KV-bandwidth-bound; the cap under the per-stream RTF < 1 rule is
**5** (B = 4 at 0.91 with margin, B = 6 at 1.16 just over). The next
structural levers are KV-side: shorter effective contexts, sub-8-bit
KV, or attention windowing — plus the reviewed D1 (DAC graph capture)
for launch-gap economy.
