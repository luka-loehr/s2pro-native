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

## 9. DAC CUDA-graph capture: resolved by measurement, not built

The design review's largest open item (D1) was capturing the vocoder
stage walk as a CUDA graph, on the strength of a comparable production
system that measured 1.6–2.2× from it. That system was launch-bound:
its streaming codec decode sat flat at ~66 ms regardless of frame count,
the signature of overhead dominating work.

Ours is not, and the profiler says so directly: a whole-buffer decode
issues **34 weight-bearing conv/tconv launches** total, and the
streaming path walks the same structure once per push — where a push
carries 8 accumulated frames for every active stream. Counting the
element-wise and history stages generously (~90 launches per push, ~300
at B = 12 where stateful ops still loop per session) at ~2.1 µs of
launch latency gives 0.2–0.6 ms per push against 371 ms of audio per
stream per push: **under 0.2 % of the budget.** Graph capture would also
force the per-frame position and history length device-side, since both
are baked into kernel arguments today.

The item is therefore closed as *not worth building in this
configuration* — the temporal accumulation and cross-session batching
that §6 added are exactly what removed the condition graphs would fix.
It would return as a live question if the accumulation were dropped for
latency (§6's F-knob discussion).

## 11. Tensor-core GEMM: the arithmetic served faster, the crossover measured

With bit-identical reorganizations exhausted (§10), the batch cost —
the O(M) FMA work — moved to the matrix units: a WMMA kernel
(m16n16k16, f32 accumulators) with the INT4 weights dequantized
in-register per tile (`S2P_GEMV_TC`, packed path). The effective weight
enters as bf16(w4 · group scale) — exactly the prefill dequant+cuBLAS
precision — so this path deliberately trades the MD5 gate for the full
parity gate, and passes it in the reference class: backbone cos min
0.9919, prefill and step-1 argmax identical, fast-AR 8/9, byte-for-byte
the metrics of the GEMV configuration.

Three versions, each fixing the measured dominant overhead of the last
(worst per-stream RTF, same protocol as §10; GEMV row repeated for
comparison):

| | B=4 | B=8 | B=12 | B=16 | M=1 probe |
| --- | ---: | ---: | ---: | ---: | ---: |
| GEMV (shipping) | 0.89 | 1.46 | 2.07 | 2.90 | 0.55 |
| v1 naive tiles | 2.06 | 2.48 | 2.94 | 3.38 | 1.80 |
| v2 + coalesced 16 B weight loads | 1.97 | 2.37 | 2.77 | 3.22 | 1.70 |
| v3 + 16 B smem row skew | 1.60 | 1.99 | 2.42 | **2.84** | 1.30 |

Two findings. First, the thesis holds: the tensor-core slope is ~0.41
worst-RTF per 4 streams against the GEMV's ~0.67 — the matrix units do
make the per-stream arithmetic cheap, and v3 crosses over at B = 16
(2.84 vs 2.90), the first configuration to beat the GEMV anywhere.
Second, the fixed per-pass cost (1.30 vs 0.55 at M = 1, after the bank
conflicts that dominated v1/v2) keeps the GEMV ahead at every practical
batch size; the remaining levers (128-wide K staging to halve barriers,
vectorized x staging) are identified but cannot change the strategic
picture: even a tensor-core path that won from B = 8 would leave B = 12
at roughly 2× real time — far from the per-stream RTF < 1 rule.

The kernel therefore ships off by default, but not as a negative
result: its flat slope is exactly the profile a **speculative-decoding
verify pass** needs, where the effective row count is sessions ×
(draft length + 1) — 12 sessions verifying 3-frame drafts is a 48-row
GEMM per linear, deep in the regime where the matrix units win. The
spec-decoding build (draft model, corpus, verify machinery) is the
companion track; this kernel is its GEMM engine.

## 13. The audit tail: five more gated landings, and what was closed

The remaining audit findings landed the same day, each gated
individually (two-stream fixed-seed MD5s unchanged unless noted):

- **Fast-AR batched launches** (P1-8): rope/append/attention ran once
  per row per layer per codebook step — 3·B·4·10 launches per frame.
  Per-(layer,row) cache-pointer tables and constant per-step position
  arrays are built when the KV slab is allocated; the loop is now three
  batched launches through the parity-proven `_ptrs` kernels.
  Bit-identical; B = 8 worst RTF 1.41 → 1.35 (`S2P_FA_LOOP=1` reverts).
- **Split-K attention tile skew** (P1-6): the K-score phase read one
  128-byte shared row per lane — a 32-way bank conflict on every
  payload read. Row stride EL+4 (word stride 33). Bit-identical;
  B = 4 0.86 → 0.85.
- **Clone-codes memo** (P2-13): long-form cloned requests re-encoded
  the reference on the worker for EVERY chunk (~1.1 s all-stream stall
  each). The worker deposits the codes into the connection once; clone
  long-form now serves at registry-voice RTF (measured 0.52).
- **Encode workspace returned** (P2-10): the DAC encode arena (GBs for
  a 60 s reference) is freed after each encode instead of persisting.
- **Dead fast-AR slab memset dropped**: every position the attention
  reads is appended in the same frame first; the reference's
  per-frame zeroing was semantically dead here. Bit-identical.
- **Tensor-core prefill** (P1-5, `S2P_PREFILL_TC=1`, DEFAULT OFF): for
  chunk-sized prefills (M ≤ 128) the WMMA kernel computes straight
  from the packed INT4 weights instead of the N·K·2 B dequant+cuBLAS
  round-trip. Parity holds the reference class (backbone cos min
  0.9926, argmax intact, fast-AR 8/9); measured TTFA 0.64 → 0.61 and
  ~4 % batch-wall at B = 8. It stays off because the summation order
  differs from the shipped path — enabling it is a numerics-class
  change reserved for a listening-signed rollout, exactly like §11.

Final concurrency series on the complete stack (same protocol as §10):
**B = 4/8/12/16 worst per-stream RTF 0.83 / 1.36 / 2.18 / 2.82.**

Two items were closed without code, with reasons on record: the decode
graph's length bucketing (P1-7) trades ≤ 1 % of tick time against an
8× multiplication of the captured-graph cache that all serving relies
on — declined at this return; and the fast-AR norm+GEMV fusion (P1-9)
is the measured negative of §3 in a new coat (same fusion class, same
bit-exactness failure mode, and CUDA graphs already amortize what it
would remove). The review's C4 sweep was attempted: GB10 exposes no
`dram__` metrics to Nsight Compute (n/a on the unified-LPDDR path);
`lts__t_bytes.sum` works and is the recorded instrument for a future
multi-session harness, and the 128× weight-traffic arithmetic of §6
continues to rest on the bandwidth model plus the timing evidence.

## 12. Sliced LM-head: the audit's largest single-stream lever

The decode head computed all 155,776 vocabulary logits every frame;
the semantic sampler reads exactly 4,096 contiguous semantic rows plus
the EOS row. Serving them from the same sidecar rows through the same
GEMV with the same accumulation into two compact buffers is
bit-identical — smoke MD5 and both two-stream fixed-seed WAVs
unchanged — and removes ~389 MB of weight stream per frame: decode
23.2 → 21.7 ms/frame, server wall RTF **0.55 → 0.51 zero-shot /
0.52 voice-ref** (same script, same day). The full-vocabulary head
remains on the parity-dump path and behind `S2P_HEAD_FULL=1`.
Found by the 43-agent full-repository audit of 2026-08-07, which also
surfaced the M > 8 dispatch bug above and a prefix-cache teardown
leak (both fixed), and whose ranked remaining levers are tracked in
the audit record.

## 10. The GEMM pipeline campaign: three bit-identical variants, all measured out

After the staged GEMV failed (§8), its two identified defects were fixed
in a proper pipeline kernel, and when that also lost, the remaining
bit-identity-preserving idea was tried too. All three share the exact
per-row arithmetic of the plain GEMV (same chunk order, same lane→k
mapping, same accumulation sequence) and were each proven bit-identical
on the two-stream fixed-seed WAV gate before being timed:

| variant | mechanism | B=4 / 8 / 12 / 16 (worst RTF) |
| --- | --- | --- |
| plain GEMV (shipping) | one warp per row, x from L2 per row | **0.89 / 1.46 / 2.07 / 2.90** |
| staged (§8) | 4-warp smem staging, barriers | 1.37 / 2.44 / 3.88 / — |
| pipelined (`S2P_GEMV_PIPE`) | 8 warps, double-buffered `cp.async` | 1.01 / 1.61 / 2.37 / 3.03 |
| register-blocked (`S2P_GEMV_ROWS2`) | 2 rows/warp, x loaded once for both | 0.95 / 1.59 / 2.35 / 3.22 |

Conditions: reconstructed load script (the box lost `/tmp` to a
reboot), ~19 s voiced takes, distinct voices, back-to-back runs on
2026-08-04; the baseline row was re-measured the same hour with the
same script, so the comparison is internally consistent. Single-stream
serving is identical in every configuration (0.54–0.55 here; M = 1
always takes the plain kernel).

The campaign's negative result is the finding. The pipelined kernel
eliminates barrier-per-chunk serialization *and* overlaps staging with
compute, and still loses; the register-blocked kernel needs no
synchronization at all, halves per-row activation traffic, and still
loses. Together they rule out the activation stream as the dominant
term: on this memory system the L2 serves the plain kernel's redundant
x reads faster than any restructuring pays for itself, and what remains
is the O(M) FMA work itself, which no bit-identical reorganization can
reduce. Within the constraint that batch output must equal single-file
output bit for bit, the plain GEMV is the measured optimum of every
class tried.

Both kernels stay in the tree behind their flags as reproduction paths.
The two honest ways past the wall are (a) a tensor-core GEMM path,
which changes the summation order and therefore trades the MD5 gate for
the full parity + listening gate, and (b) generating more audio per
tick (speculative multi-frame decoding), which shrinks the O(M) work
per second of audio instead of trying to serve it faster.

## 8. Batch GEMV with staged activations: measured out

The GEMV's O(M) activation re-read (§7) has an obvious remedy: stage the
current 512-element K chunk of every session row in shared memory once
per block, so the block's GEMV_WARPS output rows read it from there
instead of each row re-reading it from L2. Implemented as `k_gemv_ps`
(`S2P_GEMV_STAGED=1`, default off).

Result: worse at every batch size — worst-stream RTF 0.91 / 1.45 / 2.31
(unstaged) vs **1.37 / 2.44 / 3.88** (staged) at B = 4 / 8 / 12. With
only four warps per block the two barriers per chunk cost more than the
4× reduction in activation traffic saves, and the staging loads do not
overlap compute. Outputs are bit-identical either way (two-stream MD5s
unchanged), so this is purely a scheduling result: a win needs wider
blocks *and* double-buffered async copies, i.e. a real GEMM pipeline
rather than a staged GEMV. That remains the open item for concurrency.

## 7. KV-side levers (the wall's owner)

Two attacks on the per-stream KV read, both measured:

**INT4 KV: measured failure.** Halving the payload again collapses the
model (backbone cos 0.9919 → 0.0107, argmax wrong) — the outlier-channel
failure of per-token K quantization, documented with its mechanism and
the calibration-based fix in [QUANT.md §4.8](QUANT.md). INT8 is the KV
floor for this engine.

**Shorter reference prefixes: measured, then reverted.** A cap on the
registry reference (keep the last N frames of ~46.4 ms each before it
becomes the KV prefix) was implemented and measured at B = 12 with
distinct voices (~27 s takes), then removed again — the numbers below
are why. References serve at full length.

| reference | worst per-stream RTF | mean |
| --- | ---: | ---: |
| full (~1300–1450 frames) | 2.31 | 2.27 |
| 600 frames (~28 s) | 2.15 | 2.00 |
| 300 frames (~14 s) | 2.10 | 2.02 |

Single-stream serving is unaffected (0.60 / 0.57) and warm-cache TTFA
improves (0.20 → 0.17 → 0.15 s). But the concurrency gain **saturates at
about 9 %**: cutting the prefix from ~1450 to 300 frames removes roughly
2.3× of cached tokens and buys almost nothing beyond the first step.

That result falsifies the KV-bandwidth hypothesis, and the
falsification is the useful part (which is also why the cap itself was
reverted: ~9 % of a concurrency term, paid for with a change in
delivery — the same text ran 149 s at full reference, 165 s at 600
frames and 141 s at 300 — is not a trade worth carrying). If the per-stream marginal cost were KV reads, a
2.3× smaller cache would have shown up as a large drop; it did not. The
remaining candidate is the decode GEMV itself: it amortizes the *weight*
read across the batch (that is why weights stopped mattering), but its
activation loads and its arithmetic are both O(M) — each weight row is
multiplied against all M session rows, so tick time grows with the batch
even though bytes-from-DRAM do not. That is the textbook GEMV→GEMM
crossover, and the fix is a tiled GEMM path for M > 4 (weights and
activations staged in shared memory, one pass over the weight tile
serving all rows) rather than any further KV work.



**Where the wall lives now.** An earlier revision of this section
attributed the concurrency wall to backbone KV bandwidth; that claim
is corrected here, twice over. First, its arithmetic compared a
per-tick total against a per-stream marginal (a units error). Second,
the section's own reference-trimming experiment falsified it: cutting
the cached prefix 2.3× moved B = 12 by only ~9 %. The wall is the
O(M) arithmetic of the batch GEMV (§10), with the tensor-core path
(§11) as the measured-but-not-yet-winning attack on it. KV-side
levers remain real as MEMORY levers only.

A second correction: the B ≥ 9 figures originally published in §10/§11
were measured through an out-of-bounds dispatch bug (the staged-GEMV
rewrite dropped the M > 8 kernel instantiation; fixed 2026-08-07).
Re-measured on the fixed kernel WITH the sliced LM-head (§12):
B = 4/8/12/16 worst per-stream RTF **0.86 / 1.41 / 2.27 / 2.97**. The
cap under the per-stream RTF < 1 rule stays 5.
