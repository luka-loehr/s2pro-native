# DAC conv kernels: 15.1 → 2.1 ms/frame, bit-identical

Engineering log of the vocoder conv optimization on the GB10 (`sm_121`),
including the attempts that lost. Every step was gated on a byte-identical
(MD5) deterministic smoke take — the per-output floating-point
accumulation order is never changed, so the optimized kernels produce the
same PCM bit for bit.

## Measure first: the built-in profiler

`S2P_DAC_PROF=1` brackets every conv launcher with CUDA events;
`s2pdk_prof_dump()` prints per-type and per-launch totals with shapes
(`src/dac/decoder.cu`). Baseline, 120-frame whole-buffer decode:

| kernel | time | share |
| --- | ---: | ---: |
| conv1d (k=7 dilated + k=1 pointwise) | 1191 ms | 66 % |
| tconv (upsampling ConvTranspose1d) | 519 ms | 29 % |
| dwconv + elementwise + attention | ~100 ms | 5 % |

The decisive observation: a 192×192 **pointwise** conv at t=122880
measured 81 ms — that is ~18 GB of traffic where 0.2 GB is inherent,
because the one-block-row-per-output-channel layout re-reads the entire
input plane once per output channel and the plane (94 MB) far exceeds L2.
The arithmetic (81 ms ≈ 18 GB ÷ ~220 GB/s) matched exactly.

## What lost, and why (kept in the record deliberately)

1. **Register co-tiling alone** (one thread accumulates 16 output
   channels): k=1 convs 4× faster (81 → 20 ms), but k=7 convs 1.6×
   SLOWER (92 → 147 ms) and tconvs 4× slower (198 → 813 ms). Cause: the
   per-block weight working set grows to CO_TILE × cin × k × 4 B (344 KB
   at 768ch k=7, 1.5 MB for the first tconv) and thrashes L1.
2. **Co-tiling + smem weight staging for conv k=7**: no better (146 ms).
   The kernel is not input-bandwidth-bound at all once co-tiled — it is
   LSU-throughput-bound (one smem read per FMA with a single time
   position per thread).
3. **Co-tiling + per-tap smem staging for tconv**: worse still (850 ms).
   With `stride`-strided validity, only 1/stride of the lanes do work
   per tap while every lane pays the barriers.

## What won

1. **`k_conv1d_wide`** (all k ≤ 7 including pointwise): 2D register
   blocking — each thread accumulates 2 time positions × 8 output
   channels (16 accumulators), weights stream through smem in ci-chunks
   of 64, inputs ride L1 (adjacent threads read adjacent taps; measured
   effectively free on the reference kernel). Each smem weight read now
   feeds two FMAs and each input read eight. conv1d total
   **1191 → 168 ms**.
2. **`k_tconv1d` phase-partitioned**: all outputs with equal
   `to % stride` share the same valid tap set, so a block serves ONE
   phase — no masked lanes, uniform control flow, and `ti = j − m`
   coalesces across threads. Combined with the same 2-position register
   blocking and 8 KB smem weight tiles: tconv **519 → 28.6 ms**.
3. Out-of-range taps contribute an explicit `0.0f` term instead of a
   skip. IEEE-754 makes `acc + 0·w` an identity for finite weights up to
   the sign of zero; the MD5 gate confirms no representable difference
   on real audio.

Per-output accumulation order everywhere: conv `ci`-outer/`kk`-inner,
tconv `kk`-outer/`ci`-inner, chunks ascending — identical to the
reference kernels, which remain in the file as the bit-exactness anchor
and for encoder shapes (k=16) outside the tile sizing.

## Result

| | before | after |
| --- | ---: | ---: |
| whole-buffer DAC | 15.1 ms/frame | **2.1 ms/frame** |
| server wall RTF zero-shot | 0.95 | **0.76** |
| server wall RTF, 60 s voice reference | 0.94 | **0.76** |
| server wall RTF, ~147 s chunked long-form | 0.92 | **0.75** |

The DAC has moved from second-largest cost to noise; the frame budget is
now backbone-dominated, which is what the quantization ladder
(docs/QUANT.md) addresses.
