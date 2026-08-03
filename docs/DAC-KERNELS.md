# Vocoder Convolution Kernels: 15.1 → 2.1 ms/frame, Bit-Identical

Technical report. Hardware: NVIDIA DGX Spark (GB10, `sm_121`, 48 SMs,
~220 GB/s sustained). Code: `src/dac/decoder.cu`. Every optimization step
was gated on a byte-identical (MD5) deterministic smoke take.

## 1. Scope

This report covers the optimization of the DAC/Firefly-GAN vocoder's
convolution kernels: the profiling that located the cost, the bit-exactness
constraint that shaped every design, two co-tiling designs that lost — with
their measurements and causes — and the two kernel designs that won.

## 2. Problem statement

After the backbone and fast-AR optimizations, the vocoder had become the
second-largest per-frame cost. Baseline profile of a 120-frame
whole-buffer decode (built-in profiler, §6):

| kernel class | time | share |
| --- | ---: | ---: |
| conv1d (k=7 dilated + k=1 pointwise) | 1191 ms | 66 % |
| tconv (upsampling ConvTranspose1d) | 519 ms | 29 % |
| dwconv + elementwise + attention | ~100 ms | 5 % |

The decisive observation: a 192×192 pointwise convolution at t=122880
measured 81 ms — approximately 18 GB of traffic where 0.2 GB is inherent.
The one-block-row-per-output-channel reference layout re-reads the entire
input plane once per output channel, and the plane (94 MB) far exceeds L2.
The bandwidth arithmetic matches exactly: 81 ms ≈ 18 GB ÷ ~220 GB/s.

## 3. Constraint: bit-identical output

The per-output floating-point accumulation order is never changed: conv
kernels accumulate `ci`-outer/`kk`-inner, transposed convs
`kk`-outer/`ci`-inner, chunks ascending — identical to the reference
kernels, which remain in the file as the bit-exactness anchor and for
encoder shapes (k=16) outside the tile sizing. Out-of-range taps
contribute an explicit `0.0f` term instead of a branch; IEEE-754 makes
`acc + 0·w` an identity for finite weights up to the sign of zero, and the
MD5 gate confirms no representable difference on real audio. Under this
constraint, an optimized kernel produces the same PCM bit for bit, so
quality re-qualification is unnecessary by construction.

## 4. Negative results

Recorded deliberately; each shaped the final design.

1. **Register co-tiling alone** (one thread accumulates 16 output
   channels): k=1 convs 4× faster (81 → 20 ms), but k=7 convs 1.6×
   *slower* (92 → 147 ms) and tconvs 4× slower (198 → 813 ms). Cause: the
   per-block weight working set grows to CO_TILE × cin × k × 4 B (344 KB
   at 768 channels k=7; 1.5 MB for the first tconv) and thrashes L1.
2. **Co-tiling + shared-memory weight staging, conv k=7**: no better
   (146 ms). Once co-tiled, the kernel is not input-bandwidth-bound at
   all — it is LSU-throughput-bound, paying one shared-memory read per
   FMA with a single time position per thread.
3. **Co-tiling + per-tap shared-memory staging, tconv**: worse still
   (850 ms). With stride-strided tap validity, only 1/stride of the lanes
   do work on any tap while every lane pays the barriers.

## 5. Final kernel designs

1. **`k_conv1d_wide`** (all k ≤ 7, including pointwise): two-dimensional
   register blocking — each thread accumulates 2 time positions × 8
   output channels (16 accumulators); weights stream through shared
   memory in ci-chunks of 64; inputs ride L1 (adjacent threads read
   adjacent taps, measured effectively free on the reference kernel).
   Each shared-memory weight read now feeds two FMAs and each input read
   eight. conv1d total: **1191 → 168 ms**.
2. **`k_tconv1d`, phase-partitioned**: all outputs with equal
   `to % stride` share the same valid tap set, so a block serves exactly
   one phase — no masked lanes, uniform control flow, and `ti = j − m`
   coalesces across threads. Combined with the same 2-position register
   blocking and 8 KB shared-memory weight tiles: tconv **519 → 28.6 ms**.

## 6. Instrumentation

`S2P_DAC_PROF=1` brackets every convolution launcher with CUDA events;
`s2pdk_prof_dump()` prints per-kernel-type and per-launch totals with
shapes. The profiler is permanent — the baseline table in §2 and the
result table below come from it.

## 7. Results

| | before | after |
| --- | ---: | ---: |
| whole-buffer DAC | 15.1 ms/frame | **2.1 ms/frame** |
| server wall RTF, zero-shot | 0.95 | **0.76** |
| server wall RTF, 60 s voice reference | 0.94 | **0.76** |
| server wall RTF, ~147 s chunked long-form | 0.92 | **0.75** |

The vocoder has moved from second-largest cost to noise; the frame budget
is now backbone-dominated, which is the regime the quantization ladder
([QUANT.md](QUANT.md)) addresses.

## 8. FP16 weight storage

The conv/matmul weights (745 MB of the 1.6 GB codec artifact after
weight-norm folding) store as f16; conversion happens at load, kernels
convert at the shared-memory staging (or first read), so inner-loop
arithmetic, accumulation order and activation precision are identical to
the f32 path — only the rounded weight values differ. Biases, Snake
alphas, norms, layer scales, the RVQ codebooks and projections stay f32
(1.4 MB): partly measured insurance, partly because rvq.cu reads the VQ
projections directly. `S2P_DAC_F32=1` reverts.

Gate (policy fixed before measurement: ≥ 60 dB pass, 55–60 dB escalate
to envelope + listening, < 60 dB fail): SNR of the f16 decode against
the f32 decode on identical codes over a 205-frame take = **68.1 dB** —
pass with headroom. External corroboration: production vocoders
(NVIDIA Riva, BigVGAN's fused kernels, stable-audio-tools) ship FP16 by
default; measured FP16 round-trip on DAC-class weights sits ~16 binades
under the f16 ceiling with strictly positive Snake alphas. Codec weight
memory 1.6 GB → 0.75 GB; serving RTF unchanged (the DAC was already
noise in the frame budget).
