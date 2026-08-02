# Weight quantization: from FP8 failure to trained 4-bit

This document records the complete quantization ladder of s2pro-native on
the DGX Spark (GB10, `sm_121`), including every measured negative result.
All numbers were produced on this hardware with the tooling in this
repository; evidence JSONs live in `benchmarks/parity/results/`.

## Why weights decide the RTF

Decode at batch 1 is bandwidth-bound: every generated frame (46.4 ms of
audio) reads all ~7.75 B weight parameters — the 3.63 B backbone once,
the tied lm-head (0.40 B), and the 0.41 B fast-AR decoder nine times
(its residual-codebook cascade), which makes the small fast-AR the
largest single stream. Time per frame ≈ bytes ÷ memory bandwidth, so
precision is the lever:

| weight stream / frame | BF16 | INT8 | INT4 backbone + INT8 rest | all-INT4 |
| --- | ---: | ---: | ---: | ---: |
| bytes | 15.5 GB | 7.76 GB | 6.17 GB | 4.37 GB |

## The ladder

1. **FP8 (fish-scales-ops block-scale): REJECTED.** Kernel-level cos
   0.9993 vs BF16, but ~3.7 % relative noise per GEMM compounds over
   ~144 sequential GEMMs: backbone cos collapses to 0.33, prefill argmax
   wrong, immediate EOS. Depth kills block-FP8 here.
2. **INT8 per-channel weight-only: the workhorse.** absmax/127 RTN per
   output channel, fused int8×bf16 GEMV (one warp per row, FP32
   accumulate). Parity: backbone cos ≥ 0.99989, prefill/step-1 argmax
   identical, fast-AR 8/9 (the 9th is a documented bf16 near-tie).
   Decode 93.3 → 39.7 ms/frame.
3. **Naive per-channel INT4: audibly fails, and the failure is
   autoregressive.** First ~10 s of a take sound excellent (propped by
   the pristine reference block in the KV), then progressive muffling:
   per-step weight noise (backbone cos 0.977, prefill argmax WRONG,
   fast-AR argmax 1/9) compounds through the growing self-generated
   context. The HF-envelope-over-time metric (below) shows a collapse to
   a floor by ~15 s with no recovery.
4. **Group-wise INT4 + MSE clip search + mixed precision: the rescue.**
   One f16 scale per 32 in-features (4.5 bits/weight all-in), each scale
   candidate evaluated at its f16-rounded value in a 32-step clip
   search; fast-AR and lm-head stay per-channel INT8. Parity returns to
   the INT8 class: prefill AND step-1 argmax identical, fast-AR 8/9.
   Ablations (all measured): g64 loses step-1 argmax and drops fast-AR
   to 3/9; disabling the MSE search at g32 costs prefill cos
   0.9987 → 0.9907 and fast-AR 8/9 → 7/9.
5. **Packed kernels: speed without touching sound.** Two 4-bit weights
   per byte, unpacked in-register into the same accumulation order
   against the same f16 scales — packed and unpacked storage produce
   bit-identical outputs (parity JSON exactly equal; all listening takes
   MD5-identical across storages). This is what turns value-precision
   quality into bandwidth: zero-shot wall RTF 1.10 → 0.95 at the time of
   landing.
6. **QAT self-distillation of the fast-AR (tools/qat_fastar.py): the
   road to all-INT4.** See below.

## Mixed precision is measured, not precautionary

Every attempt to put any part of the fast-AR at 4 bits without training
fails the parity argmax gate (reference class: 8/9):

| fast-AR tensors at INT4-g32 | fast-AR argmax |
| --- | ---: |
| none (deployed) | **8/9** |
| fused qkv + gate/up | 5/9 |
| gate/up only | 3/9 |
| everything | 2/9 |

The module decides nine greedy 1024-way argmaxes per frame with
intra-frame feedback and only four layers of depth: quantization noise
flips near-tied codebook decisions and the flips cascade within the
frame. Group scales cannot fix a tensor whose output is consumed by an
argmax — consistent with llama.cpp's unconditional per-tensor promotions
and the logit-distortion literature.

## Measurement methodology

- **Layer parity** (`benchmarks/parity/`): per-stage cosine + argmax
  agreement against a pure-PyTorch oracle. Discrete decisions (argmax)
  are the acceptance metric; cosine alone is blind to the failure mode
  that matters (a 0.08-MSE logit row can carry the wrong top-1).
- **HF-envelope over time**: per-second high-frequency energy ratio
  (first-difference energy / total energy, 1 s windows, 10 s buckets).
  Muffling appears as a monotone collapse to a floor with no recovery;
  content-driven variation recovers. This separated the quantization
  regression (collapse) from ordinary take variation, and later showed
  the chunked long-form fix holding pause structure over 130+ s.
- **EOS termination probe**: low-bit mis-sampling concentrates on
  low-entropy tokens — where end-of-audio lives — and an envelope cannot
  see runaway generation. 24 utterance pairs INT4 vs INT8: length ratios
  0.88–1.23, zero runaway/premature flags; long takes terminate on the
  identical frame.
- **Bit-exactness (MD5)**: deterministic greedy smoke takes must hash
  identically across kernel rewrites that claim unchanged arithmetic.
- **The project owner's ear is the final gate.** Objective metrics
  select candidates; listening accepts them.

## QAT self-distillation (fast-AR → INT4)

Goal: an all-INT4 weight stream (4.37 GB/frame) without a quality loss
the ear can detect. Method (`tools/qat_fastar.py`):

- **Data**: the serving engine dumps (final-normed hidden, sampled
  semantic code) per generated frame (`S2P_DUMP_FRAMES`,
  `src/slowar/slowar.c`) during ordinary multilingual generation across
  registry voices — no external dataset.
- **Teacher**: the BF16 fast-AR, reimplemented differentiably in torch
  (seq ≤ 10 → teacher-forced parallel forward); exact greedy teacher
  trajectories are recomputed offline. Sanity anchor: teacher free-run
  vs the INT8 engine's dumped codes agree at 0.85 — a wrong port would
  read ~0.
- **Student**: same weights behind a straight-through fake quantizer
  that is a bit-faithful port of the engine's load-time quantizer (g32
  symmetric 15 levels, 32-step MSE clip search, candidates rounded
  through f16). The trained grid IS the deployed grid: exporting the
  trained BF16 tensors into the checkpoint and loading with
  `S2P_INT4_ALL=1` reproduces training numerics exactly.
- **Loss**: KL(teacher ‖ student) over the nine residual steps,
  teacher-forced; second half of training mixes DAgger batches (student
  free-run prefixes, teacher corrective targets) against exposure bias —
  the deployment failure is drift on the student's own prefixes.
- **Metric and bar**: per-step argmax agreement with the teacher on a
  held-out set, teacher-forced and free-running. The per-channel INT8
  deployment quantizer scores 0.964 / 0.896 — that is the bar. Untrained
  INT4-g32 scores 0.710 / 0.447; a 200-step smoke already moves it to
  0.805 / 0.586.
- **Deployment**: `tools/apply_qat_patch.py` writes a patched checkpoint
  copy; the engine is unchanged.

Gate chain for the QAT artifact: held-out agreement vs the INT8 bar →
HF-envelope stability on long takes → EOS probe → listening sign-off.

### Training runs

- **Run 1** (aborted): unchunked candidate search made the fake quantizer
  ~6 s/step; killed after the first eval. The chunked 8-candidate search
  (identical first-minimum tie order, verified by exact reproduction of
  the pre-QAT eval) and a no-grad quantized-weight cache in `FQLinear`
  (a DAgger rollout would otherwise re-quantize every tensor nine times)
  brought non-DAgger steps to interactive speed. DAgger steps stay
  ~4× more expensive — the 9-step greedy rollout is inherent.
- **Run 2**: 2000 steps, batch 256, lr 3e-5 cosine, DAgger second half,
  corpus 60,222 frames (360 takes: 10 texts × 12 voices × 3 seeds).
  Free-run agreement 0.4550 → 0.5556 (step 500) → 0.6029 (step 1000);
  the flattening (+10.1, then +4.7 per 500 steps) motivated the
  overnight protocol instead of blindly extending the schedule.

### Overnight protocol (corpus v2 + warm-started long run)

Two levers, applied together because per-step gains were flattening
while epochs over the 52k-frame training split were piling up:

1. **Corpus v2** (`tools/qat_corpus.py`): grow the dump from 60k toward
   ~270k frames — all 33 registry voices, 22 texts across the seven
   languages, ~30 % long-form (three ~80 s multilingual passages), 590
   sampled (voice, text, seed) combos. The dump hook opens its file in
   append mode, so the v1 frames stay valid as a prefix and the cached
   teacher trajectories are reused; only the tail is recomputed.
   The two demo texts used for final listening takes are deliberately
   excluded from v2 so the deliverables are not generated from training
   material (the v1 corpus did contain the short Sprachentag text —
   recorded here for honesty; the long demo text was never trained on).
2. **Warm-started long run**: `--init` continues from run 2's weights
   instead of restarting; `--holdout-from` draws the held-out set only
   from frames the init run never saw, keeping the eval honest. Best-by-
   holdout and last checkpoints are written at every eval interval, so
   the run can be cut at a wall-clock deadline (07:00 cutoff policy)
   and still yield its best artifact. Cosine floor `--lr-min 1e-6`,
   DAgger from 35 % of the schedule.

Results: recorded after the run completes.
