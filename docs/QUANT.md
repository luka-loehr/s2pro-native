# Weight Quantization: Methods, Measurements, and Negative Results

Technical report. Hardware: NVIDIA DGX Spark (GB10, `sm_121`). All numbers
in this document were produced on this hardware with the tooling in this
repository; parity evidence is archived as JSON under
`benchmarks/parity/results/`. Training experiments for the learned 4-bit
stage are reported separately in [QAT-RUNS.md](QAT-RUNS.md).

## 1. Scope

This report covers the complete weight-precision ladder of the serving
engine: why weight precision determines the real-time factor, each
quantization scheme I evaluated, the measurement protocol that accepted or
rejected it, and the measured failure modes — including the schemes that
did not survive. The learned (QAT) stage is described here at the method
level; its experimental runs live in the companion report.

## 2. Problem statement

Decode at batch 1 is bandwidth-bound. Every generated frame (46.4 ms of
audio) reads all ~7.75 B weight parameters: the 3.63 B backbone once, the
tied lm-head (0.40 B) once, and the 0.41 B fast-AR decoder nine times for
its residual-codebook cascade — which makes the smallest module the
largest single weight stream. Time per frame ≈ bytes ÷ memory bandwidth
(~220 GB/s sustained on this memory system), so precision is the primary
lever:

| weight stream per frame | BF16 | INT8 | INT4 backbone + INT8 rest | all-INT4 |
| --- | ---: | ---: | ---: | ---: |
| bytes | 15.5 GB | 7.76 GB | 6.17 GB | 4.37 GB |

## 3. Measurement protocol

Five instruments, in gate order:

1. **Layer parity** (`benchmarks/parity/`): per-stage cosine similarity
   and argmax agreement against a pure-PyTorch oracle. Discrete decisions
   (argmax) are the acceptance metric; cosine alone is blind to the
   failure mode that matters, since a logit row with small MSE can still
   carry the wrong top-1.
2. **HF-envelope trajectory**: per-second high-frequency energy ratio
   (first-difference energy over total energy, 1 s windows, summarized in
   10 s buckets). Spectral muffling appears as a monotone collapse to a
   floor with no recovery; content-driven variation recovers. This metric
   separated quantization regressions from ordinary take variation, and
   later verified that chunked long-form generation holds pause structure
   beyond 130 s.
3. **EOS termination probe**: low-bit mis-sampling concentrates on
   low-entropy tokens — where the end-of-audio token lives — and an
   envelope metric cannot see runaway generation. The probe compares take
   lengths pairwise (candidate vs INT8) over 24 utterance pairs and flags
   ratios outside [0.5, 1.5].
4. **Bit-exactness (MD5)**: deterministic greedy takes must hash
   identically across any kernel or storage change that claims unchanged
   arithmetic.
5. **Critical listening.** Objective metrics select candidates; I accept
   an artifact only after listening to it. No scheme in this report was
   deployed on numbers alone.

## 4. Experiments

### 4.1 FP8 block-scale (fish-scales-ops) — rejected

Kernel-level qualification passed (cos 0.9993 vs BF16 per GEMM), but
~3.7 % relative noise per GEMM compounds over the ~144 sequential GEMMs of
the backbone: layer-parity cosine collapses to 0.33, the prefill argmax is
wrong, and generation terminates immediately. The failure is inherent to
1×128/128×128 block scaling at this network depth, not an integration
bug. I keep the path in the tree for memory-system measurements only.

### 4.2 INT8 per-channel weight-only — the workhorse

Absmax/127 round-to-nearest per output channel at load; decode-M GEMMs run
a fused int8×bf16 GEMV (one warp per output row, FP32 accumulation);
prefill dequantizes layer-by-layer into a shared scratch and reuses the
proven cuBLAS BF16 call. Parity holds the BF16 class: backbone cos
≥ 0.99989, prefill and step-1 argmax identical, fast-AR 8/9 (the ninth is
a documented bf16 near-tie present in BF16 itself). Decode drops
93.3 → 39.7 ms/frame.

### 4.3 Naive per-channel INT4 — an autoregressive failure

The first ten seconds of a take sound excellent — propped up by the
pristine reference block in the KV cache — then the output muffles
progressively. Per-step weight noise (backbone cos 0.977, prefill argmax
wrong, fast-AR argmax 1/9) compounds through the growing self-generated
context; the HF envelope collapses to a floor by ~15 s with no recovery.
The failure mode is a property of autoregression: static parity metrics
understate it, which is why the envelope instrument exists.

### 4.4 Group-wise INT4 with MSE clip search — the rescue

One f16 scale per 32 in-features (4.5 bits per weight all-in), each scale
candidate evaluated at its f16-rounded value in a 32-step clip search;
fast-AR and lm-head stay per-channel INT8 (see §5). Parity returns to the
INT8 class: prefill and step-1 argmax identical, fast-AR 8/9.

Ablations, all measured: g64 loses the step-1 argmax and drops the
fast-AR to 3/9; disabling the MSE search at g32 costs prefill cosine
0.9987 → 0.9907 and fast-AR 8/9 → 7/9.

### 4.5 Packed storage — bandwidth without touching the values

Two 4-bit weights per byte, unpacked in-register into the same
accumulation order against the same f16 scales. Packed and unpacked
storage produce bit-identical output (parity JSON exactly equal; listening
takes MD5-identical across storages), so the packing converts value-level
quality directly into bandwidth: zero-shot wall RTF 1.10 → 0.95 at the
time of landing.

## 5. Mixed precision is measured, not precautionary

Every attempt to put any part of the fast-AR at 4 bits without training
fails the parity argmax gate (reference class: 8/9):

| fast-AR tensors at INT4-g32 | fast-AR argmax |
| --- | ---: |
| none (deployed) | **8/9** |
| fused qkv + gate/up | 5/9 |
| gate/up only | 3/9 |
| everything | 2/9 |

The module makes nine greedy 1024-way argmax decisions per frame with
intra-frame feedback and only four layers of depth: quantization noise
flips near-tied codebook decisions and the flips cascade within the frame.
Group scales cannot fix a tensor whose output is consumed by an argmax —
consistent with llama.cpp's unconditional per-tensor promotions and the
logit-distortion literature.

## 6. QAT self-distillation of the fast-AR

Goal: the all-INT4 weight stream (4.37 GB/frame) without a quality loss
the ear can detect. Method (`tools/qat_fastar.py`):

- **Data**: the serving engine dumps (final-normed hidden state, sampled
  semantic code) per generated frame (`S2P_DUMP_FRAMES`,
  `src/slowar/slowar.c`) during ordinary multilingual generation across
  registry voices — no external dataset. The dump file is append-only, so
  a grown corpus keeps earlier frames as a valid prefix.
- **Teacher**: the BF16 fast-AR, reimplemented differentiably in torch
  (sequence length ≤ 10 permits a teacher-forced parallel forward); exact
  greedy teacher trajectories are recomputed offline. Sanity anchor:
  teacher free-run vs the INT8 engine's dumped codes agree at 0.85 — a
  wrong port would read near zero.
- **Student**: the same weights behind a straight-through fake quantizer
  that is a bit-faithful port of the engine's load-time quantizer (g32
  symmetric, 15 levels, 32-step MSE clip search, candidates rounded
  through f16). The trained grid is therefore the deployed grid:
  exporting the trained BF16 tensors into the checkpoint and loading with
  `S2P_INT4_ALL=1` reproduces training numerics exactly, with no engine
  changes.
- **Loss**: KL(teacher ‖ student) over the nine residual steps,
  teacher-forced; the later portion of training mixes DAgger batches
  (student free-run prefixes, teacher corrective targets) against
  exposure bias — the deployment failure mode is drift on the student's
  own prefixes.
- **Metric and bar**: per-step argmax agreement with the teacher on a
  held-out set, teacher-forced and free-running. The per-channel INT8
  deployment quantizer scores 0.964 / 0.896 on this metric — that is the
  bar, and it also calibrates what "perfect" means: even the deployed
  INT8 fast-AR, which passes listening, disagrees with the teacher on
  ~10 % of free-run decisions, because near-tied codebook picks are
  perceptually equivalent.
- **Deployment**: `tools/apply_qat_patch.py` writes a patched checkpoint
  copy; the engine is unchanged.

Gate chain for a QAT artifact: held-out agreement against the INT8 bar →
HF-envelope stability on long takes → EOS probe → critical listening.
Training corpus construction (`tools/qat_corpus.py`), run conditions, and
per-run results — including the aborted first run — are reported in
[QAT-RUNS.md](QAT-RUNS.md).

## 7. Reproduction

| Env | Effect |
| --- | --- |
| `S2P_INT8=1` | per-channel INT8 weight-only path |
| `S2P_INT4=1` | group-wise INT4 backbone on top of INT8 |
| `S2P_INT4_GROUP` | group size (default 32) |
| `S2P_INT4_MSE` | per-group MSE clip search (default on) |
| `S2P_INT4_PACKED` | packed nibble storage (default on; off = int8 container A/B) |
| `S2P_INT4_ALL=1` | fast-AR + lm-head also INT4 (requires a QAT-patched checkpoint) |
| `S2P_DUMP_FRAMES=path` | append per-frame QAT training records during serving |

Parity fixtures and comparison tooling are documented in
[benchmarks/parity](../benchmarks/parity/README.md).
