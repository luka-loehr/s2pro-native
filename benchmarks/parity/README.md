# Layer-Parity Protocol and Results

Technical report. Hardware: NVIDIA DGX Spark (GB10, `sm_121`). Evidence
JSONs are archived under `results/`.

## 1. Scope

Parity gates the native engine against a pure-PyTorch reference oracle
that implements the S2-Pro slow-AR backbone, fast-AR decoder, and
modded-DAC codec directly from the safetensors checkpoint (greedy,
repetition penalty 1.1 over a 16-token window, fixed prompt). Every
weight-precision path ships only after passing this gate; the gate's
verdicts for BF16, INT8, FP8, and group-wise INT4 are recorded below.

## 2. Protocol

The oracle dumps, for one prompt:

- the residual-stream hidden state after every backbone layer (last
  prompt position) plus the final norm,
- the full prefill and step-1 logits rows,
- the nine fast-AR residual-step logits of the first frame,
- all greedy semantic tokens and frames,
- the reference codec latent and PCM.

`tools/parity_prep.py` converts a fixture set into a work directory;
`build/s2p-parity` (the `parity` make target) replays the exact prompt ids
through the native engine with the dump hooks enabled, and additionally
decodes the *oracle's* frames through the native DAC — isolating vocoder
fidelity from AR trajectory divergence; `tools/parity_compare.py` computes
the metrics. All tools are dependency-free (no numpy).

Discrete decisions (argmax agreement) are the acceptance metric; cosine
similarity is reported but is not sufficient, since a logit row with small
error can still carry the wrong top-1.

## 3. Results — 2026-08-01/02, 26-id prompt, 64 steps

### 3.1 BF16 (default path): PASS

| Stage | Result |
| --- | --- |
| Backbone hiddens (36 layers + final norm) | cos ≥ 0.999957, median 0.999981 |
| Prefill logits (155776 row) | cos 0.999986, argmax identical |
| Step-1 logits | cos 0.999415, argmax identical |
| Fast-AR first-frame residual steps | 8 of 9 argmax identical, cos ≥ 0.998328 |
| Native DAC on the oracle's frames vs reference PCM | cos 1.000000, SNR 65.85 dB |

The single fast-AR mismatch (codebook 8) is an exact bf16 tie in the
native logits (both candidates 8.9375) against a one-ULP gap in the
reference (8.9375 vs 8.8750). One flipped near-tie changes the VQ
feedback and the greedy trajectory diverges from step 2 onward — expected
behavior for greedy decoding at bf16 across different GEMM stacks, not an
implementation defect. Full-trajectory token identity is therefore not a
parity criterion; per-stage numeric agreement and first-frame argmax
agreement are.

Evidence: [`results/parity-bf16-oracle-fox-2026-08-01.json`](results/parity-bf16-oracle-fox-2026-08-01.json)

### 3.2 INT8 per-channel weight-only (`S2P_INT8=1`): PASS

| Stage | Result |
| --- | --- |
| Backbone hiddens (36 layers + final norm) | cos ≥ 0.999862, median 0.999954 |
| Prefill logits (155776 row) | cos 0.999666, argmax identical |
| Step-1 logits (int8 GEMV + int8 tied head) | cos 0.999088, argmax identical |
| Fast-AR first-frame residual steps | 8 of 9 argmax identical, cos ≥ 0.998242 |
| Native DAC on the oracle's frames vs reference PCM | cos 1.000000, SNR 65.85 dB |

Same class as BF16 at every stage — the added per-channel quantization
noise costs about one nine of cosine and flips no argmax. The single
fast-AR mismatch is the same codebook-8 near-tie documented for BF16; the
trajectory-divergence caveat applies identically. Prefill exercises the
dequant + cuBLAS fallback (M = 26), the decode steps exercise the int8
GEMV (M = 1), so both serving paths are covered by one run.

Evidence: [`results/parity-int8-oracle-fox-2026-08-01.json`](results/parity-int8-oracle-fox-2026-08-01.json)

### 3.3 FP8 (fish-scales-ops block-scale, `S2P_FP8=1`): FAIL

| Stage | Result |
| --- | --- |
| Backbone hiddens | cos min 0.329, median 0.805 |
| Prefill logits | cos 0.368, argmax WRONG |
| Generation on the oracle prompt | immediate EOS, 0 frames |

Per-GEMM FP8 noise (~3.7 % relative, cos 0.9993 — consistent with the
kernel-level qualification) compounds across ~144 sequential GEMMs into a
corrupted residual stream. This is inherent to the 1×128/128×128 UE8M0
quantization at this network depth, not an integration bug. I rejected
FP8 as a production decode path; its timings remain valid as measurements
of the memory system.

Evidence: [`results/parity-fp8-oracle-fox-2026-08-01.json`](results/parity-fp8-oracle-fox-2026-08-01.json)

### 3.4 INT4 group-wise g32 + MSE clip search, mixed precision, packed (`S2P_INT4=1`): PASS

Scheme: backbone linears group-wise symmetric INT4 (one f16 scale per 32
K-elements — 4.5 bits per weight — with a per-group MSE clip search whose
candidates are evaluated at their f16-rounded values), stored packed two
weights per byte; fast-AR and tied lm-head per-channel INT8. The mixed
precision is measured, not precautionary: group-wise g32 still collapses
the fast-AR to argmax 2/9 ([docs/QUANT.md §5](../../docs/QUANT.md)).
Packed and int8-container storages produce bit-identical outputs (parity
JSON exactly equal, server WAVs byte-identical at fixed seed).

| Stage | Result |
| --- | --- |
| Backbone hiddens | cos min 0.9927, median 0.9977 |
| Prefill logits | cos 0.998670, argmax identical |
| Step-1 logits | cos 0.991824, argmax identical |
| Fast-AR first-frame residual steps | 8 of 9 argmax identical (the known near-tie) |
| Semantic trajectory | follows the oracle 4 steps |

The fast-AR 8/9 is free-running within the frame: each residual step
consumes the previous step's own argmax, and the one mismatch did not
cascade — subsequent steps still match the oracle. Termination is probed
separately, because low-bit mis-sampling concentrates on low-entropy
tokens (where end-of-audio lives) and an envelope metric cannot see it:
24 utterance pairs INT4 vs INT8 show length ratios 0.88–1.23 with zero
runaway or premature-EOS flags, and 104 s takes terminate at the
identical frame as INT8.

For contrast, naive per-channel INT4 fails this gate: prefill argmax
wrong, fast-AR argmax 1/9, audibly progressive muffling from ~10 s
(autoregressive compounding of per-step weight noise). Group-wise scales
carry most of the recovery; disabling the MSE search at g32 costs prefill
cos 0.9987 → 0.9907 and fast-AR 8/9 → 7/9; g64 loses the step-1 argmax
and drops the fast-AR to 3/9. HF-envelope trajectories over 104 s takes
show no muffling collapse. I confirmed the scheme by listening before
deployment.

Evidence: [`results/parity-int4-g32f16-oracle-fox-2026-08-02.json`](results/parity-int4-g32f16-oracle-fox-2026-08-02.json)
(f32-scale predecessor run:
[`results/parity-int4-g32-oracle-fox-2026-08-02.json`](results/parity-int4-g32-oracle-fox-2026-08-02.json))

## 4. Listening evidence

Parity establishes numeric fidelity, not perceptual quality. The
listening set (not in git; regenerable with `s2p-parity` + `s2p-test`)
pairs the PyTorch oracle WAV with the native codec on identical frames,
the native end-to-end BF16 output, and each quantized path's output.
Final acceptance of any path is by critical listening; the verdicts are
recorded in the documents that own each change.
