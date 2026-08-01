# Layer-parity protocol and results

Parity gates the native engine against a pure-PyTorch reference oracle that
implements the S2-Pro slow-AR backbone, fast-AR decoder, and modded-DAC codec
directly from the safetensors checkpoint (greedy, repetition penalty 1.1 over
a 16-token window, fixed prompt). The oracle dumps, for one prompt:

- the residual-stream hidden after every backbone layer (last prompt
  position) plus the final norm,
- the full prefill and step-1 logits rows,
- the nine fast-AR residual-step logits of the first frame,
- all greedy semantic tokens and frames,
- the reference codec latent and PCM.

`tools/parity_prep.py` converts a fixture set into a work dir;
`build/s2p-parity` (see the `parity` make target) replays the exact prompt
ids through the native engine with the dump hooks enabled and additionally
decodes the ORACLE's frames through the native DAC, isolating vocoder
fidelity from AR trajectory divergence; `tools/parity_compare.py` computes
the metrics. All tools are dependency-free (no numpy).

## Results — 2026-08-01, DGX Spark GB10 (sm_121), 26-id prompt, 64 steps

### BF16 (default path): PASS

| Stage | Result |
| --- | --- |
| Backbone hiddens (36 layers + final norm) | cos ≥ 0.999957, median 0.999981 |
| Prefill logits (155776 row) | cos 0.999986, argmax identical |
| Step-1 logits | cos 0.999415, argmax identical |
| Fast-AR first-frame residual steps | 8 of 9 argmax identical, cos ≥ 0.998328 |
| Native DAC on the oracle's frames vs reference PCM | cos 1.000000, SNR 65.85 dB |

The single fast-AR mismatch (codebook 8) is an exact bf16 tie in the native
logits (both candidates 8.9375) against a one-ULP gap in the reference
(8.9375 vs 8.8750). One flipped near-tie changes the VQ feedback and the
greedy trajectory diverges from step 2 onward — expected behavior for greedy
decoding at bf16 across different GEMM stacks, not an implementation defect.
Full-trajectory token identity is therefore NOT a parity criterion; per-stage
numeric agreement and first-frame argmax agreement are.

Evidence: [`results/parity-bf16-oracle-fox-2026-08-01.json`](results/parity-bf16-oracle-fox-2026-08-01.json)

### FP8 (fish-scales-ops block-scale, `S2P_FP8=1`): FAIL

| Stage | Result |
| --- | --- |
| Backbone hiddens | cos min 0.329, median 0.805 |
| Prefill logits | cos 0.368, argmax WRONG |
| Generation on the oracle prompt | immediate EOS, 0 frames |

Per-GEMM FP8 noise (~3.7% relative, cos 0.9993 — consistent with the
kernel-level qualification) compounds across ~144 sequential GEMMs into a
corrupted residual stream. This is inherent to the 1×128/128×128 UE8M0
quantization at this network depth, not an integration bug. The FP8 path
stays opt-in and is REJECTED as the production decode path; the INT8
per-channel weight-only alternative is the roadmap candidate and must pass
this same gate.

Evidence: [`results/parity-fp8-oracle-fox-2026-08-01.json`](results/parity-fp8-oracle-fox-2026-08-01.json)

## Listening evidence

Parity establishes numeric fidelity, not perceptual quality. The listening
set (not in git, regenerable with `s2p-parity` + `s2p-test`) pairs the
PyTorch oracle WAV with the native codec on identical frames, the native
end-to-end BF16 output, and the FP8 output. Subjective judgment is the
project owner's and is recorded separately when given.
