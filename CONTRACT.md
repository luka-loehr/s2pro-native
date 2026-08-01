# Module contract & ownership

`include/s2pro/*.h` is the frozen contract between modules. Implementations
code against it; private headers live inside each module's `src/<dir>/` only.

This codebase was built by five parallel agents in a single coordinated pass
(one module each), integrated and smoke-tested on a DGX Spark the same day.
The contract-first split below is what made that work — it is kept as the
map of the codebase.

| Module | Directories | Deliverables |
|---|---|---|
| core    | `src/core/`            | status, tensor, CUDA utils, JSON parser, safetensors (mmap, shards), config loader, WAV writer |
| slowar  | `src/slowar/`          | shared CUDA primitives (`kernels.h`), 36-layer backbone, KV cache, prefill+decode, lockstep batch decode, two-softmax sampling |
| fastar  | `src/fastar/`, `src/text/` | 4-layer fast-AR (no qk-norm, untied head), byte-level BPE tokenizer, hand-written ChatML prompt builder, VQ-part assembly |
| dac     | `src/dac/`             | modded-DAC / Firefly-GAN decoder (RVQ `from_indices`, conv/upsample stacks), streaming windows + crossfade, encoder scaffolding |
| serve   | `src/sched/`, `src/http/`, `src/fso/`, `src/main.c` | scheduler (lockstep loop, callbacks), HTTP server, `fso_wrap.cpp` (extern-C shim over fish-scales-ops), build system |

Rules the codebase follows:

- C11 host code; CUDA kernels in `.cu` with `extern "C"` launchers. C++ only
  in `src/fso/fso_wrap.cpp` (links the CUTLASS runner) — nowhere else.
- No stubs. Genuinely blocked sub-features are implemented around and marked
  with a greppable `/* S2P_GAP: reason */`.
- Every `cudaError_t` is checked. Everything returns `s2p_status`.
- `docs/PORTING.md` is the authoritative spec (dims, exact sampling
  algorithm, prompt format, and the three silent-garbage pitfalls: bf16 RoPE
  truncation, two-softmax sampling order, interleaved VQ injection).
