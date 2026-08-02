# Module Contracts and Ownership

Specification. `include/s2pro/*.h` is the frozen contract between
modules: implementations code against it, and private headers live inside
each module's `src/<dir>/` only.

I built this codebase contract-first: the five module tracks below were
implemented in parallel against these frozen interfaces and integrated and
smoke-tested on a DGX Spark the same day. The split is kept as the map of
the codebase.

| Module | Directories | Deliverables |
|---|---|---|
| core    | `src/core/`            | status, tensor, CUDA utils, JSON parser, safetensors (mmap, shards), config loader, WAV writer |
| slowar  | `src/slowar/`          | shared CUDA primitives (`kernels.h`), 36-layer backbone, KV cache, prefill + decode, lockstep batch decode, two-softmax sampling |
| fastar  | `src/fastar/`, `src/text/` | 4-layer fast-AR (no qk-norm, untied head), byte-level BPE tokenizer, hand-written ChatML prompt builder, VQ-part assembly |
| dac     | `src/dac/`             | modded-DAC / Firefly-GAN decoder (RVQ `from_indices`, conv/upsample stacks), bit-exact incremental streaming, encoder |
| serve   | `src/sched/`, `src/http/`, `src/fso/`, `src/main.c` | scheduler (lockstep loop, callbacks), HTTP server, `fso_wrap.cpp` (extern-C shim over fish-scales-ops), build system |

Rules the codebase follows:

- C11 host code; CUDA kernels in `.cu` files with `extern "C"` launchers.
  C++ only in `src/fso/fso_wrap.cpp` (links the CUTLASS runner) — nowhere
  else.
- No stubs. A genuinely blocked sub-feature is implemented around and
  marked with a greppable `/* S2P_GAP: reason */`.
- Every `cudaError_t` is checked. Every fallible function returns
  `s2p_status`.
- [docs/PORTING.md](docs/PORTING.md) is the authoritative specification
  (dimensions, exact sampling algorithm, prompt format, and the
  silent-garbage pitfalls: bf16 RoPE truncation, two-softmax sampling
  order, interleaved VQ injection).
