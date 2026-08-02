# Contributing to s2pro-native

s2pro-native combines a native model runtime, CUDA kernels, frozen module
contracts, an HTTP service, and quantized GEMM paths. Changes must preserve
the boundaries that keep those layers auditable.

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Security-sensitive findings must
follow the private process in [SECURITY.md](SECURITY.md), not a public
issue.

## 1. Project scope

The supported inference target is exactly the
[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) checkpoint
(OpenAudio S2). The production surface accepts text — plus reference audio
for voice cloning — and returns progressive 44.1 kHz PCM or buffered WAV.

Outside the project scope:

- checkpoints other than S2-Pro, including S2.1-Pro (API-only, no public
  weights) and S1;
- Python, PyTorch, SGLang, vLLM, or any other model framework in the
  inference runtime;
- x86-64, GPU architectures other than `sm_121`/`sm_120`, or CPU-only
  inference;
- integration with unrelated production services.

Offline tooling (checkpoint conversion, parity fixtures, training scripts)
may use whatever language fits the task, but it stays outside the runtime
and must never become a hidden inference dependency.

Discuss a proposed scope expansion before implementing it.

## 2. Code rules

- C11 host code; CUDA kernels in `.cu` files behind `extern "C"`
  launchers. C++ is allowed only in `src/fso/fso_wrap.cpp`.
- `include/s2pro/*.h` are frozen module contracts
  ([CONTRACT.md](CONTRACT.md)). Changing one is an interface change and
  must be called out in the pull request.
- Every `cudaError_t` is checked; every fallible function returns
  `s2p_status`.
- No stubs. A genuinely blocked sub-feature is implemented around and
  marked with a greppable `/* S2P_GAP: reason */`.
- No third-party libraries in the runtime. The JSON parser, tokenizer,
  HTTP server, and WAV writer are first-party on purpose.

## 3. Documentation scheme

All repository documentation follows one scheme. I wrote these documents
for a technically critical readership; contributions must hold the same
standard.

**Document classes.**

- *Technical reports* (`docs/QUANT.md`, `docs/DAC-KERNELS.md`,
  `docs/QAT-RUNS.md`, `benchmarks/parity/README.md`): fixed skeleton —
  scope, problem statement, method, evaluation, results (negative results
  reported with the same rigor as positive ones), reproduction.
- *Specifications* (`docs/PORTING.md`, `CONTRACT.md`): normative
  statements; every constant traceable to source or checkpoint; inferred
  values flagged.
- *Guides and references* (`README.md`, `docs/SPARK.md`,
  `docs/VOICES.md`, tool READMEs): purpose, usage, constraints,
  pointers to the reports for evidence.

**Register.**

- English, scientific-technical prose. No narrative, no anecdote, no
  anthropomorphization.
- First person singular for design decisions and acceptance judgments;
  neutral present tense for system description.
- Every performance claim states its measurement conditions (hardware,
  GEMM path, prompt length, batch, sampling). RTF is compute ÷ audio;
  below 1.0 means faster than playback.
- Rejected approaches are documented with their measurements and the
  cause of failure, in the report that owns the topic.
- Smoke-level observations are never presented as validated results;
  the validation gate for numerics is the layer-parity protocol.

## 4. Building and verifying

Builds run inside `nvidia/cuda:13.0.3-devel-ubuntu24.04`; the host needs
only Docker with the NVIDIA runtime. See [docs/SPARK.md](docs/SPARK.md)
for the exact commands, `make syntax` for a compile-only pass, and
`make selftest` for the host-side JSON and tokenizer self-tests. A change
to model, text, or DAC code should run `build/s2p-test` against the real
checkpoint before review.

## 5. Git practices

- Model weights, checkpoint-derived assets, and secrets never enter Git.
- Commit messages are imperative, scoped, and free of filler
  (`fix(slowar): …`, `docs: …`).
- Shared branches are never force-pushed.
