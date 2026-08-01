# Contributing to s2pro-native

Thank you for helping improve s2pro-native. This project combines a native
model runtime, CUDA kernels, frozen module contracts, an HTTP service, and an
FP8 GEMM integration. Changes must preserve the boundaries that keep those
layers auditable.

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md). Security-sensitive findings must follow
the private process in [SECURITY.md](SECURITY.md), not a public issue.

## Project scope

The supported inference target is exactly the
[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) checkpoint
(OpenAudio S2). The production surface accepts text (plus, once encoder
weights are available, reference audio for voice cloning) and returns
progressive 44.1 kHz PCM or buffered WAV.

The following are outside the current project scope:

- checkpoints other than S2-Pro, including S2.1-Pro (API-only, no public
  weights) and S1;
- Python, PyTorch, SGLang, vLLM, or another model framework in the inference
  runtime;
- x86-64, a GPU architecture other than `sm_121`/`sm_120`, or CPU-only
  inference;
- integration with unrelated production services.

Offline reference tooling (checkpoint conversion, parity fixtures) may use a
language appropriate to the task, but it must remain outside the runtime and
never become a hidden inference dependency.

Discuss a proposed scope expansion before implementing it.

## Code rules

- C11 host code; CUDA kernels in `.cu` files behind `extern "C"` launchers.
  C++ is allowed only in `src/fso/fso_wrap.cpp`.
- `include/s2pro/*.h` are frozen module contracts ([CONTRACT.md](CONTRACT.md)).
  Changing one is an interface change and must be called out in the pull
  request.
- Every `cudaError_t` is checked; everything returns `s2p_status`.
- No stubs. A genuinely blocked sub-feature is implemented around and marked
  with a greppable `/* S2P_GAP: reason */`.
- No third-party libraries in the runtime. The JSON parser, tokenizer, HTTP
  server, and WAV writer are first-party on purpose.

## Documentation and evidence

All repository documentation is written in clear, professional English.
Performance numbers in the README must come from a run on the target hardware
and state their conditions (GEMM path, batch, prompt length). RTF is always
reported as compute ÷ audio, where below 1.0 means faster than playback. Do
not present smoke-level measurements as validated parity — the layer-parity
comparison against the PyTorch reference is tracked in the roadmap and its
absence is stated where relevant.

## Building and verifying

Builds run inside `nvidia/cuda:13.0.3-devel-ubuntu24.04`; the host needs only
Docker with the NVIDIA runtime. See [docs/SPARK.md](docs/SPARK.md) for the
exact commands, `make syntax` for a compile-only pass, and `make selftest`
for the host-side JSON and tokenizer self-tests. A change to model, text, or
DAC code should run `build/s2p-test` against the real checkpoint before
review.

## Git practices

- Model weights, checkpoint-derived assets, and secrets never enter Git.
- Commit messages are imperative, scoped, and free of filler
  (`fix(slowar): …`, `docs: …`).
- Shared branches are never force-pushed.
