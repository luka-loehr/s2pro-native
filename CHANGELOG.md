# Changelog

All notable changes to s2pro-native are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for published releases.

## [Unreleased]

### Added

- Complete native C11 + CUDA serving stack for Fish Audio S2-Pro on NVIDIA
  DGX Spark (GB10, `sm_121`): safetensors loader (single and sharded),
  arena JSON parser, byte-level BPE tokenizer (bit-exact against the HF
  `tokenizers` reference on a 4,000-case fuzz), hand-written ChatML prompt
  builder with VQ-part injection, 36-layer slow-AR backbone with per-head
  qk-norm and reference-exact two-softmax sampling, 4-layer fast-AR residual
  decoder, modded-DAC / Firefly-GAN vocoder with streaming crossfade,
  lockstep scheduler, and a dependency-free HTTP/1.1 streaming server.
- FP8 block-scale GEMM path through fish-scales-ops (`S2P_FP8=1`): weights
  quantized once at load, activations per call; verified numerically on
  `sm_121` (cos 0.9993 vs BF16) and measured end-to-end (RTF 2.38 → 1.39,
  compute ÷ audio).
- End-to-end smoke qualification on the real 9.1 GB checkpoint:
  deterministic greedy generation, speech-shaped non-silent audio.
- Reproducibility scripts: checkpoint fetch from Hugging Face and
  fish-scales-ops object build (`scripts/`).

- Layer-parity gate against a pure-PyTorch oracle
  (`benchmarks/parity/`): dependency-free fixture tooling
  (`tools/parity_prep.py`, `tools/parity_compare.py`), env-gated dump hooks,
  and the `s2p-parity` harness. BF16 **passes** at every stage (backbone
  cos ≥ 0.99996, DAC SNR 65.9 dB on reference frames; the one first-frame
  argmax flip is an exact bf16 tie). FP8 **fails** (backbone cos collapses
  to 0.33; immediate EOS on the oracle prompt) and is rejected as the
  production decode path.

- Voice-cloning encode activated end to end: `tools/convert_codec_full.py`
  produces the full codec artifact (encoder conv stack + quantizer
  pre-module, weight-norm folded), the already-implemented native encode
  path self-activates, and `s2p-test` gained reference-wav support
  (`S2P_TEST_REF`/`S2P_TEST_REF_TEXT`). First exercised 2026-08-01: 8.4 s
  reference → 181 VQ frames (1.1 s encode), prompt with injected reference,
  generation to natural EOS.
- Listening-run controls on `s2p-test`: `S2P_TEST_TEMP`, `S2P_TEST_SEED`,
  `S2P_TEST_FRAMES` (reference sampler temp 0.8 / top-p 0.8; natural EOS).

### Known gaps

- Single-stream decode is above realtime (BF16 RTF 2.38); the path below
  1.0 is tracked in the README roadmap, with INT8 per-channel weight-only
  GEMV as the primary 8-bit candidate after the FP8 parity failure.
- Encoder output has no oracle-fixture parity comparison yet (judged by
  cloning fidelity by ear so far).
