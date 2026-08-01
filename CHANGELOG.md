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

### Known gaps

- Layer parity against the PyTorch reference is not yet validated.
- Single-stream decode is above realtime (RTF 1.39 FP8); the path below 1.0
  is tracked in the README roadmap.
- Voice-cloning encode is disabled until the codec converter retains
  `encoder.*` tensors (`S2P_GAP` in `src/dac/dac.c`).
