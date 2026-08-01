# Changelog

All notable changes to s2pro-native are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for published releases.

## [Unreleased]

### Removed

- The committed sample voices (`neutral-female`, `young-male`,
  `deeper-male`). Reference audio is generated per deployment with
  `tools/voicegen` and is now gitignored (`voices/*.wav`, `voices/*.txt`):
  ~5.4 MB per voice is not worth carrying in git for output one command
  reproduces, and each deployment wants its own language mix anyway. `voices/`
  ships with only its README; an empty registry serves zero-shot.

### Fixed

- Streaming vocoder fidelity: the reference's 20-frame window overlap is far
  too little causal warmup for this DAC (post-module attention window is
  128 frames) — every window after the second decoded audibly differently
  from the parity-proven whole-buffer path (~5 dB SNR), and the reference
  crossfade additionally skips 512 samples of timeline per window (an
  audible click every stride). Overlap now defaults to 160 frames
  (`S2P_STREAM_OVERLAP` tunes) and the crossfade is timeline-preserving (a
  deliberate improvement over the reference). Same-codes validation vs the
  whole-buffer decode: length-exact, SNR 35.9 dB, cos 0.99988 (before:
  −3 dB, cos 0.02). All HTTP-streamed audio was affected; `s2p-test` gained
  `S2P_TEST_STREAM_WAV` to diff both DAC paths on identical codes.

### Added

- `"stream": false` on `POST /v1/tts`: buffers the take server-side and
  responds with exact `Content-Length` and exact RIFF sizes. Streamed WAVs
  necessarily carry saturated size fields (the 44 header bytes are sent
  before the first frame exists; chunked transfer cannot rewrite them),
  which strict players such as Apple's reject — buffered mode produces a
  well-formed downloadable file at the cost of time-to-first-audio.
- INT8 per-channel weight-only GEMM path (`S2P_INT8=1` / `--int8`): every
  linear is quantized per output channel at load (absmax/127,
  round-to-nearest) and the BF16 copy is freed; decode-M GEMMs run a fused
  int8×bf16 GEMV kernel (`src/sched/int8_gemv.cu`, one warp per output row,
  FP32 accumulate), prefill dequantizes layer-by-layer into a shared scratch
  and reuses the proven cuBLAS BF16 call, and the tied lm-head reads an int8
  sidecar of the embedding table. Passes the layer-parity gate (backbone cos
  ≥ 0.99989, prefill/step-1 argmax identical —
  `benchmarks/parity/results/parity-int8-oracle-fox-2026-08-01.json`).
  Measured on the GB10: decode 93.3 → 39.7 ms/frame (below the 46.4 ms frame
  budget; sustained short-context RTF 0.86 before vocoding), end-to-end RTF
  2.41 → 1.27, process peak memory ~12 → ~8.5 GB.
- `tools/voicegen` (Rust, offline): reference-voice generator on Gemini TTS
  via Vertex AI — passage authoring across arbitrary language lists, all 30
  prebuilt voices, exact 147/80 polyphase resampling to 44.1 kHz (with a
  headroom guard — band-limited interpolation overshoots a 0 dBFS source, so
  takes are scaled rather than clipped), and
  LCS-scored transcript verification (`--verify`). Writes registry-ready
  `voices/<name>.wav` + `.txt` pairs; credentials come from gcloud at
  runtime, nothing is stored.
- Voice registry (`include/s2pro/voices.h`, `src/voice/`): named references
  as `<name>.wav` + `<name>.txt` pairs, DAC-encoded once at startup. An empty
  or missing directory serves zero-shot only (docs/VOICES.md documents why
  references must come from an accent-free multilingual source).
- HTTP API: `GET /v1/voices`, `"voice"` selection on `POST /v1/tts`, and
  per-request cloning via `"reference_audio_b64"` + `"reference_text"`
  (wav, 16-bit mono 44.1 kHz, max 15 s; encoded on the scheduler thread).
- Public wav readers (`s2p_wav_read_f32`/`s2p_wav_parse_f32`) and
  multi-reference prompt support exercised up to a 51 s / 1,099-frame
  reference block (mixed-language generation in one call).
- First runtime qualification of `s2pro-server`: healthz, voice listing,
  named-voice + zero-shot + inline-clone synthesis over HTTP on the DGX
  Spark; measured TTFA 1.06 s (zero-shot) / 2.45 s (50 s reference block).

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
