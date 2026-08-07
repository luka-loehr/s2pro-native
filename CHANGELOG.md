# Changelog

All notable changes to s2pro-native are documented in this file. Entries
state their measurement conditions inline; the technical reports under
`docs/` own the full evidence.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
for published releases.

## [1.0.0] — 2026-08-07

First stable release. The serving stack, its quality gates, and its
documentation are complete; the API contract is frozen at
[`api/openapi.yaml`](api/openapi.yaml) and will evolve
backward-compatibly within 1.x.

Headline, measured on one DGX Spark (GB10, `sm_121`): wall RTF **0.51**
zero-shot / **0.52** warm voice reference / **0.50** chunked long-form,
TTFA 0.14–0.20 s, server start ~78 s warm, ~5.4 GB steady-state device
memory, up to 5 concurrent streams under the per-stream RTF < 1 rule.

### Changed
- Repository release sweep: verified-unused imports removed from the
  offline tools; full-tree artifact audit (tracked files are source,
  documentation, specifications, and parity evidence only — measured
  negative paths remain in-tree deliberately, each env-gated and
  documented in the reports).

## [0.2.0] — 2026-08-07

### Added
- Audit tail, all individually gated: fast-AR batched
  rope/append/attention (bit-identical, B=8 1.41 -> 1.35), split-K
  attention tile skew (bit-identical), per-connection clone-codes memo
  (clone long-form at registry RTF 0.52), DAC encode workspace freed
  after use, dead fast-AR slab memset dropped (bit-identical), and a
  parity-gated tensor-core prefill for chunk-sized M behind
  S2P_PREFILL_TC=1 (default off; cos 0.9926, TTFA 0.64 -> 0.61).
  Final series: B=4/8/12/16 worst RTF 0.83/1.36/2.18/2.82.
- api/openapi.yaml: the complete HTTP contract as OpenAPI 3.1 (every
  field, default, error shape, streaming/backpressure semantics).
- Sliced LM-head: the decode head serves the sampler's 4,097 candidate
  rows instead of all 155,776 — bit-identical (smoke and two-stream
  fixed-seed MD5s unchanged), ~389 MB less weight stream per frame.
  Wall RTF 0.55 -> 0.51 zero-shot / 0.52 voice-ref / 0.50 long-form.
  Full-vocab head stays on the parity-dump path and S2P_HEAD_FULL=1.
- Droppable encoder arenas: encode-only DAC weights (encoder stack, VQ
  in_proj, pre_module) load into their own arenas and are freed after
  the voice registry loads (S2P_KEEP_ENCODER=1 keeps per-request
  cloning); encode afterwards reports a clean error.
- Prefill activation scratch capped at 2048 rows (-218 MB), guarded in
  s2p_session_prefill, S2P_SCRATCH_ROWS overrides.
- Wall-clock send budget (S2P_HTTP_SEND_MS, default 2000): sockets are
  non-blocking and the scheduler-thread writer polls against a
  monotonic deadline — a trickle-reading client can no longer freeze
  all streams (verified by a slow-reader probe alongside a healthy
  parallel request).

### Fixed
- Packed-GEMV dispatch for 9..16 concurrent sessions: the staged-GEMV
  rewrite had dropped the M>8 instantiation (out-of-bounds register
  array, undefined behavior). B>=9 series re-measured on the fixed
  kernel: 0.86/1.41/2.27/2.97 at B=4/8/12/16 with the sliced head.
- Prefix-cache teardown freed only 16 of 40 entries (leak up to ~2 GB
  per destroy).
- SERVING.md 7: the KV-bandwidth attribution of the concurrency wall
  is retracted (units error; falsified by the section's own trimming
  experiment). The wall is the O(M) batch-GEMV arithmetic.

## [0.1.0] — 2026-08-03

First tagged release. Everything below this heading is the state that
`v0.1.0` points at: the complete native serving stack at wall RTF
0.58–0.60 single-stream (TTFA 0.13–0.20 s), the all-INT4 weight stream
accepted by listening, INT8 KV and embeddings, FP16 vocoder weights,
prequantized-weight and encoded-voice caches (start ~78 s), chunked
prefill, captured decode graphs to batch 16, cross-session batched DAC,
and the reports in `docs/` — including every measured negative result.

Deployment notes: quantized serving is
`S2P_INT8=1 S2P_INT4=1 S2P_INT4_ALL=1` against a QAT-patched checkpoint
(`tools/apply_qat_patch.py`); an unpatched checkpoint serves the
INT8-fast-AR configuration at 0.75–0.76. Concurrency is capped at 4
sessions under the per-stream RTF < 1 rule (5 measured, 4 with margin);
`docs/SERVING.md` records where the remaining wall is.

### Removed

- The committed sample voices (`neutral-female`, `young-male`,
  `deeper-male`). Reference audio is generated per deployment with
  `tools/voicegen` and is now gitignored (`voices/*.wav`, `voices/*.txt`):
  ~5.4 MB per voice is not worth carrying in git for output one command
  reproduces, and each deployment wants its own language mix anyway. `voices/`
  ships with only its README; an empty registry serves zero-shot.

### Changed

- Streaming vocoder REVERTED to reference-exact behavior (20-frame overlap,
  reference crossfade). An interim deviation (160-frame overlap plus a
  timeline-preserving crossfade) measurably narrowed the stream-vs-
  whole-buffer gap on identical codes (−3 dB → 35.9 dB SNR, length-exact),
  but the listening gate did not confirm it as the fix for the reported
  audible noise, so the port returns to bug-for-bug reference fidelity while
  the noise source is localized upstream. The measured reference-side
  properties stand documented: 20 frames of causal warmup against a
  128-frame DAC attention window, and a crossfade that drops 512 samples of
  timeline per window (verified against
  `fishaudio_s2_pro/streaming_vocoder.py`). `s2p-test` keeps
  `S2P_TEST_STREAM_WAV` for two-path diffs on identical codes.

### Added

- Serving-efficiency sprint (docs/SERVING.md, docs/QUANT.md §4.6/4.7,
  docs/DAC-KERNELS.md §8), each stage individually gated:
  prequantized-weight sidecar cache (server start 27.7 -> 5.1 s, smoke
  MD5 identical by construction); INT8 KV cache with g32-per-head-vector
  f16 scales (KV memory/traffic halved, parity holds the INT8 class
  exactly incl. the same cb5 near-tie); INT8 embedding lookups with the
  bf16 table dropped (-0.8 GB, argmax unchanged); FP16 vocoder weights
  (codec 1.6 -> 0.75 GB, 68.1 dB vs the f32 decode on identical codes,
  policy fixed before measurement). One measured negative result: fast-AR
  rope/append/attention launch fusion gains nothing under CUDA-graph
  replay (24.4 -> 24.3 ms/frame) and stays opt-in (S2P_FA_FUSED=1).
  Complete-stack serving: zero-shot 0.60 (TTFA 0.14 s), warm voice ref
  0.58 (TTFA 0.20 s), long-form 0.58/0.59, ~-2.3 GB memory. Batch load
  test under the per-stream RTF<1 rule: worst 0.69 at B=2, 0.99 at B=4,
  queueing cliff beyond the 4-slot cap -> cap 4 (3 with margin); levers
  documented.
- QAT training runs to completion (docs/QAT-RUNS.md): three runs (one
  aborted for trainer speed, fixed; one cold 2000-step run on the 60 k
  corpus, free-run 0.4550 → 0.6690; one warm-started 3000-step run on
  the 308 k corpus v2, best 0.6711 on a strictly-unseen holdout at step
  2500). The best artifact, deployed all-INT4 (`S2P_INT4_ALL=1`),
  passes every objective audio gate (stable HF envelope over 148 s,
  EOS probe 0/24 flags, reference-class take lengths) and serves at
  wall RTF 0.60–0.63 on an idle GPU (zero-shot TTFA 0.14 s, voice-ref
  0.20 s) — down from 0.75–0.76 with the INT8 fast-AR. Accepted by
  critical listening 2026-08-03; the all-INT4 stream is now the serving
  configuration and the README headline carries its numbers.
- QAT self-distillation tooling for the all-INT4 fast-AR
  (tools/qat_fastar.py + tools/apply_qat_patch.py, S2P_DUMP_FRAMES
  engine hook): the BF16 fast-AR trains its own 4-bit copy with the
  deployment quantizer inside the loop (teacher-forced KL + DAgger);
  held-out per-step argmax agreement against the teacher is the metric,
  the per-channel INT8 deployment quantizer (0.964 teacher-forced /
  0.896 free-run) is the bar. Methodology in docs/QUANT.md.
- GEMM-grade DAC conv kernels (docs/DAC-KERNELS.md): 2D register
  blocking (2 time positions x 8 output channels per thread, smem
  weight chunks) for all k<=7 convs and phase-partitioned tconvs (a
  block serves one to%%stride phase — no masked lanes, coalesced input
  columns). Whole-buffer DAC 15.1 -> 2.1 ms/frame, every step gated on
  byte-identical PCM (MD5); server wall RTF 0.92-0.95 -> 0.75-0.76,
  TTFA 0.15/0.21 s. The losing attempts (pure co-tiling, smem-staged
  co-tiling) are recorded with their numbers.
- DAC per-kernel-type profiler (S2P_DAC_PROF=1) and co-tiled pointwise
  convs: k=1 residual-unit convs re-read the input plane once per output
  channel (18 GB measured on 192ch where 0.2 GB is inherent); one thread
  now accumulates 16 output channels in unchanged per-output order —
  byte-identical output (MD5), 81 -> 20 ms on the heaviest launch,
  whole-buffer DAC 15.1 -> 12.5 ms/frame. The same tiling measured OUT
  for k=7 convs and tconvs (throughput-bound per output, idle lanes per
  tconv tap): those keep the reference kernels; a GEMM-grade 2D register
  tile is the remaining DAC lever. Also measured out: narrowing the
  fast-AR INT8 promotion (every tensor subset fails the parity argmax
  gate — see the fastar.cu site comment).
- Reference-block KV-prefix cache: the per-voice system block (reference
  VQ included, ~1282 of ~1360 prompt tokens at a 60 s reference) prefills
  once and later sessions seed their KV from a device blob (one strided
  2D copy, ~189 MB/voice, LRU over S2P_KV_CACHE_VOICES entries, default
  4); only the user turn still prefills. Long-form chunk chains hit the
  cache from chunk 2 on. Measured warm: voice-ref TTFA 1.58 s -> 0.23 s,
  voice-ref wall RTF 0.94, ~150 s chunked long-form takes 0.93-0.94 —
  together with the packed 4-bit backbone, EVERY serving path is below
  realtime. Also fixed: the boundary filter's tail holdback applied to
  streamed responses delayed first audio by the window length; streamed
  joins now trim leading silence only.
- Normalized inter-chunk pauses (`chunk_gap_ms`, default 1000 ms, env
  S2P_CHUNK_GAP_MS, 0 = raw concatenation): the boundary filter holds
  back a tail window per chunk, trims the trailing/leading silence around
  each join and inserts an exact runtime-tunable gap instead — listening
  verdict was that the model's own ~1-1.3 s sentence pauses are perfect
  while the raw stitched joins ran mostly under 1 s. Take-level edges are
  never trimmed. Verified: every join lands at the target within scan
  granularity; the model's in-chunk pauses are untouched. Also
  `chunk_sentences` (default 2) closes chunks after N sentences.
- Long-form sentence chunking on `POST /v1/tts` (`chunk_length`, default
  300 bytes, env `S2P_CHUNK_BYTES`, 0 = off; voiced requests only): text
  splits at sentence boundaries (`src/text/chunker.c`, UTF-8-aware,
  bracket-tag- and decimal-safe) and generates as chained scheduler
  requests against the fresh voice reference, streamed or buffered into
  ONE response. Fixes measured prosody flattening of long single-shot
  takes: punctuation pauses per 10 s bucket decayed from ~0.5-1.0 s early
  to ~0-0.2 s after ~40 s (identically at INT8 and INT4 — a property of
  the growing self-generated context, not of quantization); chunked, the
  pause structure holds through 130+ s and the same text runs ~26 %
  longer (the vanished pauses were rushing). Explicit seeds vary per
  chunk (seed + k*1000003) for reproducible-but-independent chunks.
- f16 group scales (4.5 bits per weight all-in): at g32 the f32 scale
  plane was 20% of the packed weight stream. The quantizer rounds every
  MSE candidate through f16 before evaluating it, so the stored half is
  exactly the value the search optimized and the packed/unpacked A/B
  stays bit-identical. Zero-shot wall RTF 0.97 -> 0.95, 51 s-reference
  1.08 -> 1.05, 104 s takes 1.03. EOS-termination probe (the failure
  mode an envelope cannot see): 24 utterance pairs INT4 vs INT8, length
  ratios 0.88-1.23, zero flags.
- Packed 4-bit backbone kernels: the group-wise INT4 weights store two
  per byte (`S2P_INT4_PACKED`, default on; the int8 container remains as
  the A/B fallback). The packed GEMV unpacks into the same accumulation
  order against the same f32 scales, so outputs are BIT-IDENTICAL to the
  validated unpacked path — parity JSON exactly equal, and all five
  listening takes reproduce byte-identically (MD5) on the packed server.
  Measured: zero-shot wall RTF 1.10 → 0.97 (first sub-realtime serving
  on a quality-passing path), 51 s-reference 1.19 → 1.08, 104 s takes
  1.20 → 1.05; TTFA unchanged; backbone weights 3.63 → 2.27 GB.
- Group-wise INT4 value precision (`S2P_INT4=1` on top of `S2P_INT8=1`):
  backbone linears quantize to 4-bit symmetric with one f32 scale per 32
  K-elements (`S2P_INT4_GROUP`) and a per-group MSE clip search
  (`S2P_INT4_MSE`, default on); the fast-AR and tied lm-head stay
  per-channel INT8 (mixed precision — the small decoder run 9x per frame
  is the tensor 4-bit damages most; `S2P_INT4_ALL=1` for A/B). Values
  stay in the int8 container: audio equals a packed-INT4 deployment,
  memory/bandwidth stay INT8 until packed kernels land (roadmap). This
  replaces the naive per-channel INT4 listening experiment, which
  audibly muffled from ~10 s (AR compounding of per-step weight noise).
  Parity: prefill AND step-1 argmax identical, fast-AR 8/9 (the known
  bf16 near-tie) — INT8-class discrete decisions; HF-envelope over 104 s
  takes shows no muffling collapse. Ablations: g64 loses step-1 argmax
  (fast-AR 3/9); disabling MSE at g32 costs prefill cos 0.9987 → 0.9907.
- Device-side semantic sampling: the exact two-softmax sampler (RAS detect,
  repetition penalty, torch-order top-30, seeded hash-Gumbel / xoshiro
  draw) runs as a kernel against per-session device state; the fast-AR
  consumes the sampled id device-side and one small download per frame
  replaces the 623 KB logits round-trip. Greedy is bit-identical to the
  host sampler.
- CUDA-graph capture of the steady decode tick (batch <= 4, lazy per batch
  size): one cudaGraphLaunch replays the ~1100-kernel frame; per-frame
  variation flows through pinned-buffer contents. Graph vs eager at the
  same seed: byte-identical audio. Long-context decode 42.4 -> 39.6
  ms/frame; S2P_NO_GRAPHS=1 for A/B.
- Bit-exact incremental streaming DAC (`src/dac/stream_inc.c`), now the
  default streaming path: per-layer K/V histories for the post_module and
  per-conv input histories replace the reference's window/overlap/crossfade
  scheme entirely. Each pushed frame emits its 2048 samples immediately and
  the streamed PCM equals the whole-buffer decode BIT FOR BIT (validated at
  60 and 200 frames, past the 128-frame attention-window eviction point).
  This removes both reference-side streaming defects (20-frame warmup
  against a 128-frame receptive field; 512 dropped timeline samples per
  window) by construction. `S2P_STREAM_REFERENCE=1` restores the reference
  scheme for A/B.
- DAC software pipelining in the scheduler: frame N's vocoding runs on the
  dedicated CUDA stream while frame N+1's backbone step runs on the model
  stream (`s2p_dac_stream_push_async`/`_collect`). Overlap removes the
  launch/sync serialization but NOT the DAC's GPU work — both streams share
  the same SMs — so the engine additionally batches up to eight frames per
  push (`S2P_STREAM_BATCH`, first push always one frame for TTFA): kernel
  launches drop 8x and the small early-stage convs get real occupancy.
  Measured server-side after the full pass (streamed, INT8, graphs +
  device sampling + batched DAC + flash attention): wall RTF 2.05 → 1.19
  with a 51 s voice reference (zero-shot long text: 1.10), TTFA 0.48 s
  zero-shot / 1.77 s with the reference block.
- Split-K flash-decode attention (`s2pk_attention_decode`): K/V tiles staged
  in shared memory coalesced, 4 GQA q-heads served per tile, per-split
  partials merged log-sum-exp. Long-context decode (51 s reference, 1445
  prompt tokens) drops 46.9 → 42.7 ms/frame; parity unchanged
  (`S2P_ATTN_LEGACY=1` keeps the single-pass kernel for decode).
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

- The all-INT4 configuration (`S2P_INT4_ALL=1`) requires a QAT-patched
  checkpoint; training runs are in progress
  ([docs/QAT-RUNS.md](docs/QAT-RUNS.md)). Without it, the fast-AR and
  tied lm-head stay per-channel INT8 (measured necessity, not caution).
- Encoder output has no oracle-fixture parity comparison yet (validated
  through cloning fidelity in listening so far).
