![s2pro-native banner](docs/assets/banner.svg)

[![Language](https://img.shields.io/badge/C-C11-00599C?style=flat&logo=c&logoColor=white)](https://en.cppreference.com/w/c/11)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/target-sm__121%20(DGX%20Spark%20GB10)-1f6feb?style=flat)](docs/SPARK.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE)

Native C11 + CUDA inference for
[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro)
(OpenAudio S2), designed and qualified for the NVIDIA DGX Spark.

I built the complete inference path — prompt preparation, the 36-layer
slow-AR backbone, the 4-layer fast-AR residual decoder, the DAC vocoder,
scheduling, and HTTP delivery — in native C and CUDA. Python, PyTorch,
SGLang, and vLLM are not part of the runtime. Every numerical stage is
validated against a pure-PyTorch oracle (layer parity), every
optimization is gated on measured evidence, and rejected approaches are
documented with the same rigor as the ones that shipped.

## 1. Results

Measured on one DGX Spark (GB10, `sm_121`), single stream, streamed
serving over HTTP, sampled (temperature 0.8, top-p 0.8, fixed seed). RTF
is compute ÷ audio; below 1.0 means synthesis outruns playback.

| serving path | wall RTF | TTFA |
| --- | ---: | ---: |
| zero-shot | **0.60** | 0.14 s |
| 60 s voice reference (warm voice cache) | **0.58** | 0.20 s |
| ~148 s chunked long-form | **0.58–0.59** | 0.20 s |

Configuration: all-INT4 weight stream (packed group-wise INT4 backbone +
QAT-distilled INT4 fast-AR and lm-head, `S2P_INT4_ALL=1` on a
QAT-patched checkpoint), accepted by critical listening on 2026-08-03.
An unpatched checkpoint serves the INT8-fast-AR configuration at
0.75–0.76. Starting point of the same stack: wall RTF 2.05. The complete
measured path from 2.05 to 0.60 — including the negative results — is
recorded in the technical reports (§7).

> **Status: v0.1.0, the first tagged release.** BF16, INT8, and the
> packed group-wise INT4 backbone pass the layer-parity gate against the
> PyTorch reference; voice cloning, the multilingual voice registry, and
> the HTTP streaming server are exercised end to end on real hardware.
> Deploy from the tag or a reviewed, pinned commit.

## 2. System

- Native S2-Pro Dual-AR inference with custom CUDA kernels and cuBLAS
  execution on real `sm_121` SASS.
- Weight-precision ladder under a strict quality gate: per-channel INT8
  (the workhorse), packed group-wise INT4 backbone (4.5 bits per weight),
  an FP8 path kept for measurement after failing parity, and QAT
  self-distillation to all-INT4 (shipped)
  ([docs/QUANT.md](docs/QUANT.md), [docs/QAT-RUNS.md](docs/QAT-RUNS.md)).
- Bit-exact incremental streaming vocoder — streamed PCM equals the
  whole-buffer decode bit for bit — with GEMM-grade convolution kernels
  ([docs/DAC-KERNELS.md](docs/DAC-KERNELS.md)).
- A byte-level BPE tokenizer that is bit-exact against the HF
  `tokenizers` reference on a 4,000-case fuzz, and a hand-written ChatML
  prompt builder with VQ-part injection.
- One shared engine with lockstep-batched sessions, first-frame priority,
  cancellation, graceful shutdown, CUDA-graph replay of the decode tick,
  and device-side sampling.
- A multilingual named-voice registry (drop `<name>.wav` + `<name>.txt`
  into `voices/`) plus per-request cloning over HTTP
  ([docs/VOICES.md](docs/VOICES.md)); reference audio is generated per
  deployment by [`tools/voicegen`](tools/voicegen/README.md), not
  committed.
- Frozen module contracts (`include/s2pro/`) that allowed the five module
  tracks to be built in parallel and keep the codebase auditable
  ([CONTRACT.md](CONTRACT.md)).

The project intentionally targets S2-Pro only — no S2.1-Pro (API-only, no
public weights), no S1, no model framework in the runtime.

### Supported languages and audio

S2-Pro is multilingual (80+ languages) with inline free-form `[bracket]`
prosody and emotion control; language coverage and control quality are
properties of the upstream checkpoint and are not re-qualified per
language here. Audio is emitted as 44,100 Hz mono signed 16-bit
little-endian PCM; each codec frame represents 2,048 samples (~46.4 ms).

## 3. Architecture

```text
HTTP client
    |
    v
native C HTTP server (validation, auth, chunked streaming, cancellation)
    |
    v
lockstep scheduler (sessions, first-frame priority, backpressure)
    |
    +--> slow-AR backbone, 36L d2560 GQA 32/8 + qk-norm (CUDA/cuBLAS)
    |          |
    |          `-- one semantic token per frame (two-softmax sampling)
    |
    +--> fast-AR residual decoder, 4L (9 greedy codebook steps per frame)
    |          |
    |          `-- 10 RVQ codes per frame
    |
    `--> DAC / Firefly-GAN vocoder (CUDA, bit-exact incremental)
               |
               `-- progressive 44.1 kHz PCM
```

| Path | Contents |
| --- | --- |
| `src/core` | Arena JSON parser, safetensors mmap loader (single + sharded), tensors, config, WAV. |
| `src/slowar` | Shared CUDA primitives, 36-layer backbone, KV cache + KV-prefix cache, lockstep batch decode, reference-exact sampling. |
| `src/fastar` | 4-layer residual decoder: untied head, no qk-norm, KV depth 11. |
| `src/text` | Byte-level BPE tokenizer, ChatML prompt builder, sentence chunker. |
| `src/dac` | RVQ `from_indices`, causal/dilated conv stacks, snake activations, bit-exact incremental streaming, encoder. |
| `src/sched`, `src/http` | Scheduler, INT8/INT4 GEMV kernels, dependency-free HTTP/1.1 server. |
| `src/voice` | Named-voice registry, DAC-encoded once at startup ([docs/VOICES.md](docs/VOICES.md)). |
| `src/fso` | The single C++ TU: extern-C shim over the fish-scales-ops FP8 GEMM. |
| `tools` | Offline tooling: codec conversion, parity fixtures, QAT trainer + corpus + patcher, voice generator (Rust). Never part of the runtime. |

## 4. Quickstart

Requires Docker with the NVIDIA runtime on an `sm_121` machine; the host
needs no CUDA toolkit, Python, or PyTorch. Full walkthrough:
[docs/SPARK.md](docs/SPARK.md).

```bash
# 1. checkpoint (Fish Audio Research License — not distributed here)
scripts/fetch_model.sh model

# 2. fish-scales-ops objects (one-time)
docker run --rm -v "$PWD":/work -w /work nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    bash -c "ARCH=121a FSO_DIR=/work/3rdparty/fish-scales-ops FSO_OUT=/work/3rdparty/build scripts/build_fso.sh"

# 3. build
docker run --rm -v "$PWD":/work -w /work nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    make -j8 FSO_OBJS="/work/3rdparty/build/runner_121a.o /work/3rdparty/build/quant_121a.o"

# 4. smoke test (writes /tmp/s2p_smoke.wav, prints timings and RTF)
docker run --rm --gpus all -v "$PWD":/work -w /work -v "$PWD/model":/model:ro \
    -v /path/to/codec:/codec:ro nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    ./build/s2p-test /model /codec

# 5. serve
./build/s2pro-server --model-dir /model --codec-dir /codec --port 8010
curl -X POST localhost:8010/v1/tts -d '{"text":"Hello.","format":"wav"}' -o hello.wav
```

## 5. HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /healthz` | Engine and scheduler statistics. |
| `GET /v1/voices` | Named-voice registry listing. |
| `POST /v1/tts` | Chunked streaming synthesis (WAV or raw PCM). |

`POST /v1/tts` accepts `{"text", "format": "wav"|"pcm", "temperature",
"top_p", "seed", "stream", "chunk_length", "chunk_gap_ms", "voice",
"reference_audio_b64", "reference_text"}`; with `--token` set, requests
require `Authorization: Bearer <token>`.

**Voice selection.** `voice` selects a pre-encoded registry voice;
`reference_audio_b64` + `reference_text` clone a per-request wav (max
15 s) on the fly; neither selects zero-shot (a random, unpinned speaker).
Mixed-language text is one generation — voices are multilingual by
construction ([docs/VOICES.md](docs/VOICES.md)).

**Long-form chunking.** Voiced requests are served as a chunk chain: the
text splits at sentence boundaries into chunks that close after
`chunk_sentences` sentences (default 2, env `S2P_CHUNK_SENTENCES`) or
`chunk_length` bytes (default 300, env `S2P_CHUNK_BYTES`), whichever
comes first, and each chunk generates against the fresh voice reference,
all audio in one response. The measured reason: single-shot prosody
flattens — punctuation pauses per 10 s bucket decay from ~0.5–1.0 s early
to ~0–0.2 s after ~40 s at every weight precision — while chunked
generation holds the opening-quality prosody across the whole take (2–5
pauses per bucket through 130+ s; the same text runs ~26 % longer because
the rushing is gone). Zero-shot requests never chunk (each chunk would
draw a new voice).

**Join normalization.** Boundary silence at chunk joins is trimmed on
both sides and replaced by exactly `chunk_gap_ms` of silence (default
1000, env `S2P_CHUNK_GAP_MS`; `0` = raw concatenation). The model's own
sentence pauses inside a chunk are untouched; only the stitched joins get
a deterministic, runtime-tunable pause.

**Streaming semantics.** `wav` streams a header followed by S16LE frames
as they are generated; playback can start while generation runs. A
streamed WAV necessarily advertises saturated RIFF sizes — the header is
on the wire before the first frame is sampled, and chunked transfer
cannot rewrite sent bytes. Tolerant players handle that; strict ones
(Apple's, notably) refuse it. `"stream": false` buffers server-side and
responds with exact `Content-Length` and RIFF sizes — a well-formed file
at the cost of time-to-first-audio.

The server binds loopback by default and does not terminate TLS. Public
deployments must place it behind an authenticated, rate-limited proxy.
Review [SECURITY.md](SECURITY.md) before exposing the service.

## 6. Performance

Why weight precision dominates: at batch 1 every frame reads ~7.75 B
weight parameters (backbone once, tied lm-head once, nine sequential
fast-AR passes), so decode is bandwidth-bound and 16-bit weights sit
above realtime on this memory system by construction (~220 GB/s
sustained). The full analysis, per-scheme measurements, and ablations are
in [docs/QUANT.md](docs/QUANT.md).

Engine-level decode (single stream, 67-token prompt, 60 frames):

| GEMM path | Prefill | Decode per frame | RTF (compute ÷ audio) |
| --- | ---: | ---: | ---: |
| BF16 cuBLAS (default) | 306.6 ms | 93.3 ms | 2.41 |
| INT8 weight-only (`S2P_INT8=1`) | 353.4 ms | **39.7 ms** | **1.27** |

Serving configuration (`S2P_INT8=1 S2P_INT4=1 S2P_INT4_ALL=1`, QAT-
patched checkpoint): packed group-wise INT4 backbone (4.5 bits per
weight, bit-identical to the unpacked container), QAT-distilled INT4
fast-AR and lm-head (untrained 4-bit collapses the fast-AR's argmax
cascade — the distillation is what makes this tier possible), GEMM-grade
DAC kernels (2.1 ms/frame, bit-identical PCM), per-voice KV-prefix cache
(~189 MB per voice, LRU). Result: the wall-RTF table in §1. Process
memory: ~8.5 GB (INT8) vs ~12 GB (BF16); backbone weights 2.04 GB
packed.

Numeric fidelity is gated by the layer-parity protocol
([benchmarks/parity](benchmarks/parity/README.md)): BF16 and INT8 pass at
every stage (backbone cos ≥ 0.99989, prefill/step-1 argmax identical,
native DAC at SNR 65.9 dB); the packed INT4 backbone holds the same
argmax class; FP8 fails (backbone cos collapses to 0.33 over 36 layers)
and is retained for memory-system measurements only.

## 7. Optimization record

Each step shipped only after its gate (parity, MD5 bit-exactness, or the
audio battery); the reports own the details, including what lost.

| step | measured effect |
| --- | --- |
| layer-parity validation vs the PyTorch reference | BF16 PASS, INT8 PASS, FP8 FAIL |
| INT8 per-channel weight-only GEMV | decode 93.3 → 39.7 ms/frame; memory ~12 → ~8.5 GB |
| DAC on a dedicated CUDA stream + batched pushes | wall RTF 2.05 → 1.19 (51 s reference) |
| split-K flash-decode attention | long-context decode 46.9 → 42.7 ms/frame |
| device-side sampling (exact two-softmax port) | removes the 623 KB per-frame logits round-trip |
| CUDA-graph replay of the decode tick | 42.4 → 39.6 ms/frame; graph vs eager byte-identical |
| bit-exact incremental streaming DAC | streamed PCM ≡ whole-buffer decode, bit for bit |
| group-wise INT4 backbone (g32 + MSE clip search) | INT8-class parity decisions; rescued naive INT4's audible failure |
| packed nibbles + f16 group scales (4.5 bpw) | zero-shot wall RTF 1.10 → 0.95, bit-identical to unpacked |
| sentence chunking + join normalization | long-form prosody holds through 130+ s (was flattening at ~40 s) |
| per-voice KV-prefix cache | voice-ref TTFA 1.58 → 0.23 s; every path below realtime |
| GEMM-grade DAC conv kernels | DAC 15.1 → 2.1 ms/frame; wall RTF 0.75–0.76 |
| QAT self-distillation of the fast-AR → all-INT4 | weight stream 6.17 → 4.37 GB/frame; wall RTF 0.60–0.63; accepted by listening |
| prequantized-weight sidecar cache | server start 27.7 → 5.1 s, bit-identical by construction |
| INT8 KV cache (g32 per head vector) | KV memory/traffic halved; parity holds the INT8 class exactly |
| INT8 embedding lookups (bf16 table dropped) | −0.8 GB; prefill/step-1 argmax unchanged |
| FP16 vocoder weights | codec memory 1.6 → 0.75 GB; 68.1 dB vs the f32 decode |
| fast-AR launch fusion | measured OUT: graphs already amortize launches ([docs/SERVING.md](docs/SERVING.md)) |

The precision ladder is complete: every module now serves at 4 bits
(backbone by construction, fast-AR and lm-head by overnight
self-distillation — runs, gates, and telemetry in
[docs/QAT-RUNS.md](docs/QAT-RUNS.md)), the KV cache and embeddings at
8, the vocoder at 16. Serving infrastructure and concurrency behavior
(4 streams under the per-stream RTF < 1 rule; the levers for more) are
measured in [docs/SERVING.md](docs/SERVING.md). Remaining engineering
is incremental: cross-session DAC batching, chunked prefill, a
norm+GEMV megakernel, further distillation rounds if listening ever
demands them, and a published release line.

## 8. Documentation map

| document | class | scope |
| --- | --- | --- |
| [docs/PORTING.md](docs/PORTING.md) | specification | the model, exact algorithms, and every fidelity pitfall of the port |
| [CONTRACT.md](CONTRACT.md) | specification | frozen module interfaces and ownership |
| [docs/QUANT.md](docs/QUANT.md) | report | weight-quantization ladder, methods, negative results |
| [docs/DAC-KERNELS.md](docs/DAC-KERNELS.md) | report | vocoder kernel optimization, winning and losing designs |
| [docs/QAT-RUNS.md](docs/QAT-RUNS.md) | report | QAT distillation runs: conditions, telemetry, results |
| [docs/SERVING.md](docs/SERVING.md) | report | startup cache, launch-overhead result, concurrency limits |
| [benchmarks/parity/README.md](benchmarks/parity/README.md) | report | layer-parity protocol and per-path verdicts |
| [docs/SPARK.md](docs/SPARK.md) | guide | build and deployment on the DGX Spark |
| [docs/VOICES.md](docs/VOICES.md) | guide | the voice reference system and the accent constraint |
| [CHANGELOG.md](CHANGELOG.md) | record | all notable changes with measurement conditions |

## 9. Contributing, security, license

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change — it also
defines the documentation scheme this repository follows. Report
vulnerabilities privately per [SECURITY.md](SECURITY.md). Community
participation is governed by the [Code of Conduct](CODE_OF_CONDUCT.md);
cite the software via [`CITATION.cff`](CITATION.cff).

The application source is licensed under the
[Apache License 2.0](LICENSE). Model weights are not included: Fish Audio
S2-Pro is licensed by its upstream publisher under the Fish Audio
Research License (non-commercial; commercial licensing via Fish Audio).
Built binaries link Apache-2.0 and BSD-3-Clause NVIDIA components through
fish-scales-ops — see the
[third-party notices](THIRD_PARTY_NOTICES.md).
