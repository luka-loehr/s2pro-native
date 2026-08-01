![s2pro-native banner](docs/assets/banner.svg)

[![Language](https://img.shields.io/badge/C-C11-00599C?style=flat&logo=c&logoColor=white)](https://en.cppreference.com/w/c/11)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/target-sm__121%20(DGX%20Spark%20GB10)-1f6feb?style=flat)](docs/SPARK.md)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE)

Native C and CUDA inference for
[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro) (OpenAudio S2),
designed and qualified for NVIDIA DGX Spark.

The project turns text into progressive 44.1 kHz mono PCM. The complete
inference path—prompt preparation, the 36-layer slow-AR backbone, the
4-layer fast-AR residual decoder, the DAC vocoder, scheduling, and HTTP
delivery—runs in native C and CUDA. Python, PyTorch, SGLang, and vLLM are
not part of the runtime.

## At a glance

Measured on one DGX Spark (GB10, `sm_121`), single stream, 41-token prompt,
60 frames, greedy sampling (details in [Performance](#performance)):

|  | BF16 cuBLAS (default) | FP8 fish-scales-ops (`S2P_FP8=1`) |
| --- | ---: | ---: |
| Decode per frame (46.4 ms of audio) | 92.6 ms | **49.4 ms** |
| RTF (compute ÷ audio, lower is better) | 2.38 | **1.39** |

The 8-bit weight path's bandwidth floor on this hardware is ~35 ms/frame
(RTF ~0.75); the [roadmap](#roadmap-to-rtf--1) tracks closing the remaining
gap to below-realtime synthesis.

> **Status: functional pre-release.** The full pipeline runs end to end on
> the real 9.1 GB checkpoint and produces deterministic, speech-shaped
> audio. Layer parity against the PyTorch reference is not yet validated,
> and single-stream synthesis is not yet below realtime. There is no
> published release line yet; deploy from a reviewed, pinned `main` commit.

## What this project provides

- Native S2-Pro Dual-AR inference with custom CUDA kernels and cuBLAS
  execution on real `sm_121` SASS.
- An FP8 block-scale GEMM path through
  [fish-scales-ops](https://github.com/fishaudio/fish-scales-ops), weights
  quantized once at load, verified numerically on this GPU (cos 0.9993 vs
  BF16).
- Incremental frame generation and streaming vocoder decode without a Python
  or framework sidecar.
- A byte-level BPE tokenizer that is bit-exact against the HF `tokenizers`
  reference on a 4,000-case fuzz, and a hand-written ChatML prompt builder
  with VQ-part injection.
- One shared engine with lockstep-batched sessions, first-frame priority,
  cancellation, and graceful shutdown.
- Frozen module contracts (`include/s2pro/`) that let independent
  contributors build against stable interfaces ([CONTRACT.md](CONTRACT.md)).

The project intentionally targets **S2-Pro only**. It does not include
S2.1-Pro (API-only, no public weights), S1, or any model framework in the
runtime. Voice-cloning encode is scaffolded but disabled until the codec
converter retains encoder tensors.

## Supported languages and audio

S2-Pro is multilingual (80+ languages) with inline free-form `[bracket]`
prosody and emotion control; language coverage and control quality are
properties of the upstream checkpoint, not re-qualified per language here.

Audio is emitted as 44,100 Hz, mono, signed 16-bit little-endian PCM. Each
codec frame represents 2,048 samples, or ~46.4 ms; streaming delivery uses
overlapping vocoder windows with crossfade.

## Architecture

```text
HTTP client
    |
    v
native C HTTP server (validation, auth, chunked streaming, cancellation)
    |
    v
lockstep scheduler (sessions, first-frame priority, backpressure)
    |
    +--> slow-AR backbone, 36L d2560 GQA 32/8 + qk-norm (CUDA/cuBLAS/FP8)
    |          |
    |          `-- one semantic token per frame (two-softmax sampling)
    |
    +--> fast-AR residual decoder, 4L (9 greedy codebook steps per frame)
    |          |
    |          `-- 10 RVQ codes per frame
    |
    `--> DAC / Firefly-GAN vocoder (CUDA)
               |
               `-- progressive 44.1 kHz PCM
```

| Path | Contents |
| --- | --- |
| `src/core` | Arena JSON parser, safetensors mmap loader (single + sharded), tensors, config, WAV. |
| `src/slowar` | Shared CUDA primitives, 36-layer backbone, KV cache, lockstep batch decode, reference-exact sampling. |
| `src/fastar` | 4-layer residual decoder: untied head, no qk-norm, KV depth 11. |
| `src/text` | Byte-level BPE tokenizer and ChatML prompt builder. |
| `src/dac` | RVQ `from_indices`, causal/dilated conv stacks, snake activations, streaming crossfade. |
| `src/sched`, `src/http` | Scheduler and dependency-free HTTP/1.1 server. |
| `src/fso` | The single C++ TU: extern-C shim over the fish-scales-ops FP8 GEMM. |
| `docs` | Porting spec (`PORTING.md`), Spark build guide (`SPARK.md`). |
| `scripts` | Checkpoint fetch, fish-scales-ops object build. |
| Root community files | Contribution, security, conduct, changelog, citation, and Apache-2.0 license policies. |

## Quickstart

Requires Docker with the NVIDIA runtime on an `sm_121` machine; the host
needs no CUDA toolkit, Python, or PyTorch. See [docs/SPARK.md](docs/SPARK.md)
for the full walkthrough.

```bash
# 1. checkpoint (Fish Audio Research License — not distributed here)
scripts/fetch_model.sh model

# 2. fish-scales-ops objects for the FP8 path (one-time)
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

## HTTP API

| Method and path | Purpose |
| --- | --- |
| `GET /healthz` | Engine and scheduler statistics. |
| `POST /v1/tts` | Chunked streaming synthesis (WAV or raw PCM). |

`POST /v1/tts` accepts `{"text": "…", "format": "wav"|"pcm", "temperature",
"top_p", "seed"}` and, when the server is started with `--token`, requires
`Authorization: Bearer <token>`. `wav` streams a header followed by S16LE
frames as they are generated; a client can start playback while generation
is still running.

The server binds loopback by default and does not terminate TLS. Public
deployments must place it behind an authenticated, rate-limited proxy.
Review [SECURITY.md](SECURITY.md) before exposing the service.

## Performance

Single stream, 41-token prompt, 60 frames, greedy sampling, container on an
otherwise idle GB10:

| GEMM path | Prefill | Decode per frame | DAC per frame | RTF |
| --- | ---: | ---: | ---: | ---: |
| BF16 cuBLAS (default) | 307.7 ms | 92.6 ms | ~13 ms | 2.38 |
| FP8 fish-scales-ops | 40.4 ms | 49.4 ms | ~14 ms | 1.39 |

RTF is compute ÷ audio; below 1.0 means synthesis is faster than playback.
At batch 1 every frame reads ~7.75 B weight parameters (backbone, tied LM
head, and nine sequential fast-AR passes), so 16-bit weights are
bandwidth-bound above realtime on this memory system by construction — the
8-bit floor is ~35 ms/frame. These are smoke-level measurements: generation
is deterministic and audio is speech-shaped, but output quality is not yet
parity-validated against the PyTorch reference, and FP8 audio quality in
particular is unvalidated.

## Roadmap to RTF < 1

- [ ] per-frame CUDA graph capture (~500 kernel launches/frame today)
- [ ] device-side sampling (removes a 623 KB D2H copy + host softmax per frame)
- [ ] DAC on a dedicated stream, overlapped with next-frame decode
- [ ] INT8 per-channel GEMV alternative for batch 1 (crossover vs FP8 measured at M≈4–5)
- [ ] layer-parity validation against the PyTorch reference
- [ ] voice-cloning encode (blocked on encoder tensors in the converted codec, `S2P_GAP`)

## Contributing and security

Read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a change.
Documentation must be written in English, performance claims must state
their measurement conditions, model weights must never enter Git, and shared
branches must never be force-pushed.

Report vulnerabilities privately according to [SECURITY.md](SECURITY.md).
Community participation is governed by the
[Code of Conduct](CODE_OF_CONDUCT.md). Use [`CITATION.cff`](CITATION.cff)
when citing the software.

## License and model provenance

The application source is licensed under the
[Apache License 2.0](LICENSE). Model weights are not included: Fish Audio
S2-Pro is licensed by its upstream publisher under the Fish Audio Research
License (non-commercial; commercial licensing via Fish Audio). Built
binaries link Apache-2.0 and BSD-3-Clause NVIDIA components through
fish-scales-ops. See the
[third-party notices](THIRD_PARTY_NOTICES.md).
