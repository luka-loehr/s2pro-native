![s2pro-native banner](docs/assets/banner.svg)

[![Language](https://img.shields.io/badge/C-C11-00599C?style=flat&logo=c&logoColor=white)](https://en.cppreference.com/w/c/11)
[![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/target-sm__121a%20(DGX%20Spark%20GB10)-1f6feb?style=flat)](docs/SPARK.md)
[![Runtime deps](https://img.shields.io/badge/runtime%20deps-none-2ea043?style=flat)](#architecture)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat)](LICENSE)

A complete, self-contained serving stack for **[Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro)**
text-to-speech, written from scratch in C11 and CUDA C. No Python, no
PyTorch, no serving framework at runtime — the binary loads safetensors,
tokenizes, runs the Dual-AR transformer and the RVQ vocoder, and streams
44.1 kHz audio over a dependency-free HTTP server.

```
              ┌───────────────────────── s2pro-server ─────────────────────────┐
 text ──►  BPE tokenizer + ChatML prompt (C)                                   │
              │         Slow-AR   36L · d2560 · GQA 32/8 · qk-norm  ──┐        │
              │         Fast-AR    4L · 9 residual codebooks         ─┤ 10 codes/frame
              │                                                       ▼        │
 HTTP chunked stream ◄── lockstep scheduler ◄── DAC / Firefly-GAN vocoder ── 44.1 kHz PCM
              └────────────────────────────────────────────────────────────────┘
```

## Status

End-to-end functional on real hardware: loads the 9.1 GB sharded checkpoint,
generates deterministic frames, produces speech-shaped audio through the
full pipeline. Measured on a DGX Spark (GB10, `sm_121`, single stream,
41-token prompt, 60 frames):

| GEMM path | decode ms/frame | RTF (compute ÷ audio) |
|---|---:|---:|
| BF16 cuBLAS (default) | 92.6 | 2.38 |
| FP8 block-scale (`S2P_FP8=1`, fish-scales-ops) | 49.4 | 1.39 |
| bandwidth floor, 8-bit weights (theoretical) | ~35 | ~0.75 |

RTF < 1 means audio is produced faster than it plays. One frame is 2048
samples (46.4 ms); at batch 1 every frame reads ~7.75 B weight parameters
(36-layer backbone + tied LM head + 9 sequential fast-AR passes), so 16-bit
weights are bandwidth-bound above realtime by construction — the roadmap
below is about closing the remaining gap on the 8-bit path.

## Quickstart

Requires Docker with the NVIDIA runtime on an `sm_121` machine (see
[docs/SPARK.md](docs/SPARK.md) for the full walkthrough and exact flags).

```sh
# 1. checkpoint (Fish Audio Research License — not distributed here)
scripts/fetch_model.sh model

# 2. FP8 kernel objects from fish-scales-ops (one-time)
docker run --rm -v "$PWD":/work -w /work nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    bash -c "ARCH=121a FSO_DIR=/work/3rdparty/fish-scales-ops FSO_OUT=/work/3rdparty/build scripts/build_fso.sh"

# 3. build
docker run --rm -v "$PWD":/work -w /work nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    make -j8 FSO_OBJS="/work/3rdparty/build/runner_121a.o /work/3rdparty/build/quant_121a.o"

# 4. smoke test (writes /tmp/s2p_smoke.wav, prints timings)
docker run --rm --gpus all -v "$PWD":/work -w /work -v "$PWD/model":/model:ro \
    -v /path/to/codec:/codec:ro nvidia/cuda:13.0.3-devel-ubuntu24.04 \
    ./build/s2p-test /model /codec

# 5. serve
./build/s2pro-server --model-dir /model --codec-dir /codec --port 8010
curl -X POST localhost:8010/v1/tts -d '{"text":"Hello.","format":"wav"}' -o hello.wav
```

## Architecture

| Module | Code | What it does |
|---|---|---|
| core | `src/core/` | arena JSON parser (12 MB tokenizer.json, single pass), safetensors mmap loader (single + sharded), tensors, config, WAV |
| slow-AR | `src/slowar/` | 36-layer backbone: fused-QKV GQA with per-head qk-norm, bf16-exact RoPE (θ=1e6), SwiGLU 9728, KV cache, lockstep batched decode, reference-exact two-softmax sampling |
| fast-AR | `src/fastar/` | 4-layer residual decoder (untied head, no qk-norm, KV depth 11): 9 greedy codebook steps per frame |
| text | `src/text/` | byte-level BPE (bit-exact vs HF `tokenizers` on a 4000-case fuzz), hand-written ChatML prompt builder with VQ-part injection |
| dac | `src/dac/` | modded-DAC / Firefly-GAN vocoder: RVQ `from_indices`, causal/dilated conv + transposed-conv stacks, snake activations, streaming windows with crossfade |
| serve | `src/sched/` `src/http/` | lockstep scheduler (first-frame priority, cancel, stats), POSIX HTTP/1.1 server with chunked WAV/PCM streaming |
| fso | `src/fso/` | the single C++ TU: extern-C shim over the [fish-scales-ops](https://github.com/fishaudio/fish-scales-ops) FP8 block-scale GEMM (weights quantized once at load, activations per call) |

Contracts between modules are frozen in `include/s2pro/` — the codebase was
written by five parallel agents against those headers in a single pass and
integrated the same day ([CONTRACT.md](CONTRACT.md)).
[docs/PORTING.md](docs/PORTING.md) is the authoritative port spec: every
dimension, the exact sampling algorithm, and the numerically load-bearing
details (bf16 RoPE truncation, two-softmax order, interleaved VQ injection).

## HTTP API

```
GET  /healthz            → 200 {"active_sessions":…,"frames_decoded":…}
POST /v1/tts             → chunked audio stream
     {"text":"…","format":"wav"|"pcm","temperature":…,"top_p":…,"seed":…}
     Authorization: Bearer <token>        (when started with --token)
```

`wav` streams a header followed by S16LE frames as they are generated; a
client can start playback while generation is still running.

## Roadmap to RTF < 1

- [ ] per-frame CUDA graph capture (~500 kernel launches/frame today)
- [ ] device-side sampling (removes a 623 KB D2H + host softmax per frame)
- [ ] DAC on a dedicated stream, overlapped with next-frame decode
- [ ] INT8 per-channel GEMV alternative for batch 1 (crossover vs FP8 measured at M≈4–5)
- [ ] layer-parity validation against the PyTorch reference
- [ ] voice-cloning encode (blocked on encoder tensors in the converted codec, `S2P_GAP`)

## License

Apache-2.0 for everything in this repository ([LICENSE](LICENSE)).
Model weights are **not** included: Fish Audio S2-Pro is under the Fish
Audio Research License (non-commercial; commercial licensing via Fish
Audio). Built binaries link Apache-2.0/BSD-3 NVIDIA components through
fish-scales-ops — see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
