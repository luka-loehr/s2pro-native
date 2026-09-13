![s2pro-native banner](docs/assets/banner.svg)

# s2pro-native – Native C and CUDA runtime for Fish Audio S2-Pro

[![C](https://img.shields.io/badge/C-C11-00599C?style=flat&logo=c&logoColor=white)](https://en.cppreference.com/w/c/11) [![CUDA](https://img.shields.io/badge/CUDA-13.0-76B900?style=flat&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit) [![Platform](https://img.shields.io/badge/Platform-DGX%20Spark%20(sm__121)-1f6feb?style=flat)](docs/SPARK.md) [![License](https://img.shields.io/badge/License-Apache--2.0-orange?style=flat)](LICENSE)

**s2pro-native** is a streaming text-to-speech runtime for [Fish Audio S2-Pro](https://huggingface.co/fishaudio/s2-pro), written in C11 and CUDA for the NVIDIA DGX Spark. Python, PyTorch, SGLang, and vLLM are not part of the runtime.

---

## Features

- **Full native path** tokenizer, 36-layer slow-AR backbone, fast-AR decoder, DAC vocoder, scheduler, and HTTP server in C and CUDA
- **Faster than realtime** wall RTF ~0.51 and 0.14 s time-to-first-audio on one DGX Spark ([measurements](docs/PERFORMANCE.md))
- **Quantization under a parity gate** INT8, packed group-wise INT4, and QAT-distilled all-INT4, checked against a PyTorch reference
- **Streaming HTTP API** chunked WAV or raw PCM, bearer auth, cancellation, long-form chunking
- **Voices** multilingual named-voice registry plus per-request voice cloning
- **Lockstep batching** shared engine with CUDA-graph decode and device-side sampling
- **Container image** checkpoint downloaded on first start, never baked into the image

---

## Quick start

```bash
docker pull ghcr.io/luka-loehr/s2pro-native:1.1.0
docker run --gpus all -p 8010:8010 -v s2pro-data:/data ghcr.io/luka-loehr/s2pro-native:1.1.0
curl -X POST localhost:8010/v1/tts -d '{"text":"Hello.","format":"wav"}' -o hello.wav
```

Building from source: [docs/GETTING-STARTED.md](docs/GETTING-STARTED.md).

---

## Documentation

- [Getting started](docs/GETTING-STARTED.md) – container and from-source setup
- [DGX Spark guide](docs/SPARK.md) – build and deployment walkthrough
- [HTTP API](docs/API.md) – endpoints, request fields, chunking, streaming
- [Architecture](docs/ARCHITECTURE.md) – pipeline, source layout, full documentation map
- [Performance](docs/PERFORMANCE.md) – results and optimization record
- [Quantization](docs/QUANT.md) and [layer parity](benchmarks/parity/README.md) – precision ladder and verdicts
- [Voices](docs/VOICES.md) – voice registry and cloning
- [Contributing](CONTRIBUTING.md), [Security](SECURITY.md), [Changelog](CHANGELOG.md)

---

## License

Apache License 2.0 - [View License](LICENSE)  
Model weights are not included: Fish Audio S2-Pro is licensed by Fish Audio under the Fish Audio Research License (non-commercial; commercial licensing via Fish Audio). Bundled NVIDIA components: [third-party notices](THIRD_PARTY_NOTICES.md).

---

## Support

- [Report bugs](https://github.com/luka-loehr/s2pro-native/issues)  
- [luka@lukaloehr.com](mailto:luka@lukaloehr.com)  

---

Developed by [Luka Löhr](https://github.com/luka-loehr)
