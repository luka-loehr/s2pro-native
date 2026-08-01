# Third-party notices

This repository contains only first-party code (Apache-2.0, see `LICENSE`).
It **links against** and **loads assets from** the following third-party
components, none of which are vendored in this tree:

## fish-scales-ops (linked at build time)

The FP8 GEMM path compiles against headers from and links objects built from
[fishaudio/fish-scales-ops](https://github.com/fishaudio/fish-scales-ops)
(Apache-2.0, © Fish Audio). fish-scales-ops itself vendors:

- **NVIDIA TensorRT-LLM** block-scale GEMM runner (Apache-2.0, © NVIDIA
  CORPORATION) — the `CutlassFp8BlockScaleGemmRunner` interface used by
  `src/fso/fso_wrap.cpp`.
- **NVIDIA CUTLASS** (BSD-3-Clause, © NVIDIA CORPORATION), as a submodule.

Binaries produced by this repository's build therefore embed code under those
licenses. See `scripts/build_fso.sh` for the exact provenance.

## Fish Audio S2-Pro model (loaded at runtime, NOT distributed here)

Model weights, tokenizer, and configuration are downloaded separately from
[huggingface.co/fishaudio/s2-pro](https://huggingface.co/fishaudio/s2-pro)
(`scripts/fetch_model.sh`) and are licensed under the **Fish Audio Research
License**: free for research and non-commercial use; commercial use requires
a separate license from Fish Audio (business@fish.audio). This repository
intentionally contains no checkpoint-derived files.

## NVIDIA CUDA Toolkit

Built binaries link `cublas`, `cudart`, `nvrtc`, and the CUDA driver, subject
to the NVIDIA CUDA Toolkit EULA.

## Reference implementation (not included)

The PyTorch reference this port was validated against is the SGLang-Omni
day-0 integration of S2-Pro (Apache-2.0). No reference code is included in
this repository; `docs/PORTING.md` documents the port in prose and tables.
