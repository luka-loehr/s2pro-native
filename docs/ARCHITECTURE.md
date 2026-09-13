# Architecture

s2pro-native implements the complete S2-Pro inference path (prompt
preparation, the 36-layer slow-AR backbone, the 4-layer fast-AR residual
decoder, the DAC vocoder, scheduling, and HTTP delivery) in native C and
CUDA. Python, PyTorch, SGLang, and vLLM are not part of the runtime.
Numerical stages are validated against a pure-PyTorch oracle (layer
parity), and optimizations are gated on measurements; rejected approaches
are documented in the reports.

## 1. System

- Native S2-Pro Dual-AR inference with custom CUDA kernels and cuBLAS
  execution on `sm_121`.
- Weight-precision ladder under a strict quality gate: per-channel INT8,
  packed group-wise INT4 backbone (4.5 bits per weight), an FP8 path kept
  for measurement only after failing parity, and QAT self-distillation to
  all-INT4 ([QUANT.md](QUANT.md), [QAT-RUNS.md](QAT-RUNS.md)).
- Incremental streaming vocoder whose streamed PCM matches the
  whole-buffer decode (MD5-checked), with optimized convolution kernels
  ([DAC-KERNELS.md](DAC-KERNELS.md)).
- A byte-level BPE tokenizer that matches the HF `tokenizers` reference
  on a 4,000-case fuzz, and a hand-written ChatML prompt builder with
  VQ-part injection.
- One shared engine with lockstep-batched sessions, first-frame priority,
  cancellation, graceful shutdown, CUDA-graph replay of the decode tick,
  and device-side sampling.
- A multilingual named-voice registry (drop `<name>.wav` + `<name>.txt`
  into `voices/`) plus per-request cloning over HTTP
  ([VOICES.md](VOICES.md)); reference audio is generated per deployment
  by [`tools/voicegen`](../tools/voicegen/README.md), not committed.
- Frozen module contracts (`include/s2pro/`) that allowed the module
  tracks to be built in parallel ([CONTRACT.md](../CONTRACT.md)).

The project targets S2-Pro only: no S2.1-Pro (API-only, no public
weights), no S1, no model framework in the runtime.

### Supported languages and audio

S2-Pro is multilingual (80+ languages) with inline free-form `[bracket]`
prosody and emotion control; language coverage and control quality are
properties of the upstream checkpoint and are not re-qualified per
language here. Audio is emitted as 44,100 Hz mono signed 16-bit
little-endian PCM; each codec frame represents 2,048 samples (~46.4 ms).

## 2. Pipeline

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
    `--> DAC / Firefly-GAN vocoder (CUDA, incremental)
               |
               `-- progressive 44.1 kHz PCM
```

## 3. Source layout

| Path | Contents |
| --- | --- |
| `src/core` | Arena JSON parser, safetensors mmap loader (single + sharded), tensors, config, WAV. |
| `src/slowar` | Shared CUDA primitives, 36-layer backbone, KV cache + KV-prefix cache, lockstep batch decode, reference-exact sampling. |
| `src/fastar` | 4-layer residual decoder: untied head, no qk-norm, KV depth 11. |
| `src/text` | Byte-level BPE tokenizer, ChatML prompt builder, sentence chunker. |
| `src/dac` | RVQ `from_indices`, causal/dilated conv stacks, snake activations, incremental streaming, encoder. |
| `src/sched`, `src/http` | Scheduler, INT8/INT4 GEMV kernels, dependency-free HTTP/1.1 server. |
| `src/voice` | Named-voice registry, DAC-encoded once at startup ([VOICES.md](VOICES.md)). |
| `src/fso` | The single C++ TU: extern-C shim over the fish-scales-ops FP8 GEMM. |
| `tools` | Offline tooling: codec conversion, parity fixtures, QAT trainer + corpus + patcher, voice generator (Rust). Never part of the runtime. |

## 4. Documentation map

| document | class | scope |
| --- | --- | --- |
| [PORTING.md](PORTING.md) | specification | the model, exact algorithms, and fidelity pitfalls of the port |
| [CONTRACT.md](../CONTRACT.md) | specification | frozen module interfaces and ownership |
| [QUANT.md](QUANT.md) | report | weight-quantization ladder, methods, negative results |
| [DAC-KERNELS.md](DAC-KERNELS.md) | report | vocoder kernel optimization, accepted and rejected designs |
| [QAT-RUNS.md](QAT-RUNS.md) | report | QAT distillation runs: conditions, telemetry, results |
| [SERVING.md](SERVING.md) | report | startup cache, launch-overhead result, concurrency limits |
| [SPECULATIVE.md](SPECULATIVE.md) | report | speculative multi-frame decoding: draft model, corpus, training |
| [PERFORMANCE.md](PERFORMANCE.md) | report | headline results and the optimization record |
| [benchmarks/parity/README.md](../benchmarks/parity/README.md) | report | layer-parity protocol and per-path verdicts |
| [GETTING-STARTED.md](GETTING-STARTED.md) | guide | container and from-source setup |
| [SPARK.md](SPARK.md) | guide | build and deployment on the DGX Spark |
| [API.md](API.md) | reference | HTTP endpoints, request fields, chunking, streaming |
| [VOICES.md](VOICES.md) | guide | the voice reference system and the accent constraint |
| [CHANGELOG.md](../CHANGELOG.md) | record | all notable changes with measurement conditions |

## 5. Contributing and security

Read [CONTRIBUTING.md](../CONTRIBUTING.md) before opening a change; it
also defines the documentation scheme this repository follows. Report
vulnerabilities privately per [SECURITY.md](../SECURITY.md). Community
participation is governed by the [Code of Conduct](../CODE_OF_CONDUCT.md);
cite the software via [`CITATION.cff`](../CITATION.cff).
