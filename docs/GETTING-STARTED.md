# Getting started

## Container

```bash
docker pull ghcr.io/luka-loehr/s2pro-native:1.1.0
docker run --gpus all -p 8010:8010 -v s2pro-data:/data \
    ghcr.io/luka-loehr/s2pro-native:1.1.0
```

On first start the entrypoint downloads the S2-Pro checkpoint into the
`s2pro-data` volume (Fish Audio Research License, non-commercial use;
the checkpoint is never part of the image) and converts the codec
artifact; every later start is warm. Optional: place a named-voice
registry under `voices/` in the volume, a `qat_patch.safetensors` for
the all-INT4 fast-AR, set `S2P_TOKEN` for bearer auth, `HF_TOKEN` if
the checkpoint repo requires it. The image is built from
[`containers/Dockerfile`](../containers/Dockerfile) on an `sm_121` host.

Deploy from a release tag (latest: `v1.1.0`) or a reviewed, pinned
commit.

## From source

Requires Docker with the NVIDIA runtime on an `sm_121` machine; the host
needs no CUDA toolkit, Python, or PyTorch. Full walkthrough:
[SPARK.md](SPARK.md).

```bash
# 1. checkpoint (Fish Audio Research License, not distributed here)
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

`make syntax` runs a compile-only pass and `make selftest` the host-side
JSON, tokenizer, and voice-cache self-tests. See [API.md](API.md) for the
request format.
