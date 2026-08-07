# syntax=docker/dockerfile:1
# s2pro-native — self-contained serving image (linux/arm64, DGX Spark).
#
# Build (on an sm_121 host):
#   docker build -f containers/Dockerfile -t ghcr.io/luka-loehr/s2pro-native:1.0.0 .
#
# Run (two commands, see README §Quickstart):
#   docker pull ghcr.io/luka-loehr/s2pro-native:1.0.0
#   docker run --gpus all -p 8010:8010 -v s2pro-data:/data \
#       ghcr.io/luka-loehr/s2pro-native:1.0.0
#
# The S2-Pro checkpoint is NOT in the image (Fish Audio Research License);
# the entrypoint downloads it into the /data volume on first start and
# converts the codec artifact there. Every later start is warm (~80 s).

ARG CUDA_TAG=13.0.3

# ---- stage 1: fish-scales-ops objects --------------------------------------
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS fso
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work
COPY scripts/build_fso.sh scripts/build_fso.sh
RUN git clone --depth 1 https://github.com/fishaudio/fish-scales-ops.git \
        /work/3rdparty/fish-scales-ops \
    && ARCH=121a FSO_DIR=/work/3rdparty/fish-scales-ops \
       FSO_OUT=/work/3rdparty/build scripts/build_fso.sh

# ---- stage 2: engine build --------------------------------------------------
FROM nvidia/cuda:${CUDA_TAG}-devel-ubuntu24.04 AS build
WORKDIR /work
COPY Makefile ./
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
COPY --from=fso /work/3rdparty/build /fso/build
COPY --from=fso /work/3rdparty/fish-scales-ops /fso/fish-scales-ops
RUN make -j"$(nproc)" nvcc-build FSO_DIR=/fso/fish-scales-ops \
        FSO_OBJS="/fso/build/runner_121a.o /fso/build/quant_121a.o" \
    && strip --strip-unneeded build/s2pro-server

# ---- stage 3: runtime -------------------------------------------------------
FROM nvidia/cuda:${CUDA_TAG}-runtime-ubuntu24.04
LABEL org.opencontainers.image.source="https://github.com/luka-loehr/s2pro-native" \
      org.opencontainers.image.description="Native C11+CUDA S2-Pro TTS serving for NVIDIA DGX Spark (sm_121)" \
      org.opencontainers.image.licenses="Apache-2.0"
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    # CPU torch only for the ONE-TIME codec conversion on first start;
    # safetensors + huggingface_hub for fetch and QAT patching
    && pip3 install --break-system-packages --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu \
    && pip3 install --break-system-packages --no-cache-dir \
        safetensors "huggingface_hub[cli]"
WORKDIR /app
COPY --from=build /work/build/s2pro-server ./s2pro-server
COPY scripts/fetch_model.sh scripts/fetch_model.sh
COPY tools/convert_codec_full.py tools/apply_qat_patch.py tools/
COPY containers/entrypoint.sh /entrypoint.sh
COPY LICENSE THIRD_PARTY_NOTICES.md api/openapi.yaml ./
RUN chmod +x /entrypoint.sh scripts/fetch_model.sh

# Serving defaults: all-INT4 weight stream, prequant sidecar into /data,
# encoder kept so per-request voice cloning works without a registry.
ENV S2P_INT8=1 S2P_INT4=1 S2P_INT4_ALL=1 \
    S2P_QCACHE_DIR=/data/qcache S2P_KEEP_ENCODER=1 \
    S2P_PORT=8010 S2P_BIND=0.0.0.0
VOLUME /data
EXPOSE 8010
ENTRYPOINT ["/entrypoint.sh"]
