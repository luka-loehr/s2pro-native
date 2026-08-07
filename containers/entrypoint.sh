#!/usr/bin/env bash
# s2pro-native container entrypoint: prepare /data on first start, then serve.
#
# /data layout (persistent volume):
#   model/       S2-Pro checkpoint (downloaded on first start; Fish Audio
#                Research License — non-commercial without a Fish license)
#   codec-full/  converted codec artifact (built on first start)
#   qcache/      prequantized-weight sidecar (written by the server)
#   voices/      OPTIONAL named-voice registry (<name>.wav + <name>.txt)
#   qat_patch.safetensors  OPTIONAL QAT patch; if present it is applied
#                to a copy of the checkpoint and served (all-INT4 stream)
#
# Environment: HF_TOKEN (only if the HF repo requires auth), S2P_TOKEN
# (bearer auth for the API), plus every S2P_* engine variable.
set -euo pipefail

DATA=/data
MODEL="$DATA/model"
CODEC="$DATA/codec-full"
mkdir -p "$DATA" "$DATA/qcache"

if [ ! -f "$MODEL/config.json" ]; then
    echo "[entrypoint] first start: downloading the S2-Pro checkpoint"
    echo "[entrypoint] (Fish Audio Research License — non-commercial use;"
    echo "[entrypoint]  commercial use requires a license from Fish Audio)"
    scripts/fetch_model.sh "$MODEL"
fi

if [ -f "$DATA/qat_patch.safetensors" ] && [ ! -f "$MODEL-qat/.done" ]; then
    echo "[entrypoint] applying QAT patch -> $MODEL-qat"
    python3 tools/apply_qat_patch.py --model "$MODEL" \
        --patch "$DATA/qat_patch.safetensors" --out "$MODEL-qat" \
    && touch "$MODEL-qat/.done"
fi
if [ -f "$MODEL-qat/.done" ]; then
    MODEL="$MODEL-qat"
fi

if [ ! -f "$CODEC/codec.idx" ]; then
    echo "[entrypoint] first start: converting the codec artifact (~1 min)"
    python3 tools/convert_codec_full.py --model "$DATA/model" --out "$CODEC"
fi

ARGS=(--model-dir "$MODEL" --codec-dir "$CODEC"
      --port "${S2P_PORT:-8010}" --bind "${S2P_BIND:-0.0.0.0}")
if [ -d "$DATA/voices" ]; then
    ARGS+=(--voices-dir "$DATA/voices")
fi
if [ -n "${S2P_TOKEN:-}" ]; then
    ARGS+=(--token "$S2P_TOKEN")
fi

echo "[entrypoint] starting s2pro-server ${ARGS[*]}"
exec /app/s2pro-server "${ARGS[@]}"
