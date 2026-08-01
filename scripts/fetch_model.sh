#!/usr/bin/env bash
# Download the Fish Audio S2-Pro checkpoint (weights + tokenizer + config)
# into ./model. Not distributed with this repository — the checkpoint is
# licensed under the Fish Audio Research License (non-commercial; commercial
# use requires a license from Fish Audio).
#
# Requires: huggingface-cli (pip install -U huggingface_hub) or hf.
# Usage: scripts/fetch_model.sh [TARGET_DIR]   (default: ./model)
set -euo pipefail

TARGET="${1:-model}"
REPO="fishaudio/s2-pro"

if command -v hf >/dev/null 2>&1; then
    HF=hf
elif command -v huggingface-cli >/dev/null 2>&1; then
    HF=huggingface-cli
else
    echo "error: install the Hugging Face CLI first: pip install -U huggingface_hub" >&2
    exit 1
fi

mkdir -p "$TARGET"
"$HF" download "$REPO" \
    --include "config.json" "tokenizer.json" "tokenizer_config.json" \
              "special_tokens_map.json" "chat_template.jinja" \
              "model.safetensors.index.json" "model-*.safetensors" \
    --local-dir "$TARGET"

echo "checkpoint in $TARGET:"
ls -lh "$TARGET"
echo
echo "note: the DAC codec (codec.pth) needs conversion to codec.bin/codec.idx"
echo "for the native loader; see docs/SPARK.md."
