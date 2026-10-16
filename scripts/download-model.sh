#!/usr/bin/env bash
# Download a GGUF model with resume support.
#
# Usage: download-model.sh [URL] [DEST_FILENAME]
#
# Qwen3-8B-Instruct Q4_K_M preset (A/B test candidate):
#   ./scripts/download-model.sh \
#     "https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf" \
#     Qwen3-8B-Instruct-Q4_K_M.gguf
set -euo pipefail

MODEL_URL="${1:-https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf}"
MODELS_DIR="${ASSISTANT_MODELS_DIR:-/mnt/Vault/models}"
DEST_FILENAME="${2:-${MODEL_URL##*/}}"

# The official Qwen repo re-sharded the q4_k_m quant into two parts, so the
# original single-file URL 404s. These mirrors still carry the single file.
FALLBACK_URLS=(
    "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    "https://huggingface.co/lmstudio-community/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf"
)

mkdir -p "$MODELS_DIR"
DEST="$MODELS_DIR/$DEST_FILENAME"

for url in "$MODEL_URL" "${FALLBACK_URLS[@]}"; do
    echo "==> URL:     $url"
    echo "==> Target:  $DEST"
    echo "==> Resuming partial downloads if a .part file exists."
    if curl -L --fail --retry 5 --retry-delay 10 --continue-at - -o "$DEST.part" "$url"; then
        mv "$DEST.part" "$DEST"
        SIZE="$(du -h "$DEST" | cut -f1)"
        echo "==> Download complete: $DEST ($SIZE)"
        exit 0
    fi
    echo "==> This URL failed; trying the next one." >&2
done

echo "ERROR: all download URLs failed." >&2
exit 1
