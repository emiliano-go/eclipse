#!/usr/bin/env bash
# Fetch pinned teacher artifacts. All weights live outside the repository.
set -euo pipefail

ROOT="${ROOT:-/data/ornith-byte}"
BASE_REPO="ornith-ai/Ornith-1.5-9B"
BASE_REV="489cb97981b8654bcfcf30ce1f94ed1b62e07b53"
GGUF_REPO="ornith-ai/Ornith-1.5-9B-GGUF"
GGUF_REV="abdd624b12ebf020b767fff532ff44fe552b28c3"

mkdir -p "$ROOT/models/ornith" "$ROOT/models/gguf"

hf download "$BASE_REPO" \
  config.json tokenizer.json tokenizer_config.json \
  --revision "$BASE_REV" \
  --local-dir "$ROOT/models/ornith"

hf download "$GGUF_REPO" \
  Ornith-1.5-9B-Q5_K_M.gguf Ornith-1.5-9B-Q8_0.gguf \
  --revision "$GGUF_REV" \
  --local-dir "$ROOT/models/gguf"

find "$ROOT/models" -type f -printf '%s %p\n' | sort -k2
