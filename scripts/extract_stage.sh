#!/usr/bin/env bash
# Extract a staged token budget and validate the result.
# Usage: scripts/extract_stage.sh [TOKENS] [OUT_DIR]
set -euo pipefail

ROOT="${ROOT:-/data/ornith-byte}"
TOKENS="${1:-1000000}"
OUT="${2:-$ROOT/targets/train-1m}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
NGL="${NGL:-32}"
CPP="${CPP:-1}"
if [ "$CPP" = "1" ]; then
  EXTRA=(--cpp-targets --threads 8)
else
  EXTRA=(--workers 8)
fi
SHA="$(python3 -c 'import json;print(json.load(open("models.lock"))["teacher"]["files"]["Ornith-1.5-9B-Q5_K_M.gguf"]["sha256"])')"

rm -rf "$OUT"
cd "$HERE"
PYTHONPATH=src "$PY" -m byte_agent.extract \
  --out "$OUT" \
  --split train \
  --limit-tokens "$TOKENS" \
  --model "$ROOT/models/gguf/Ornith-1.5-9B-Q5_K_M.gguf" \
  --teacher-sha "$SHA" \
  --ctx 4096 \
  --batch 64 \
  --ngl "$NGL" \
  --shard-positions 250000 \
  "${EXTRA[@]}"

PYTHONPATH=src "$PY" scripts/validate_shards.py --root "$OUT" --sample 0
