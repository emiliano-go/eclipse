#!/usr/bin/env bash
# Run the matched EOT / Marginalize-It / byte-CE comparison.
# Usage: scripts/train_compare.sh [TARGET_DIR] [STEPS] [BATCH] [OUT_ROOT]
set -euo pipefail

ROOT="${ROOT:-/data/ornith-byte}"
TARGETS="${1:-$ROOT/targets/train-1m}"
STEPS="${2:-1500}"
BATCH="${3:-8}"
OUT_ROOT="${4:-$ROOT/runs/1m}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"

mkdir -p "$OUT_ROOT"
cd "$HERE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
VAL_ARGS=()
if [ -n "${VAL_TARGETS:-}" ]; then
  VAL_ARGS=(--val-targets "$VAL_TARGETS")
fi

for OBJ in eot mi ce; do
  echo "=== $OBJ"
  rm -rf "$OUT_ROOT/$OBJ"
  PYTHONPATH=src "$PY" -m byte_agent.train \
    --targets "$TARGETS" \
    --out "$OUT_ROOT/$OBJ" \
    --objective "$OBJ" \
    --steps "$STEPS" \
    --batch "$BATCH" \
    --seq-len 1024 \
    --val-fraction 0.1 \
    "${VAL_ARGS[@]}" \
    --eval-every 100 \
    --save-every 500 \
    > "$OUT_ROOT/$OBJ.log" 2>&1
  tail -6 "$OUT_ROOT/$OBJ.log"
done
