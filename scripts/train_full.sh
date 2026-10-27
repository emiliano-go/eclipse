#!/usr/bin/env bash
# Reproduce the full-data EOT + 10% CE run and its document evaluation.
set -euo pipefail
ROOT="${ROOT:-/data/ornith-byte}"
OUT="${1:-$ROOT/runs/full-eot-ce-10}"
PY="$ROOT/venv/bin/python"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$OUT"
cd "$HERE"
PYTHONPATH=src "$PY" -m byte_agent.train \
  --targets "$ROOT/targets/train-full" \
  --val-targets "$ROOT/targets/val-full" \
  --out "$OUT" \
  --objective mix --alpha-kl 0.9 --alpha-ce 0.1 \
  --steps 10000 --batch 8 --seq-len 1024 \
  --eval-every 1000 --save-every 2000 --seed 1337
PYTHONPATH=src "$PY" scripts/eval_model.py \
  --checkpoint "$OUT/final.pt" \
  --val-targets "$ROOT/targets/val-full" \
  --prompt-format eot \
  --out "$OUT/doc_eval.json"
