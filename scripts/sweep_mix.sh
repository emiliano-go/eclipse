#!/usr/bin/env bash
# Mixed-loss sweep on the 1M shard, evaluated on held-out documents.
# Usage: scripts/sweep_mix.sh [TARGETS] [VAL_TARGETS] [STEPS] [OUT_ROOT]
set -euo pipefail
ROOT="${ROOT:-/data/ornith-byte}"
TARGETS="${1:-$ROOT/targets/train-1m}"
VAL="${2:-$ROOT/targets/val-full}"
STEPS="${3:-2000}"
OUT="${4:-$ROOT/runs/1m-mix}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/venv/bin/python"
mkdir -p "$OUT"
cd "$HERE"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
  local name="$1"; shift
  echo "=== $name"
  rm -rf "$OUT/$name"
  PYTHONPATH=src "$PY" -m byte_agent.train \
    --targets "$TARGETS" --val-targets "$VAL" \
    --out "$OUT/$name" --steps "$STEPS" --batch 8 --seq-len 1024 \
    --eval-every 100 --save-every 500 --seed 1337 "$@" \
    > "$OUT/$name.log" 2>&1
  PYTHONPATH=src "$PY" scripts/eval_model.py --checkpoint "$OUT/$name/final.pt" \
    --val-targets "$VAL" --out "$OUT/$name/doc_eval.json" \
    > "$OUT/$name.eval.log" 2>&1
  tail -3 "$OUT/$name.log"
  tail -2 "$OUT/$name.eval.log"
}

run eot_ce_10 --objective mix --alpha-kl 0.9 --alpha-ce 0.1
run eot_ce_25 --objective mix --alpha-kl 0.75 --alpha-ce 0.25
run eot_ce_50 --objective mix --alpha-kl 0.5 --alpha-ce 0.5
run mi_ce_25 --objective mix --alpha-kl 0 --alpha-mi 0.75 --alpha-ce 0.25
run eot_mi_ce --objective mix --alpha-kl 0.5 --alpha-mi 0.25 --alpha-ce 0.25
run eot_ce_25_T2 --objective mix --alpha-kl 0.75 --alpha-ce 0.25 --temperature 2.0
run eot_stage1 --objective eot --steps 1500
run twostage --objective ce --steps 500 --init-model "$OUT/eot_stage1/ckpt.pt"
