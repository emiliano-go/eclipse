#!/usr/bin/env bash
# Build the standalone C++ target converter (no llama.cpp dependency).
set -euo pipefail
ROOT="${ROOT:-/data/ornith-byte}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/tools"
g++ -O3 -march=native -fopenmp -std=c++17 \
  "$HERE/tools/convert_targets.cpp" \
  -o "$ROOT/tools/convert_targets"
ls -l "$ROOT/tools/convert_targets"
