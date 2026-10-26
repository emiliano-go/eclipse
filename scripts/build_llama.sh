#!/usr/bin/env bash
# Build llama.cpp with CUDA for sm_86 and install the logits dumper example.
set -euo pipefail

ROOT="${ROOT:-/data/ornith-byte}"
SHA="${LLAMA_CPP_COMMIT:-d854d98bae598388ecea1ed84396587e5edc326d}"
SRC="$ROOT/tools/llama.cpp"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

if [ ! -d "$SRC" ]; then
  tmp="$(mktemp -d)"
  curl -sSL --retry 5 --retry-delay 3 \
    -o "$tmp/llama.tar.gz" \
    "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/$SHA"
  tar xzf "$tmp/llama.tar.gz" -C "$ROOT/tools"
  mv "$ROOT/tools/llama.cpp-$SHA" "$SRC"
  rm -rf "$tmp"
fi
echo "$SHA" > "$ROOT/tools/llama.commit"

mkdir -p "$SRC/examples/dump-logits"
cp "$HERE/tools/dump_logits.cpp" "$SRC/examples/dump-logits/dump_logits.cpp"
cp "$HERE/tools/CMakeLists.txt" "$SRC/examples/dump-logits/CMakeLists.txt"
if ! grep -q "add_subdirectory(dump-logits)" "$SRC/examples/CMakeLists.txt"; then
  sed -i 's/add_subdirectory(debug)/add_subdirectory(debug)\n    add_subdirectory(dump-logits)/' "$SRC/examples/CMakeLists.txt"
fi

cmake -S "$SRC" -B "$SRC/build" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "$SRC/build" -j"$(nproc)" --target llama-cli dump_logits
"$SRC/build/bin/llama-cli" --version | head -5
ls -l "$SRC/build/bin/llama-cli" "$SRC/build/bin/dump_logits"
