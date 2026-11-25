#!/usr/bin/env bash
# One-time setup for `congress-trades distill export`: llama.cpp's HF -> GGUF converter.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p tools
[ -d tools/llama.cpp ] || git clone --depth 1 https://github.com/ggml-org/llama.cpp tools/llama.cpp
uv venv -p 3.12 tools/convert-venv
uv pip install -p tools/convert-venv/bin/python \
  -r tools/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
echo "Converter ready: tools/convert-venv/bin/python tools/llama.cpp/convert_hf_to_gguf.py"
