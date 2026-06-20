#!/usr/bin/env bash
# Unload whatever model llama-swap currently has resident, freeing its VRAM now.
#
# Why: gpt-oss-20b runs "tight" on this box (~12 GB, only ~1.4 GB free alongside the
# desktop), so it must yield the GPU on demand for gaming / GPU-heavy media. The next
# chat request cold-reloads the model automatically (~3s), so this is cheap to use.
# Safe to run when nothing is loaded (llama-swap just returns 200).
#
# Usage:
#   scripts/llm-unload.sh                       # free VRAM right now
#   scripts/llm-unload.sh %command%             # Steam launch option: free VRAM, then
#                                               #   exec the game (use the absolute path)
#
# Override the endpoint with $LLM_BASE_URL (default http://127.0.0.1:8081; a trailing
# /v1 is accepted and stripped).
set -euo pipefail

base_url="${LLM_BASE_URL:-http://127.0.0.1:8081}"
base_url="${base_url%/v1}"
base_url="${base_url%/}"

curl -fsS "${base_url}/unload" >/dev/null && echo "llm: model unloaded, VRAM freed"

# Used as a launcher wrapper? Now that the GPU is free, run the wrapped command.
if [ "$#" -gt 0 ]; then
  exec "$@"
fi
