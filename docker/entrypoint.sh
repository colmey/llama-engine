#!/bin/sh
# Inject ONLY the API key into the llama-swap config from the environment (so the
# secret is never baked into the image or the committed config), then start llama-swap.
# llama-swap's own ${PORT}/${macro} refs are left intact for it to expand.
# NOTE: envsubst does NOT support ${VAR:-default} — keep all other settings as
# literals in llama-swap.yaml, not as env placeholders.
set -eu

: "${LLM_API_KEY:?LLM_API_KEY is required — set it in .env}"

envsubst '${LLM_API_KEY}' < /config/llama-swap.yaml > /run/llama-swap.yaml

exec llama-swap --config /run/llama-swap.yaml --listen 0.0.0.0:8080
