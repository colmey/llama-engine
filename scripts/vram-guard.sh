#!/bin/sh
# Sidecar guard: auto-evict the resident llama-swap model when the GPU runs low on
# free VRAM, so gaming / other GPU work can reclaim the card without OOMing.
#
# gpt-oss-20b runs "tight" on this 16 GB card (~12.3 GB, ~1.4 GB free at rest). When
# something external (a game, heavy media) pushes free VRAM below FLOOR, this GETs
# /unload to free ~12 GB; the next chat request cold-reloads the model (~3s), so an
# eviction is cheap. It only fires once it has seen a model "ready" WITH healthy
# headroom (arming), so it never fights a model that loaded into an already-tight
# desktop — it acts on a *drop* in free VRAM, not on a model that was never roomy.
#
# Runs INSIDE the llm container (started by the compose entrypoint). Sees global GPU
# free VRAM via nvidia-smi (CDI) and talks to llama-swap on localhost:8080.
#
# Tunables (env — set in docker-compose.yml `environment:` or .env):
#   VRAM_GUARD_FLOOR_MB   evict when free VRAM drops below this   (default 700)
#   VRAM_GUARD_INTERVAL   poll seconds                            (default 2)
#   VRAM_GUARD_DEBOUNCE   consecutive low polls before evicting   (default 2)
#   VRAM_GUARD_DISABLE=1  disable the guard (idle no-op)
#   LLM_API_KEY           bearer token (already in the container env)
set -u

log() { echo "$(date -u +%H:%M:%S) vram-guard: $*"; }

if [ "${VRAM_GUARD_DISABLE:-0}" = "1" ]; then
  log "disabled (VRAM_GUARD_DISABLE=1)"
  exec sleep infinity
fi

floor=${VRAM_GUARD_FLOOR_MB:-700}
interval=${VRAM_GUARD_INTERVAL:-2}
debounce=${VRAM_GUARD_DEBOUNCE:-2}
url="http://127.0.0.1:8080"
key="${LLM_API_KEY:-}"

# Wait for the GPU to be visible (CDI device + driver ready) before guarding.
n=0
while ! nvidia-smi -L >/dev/null 2>&1; do
  n=$((n + 1))
  if [ "$n" -ge 30 ]; then log "no GPU visible after 60s — guard exiting"; exit 0; fi
  sleep 2
done
log "active: evict the loaded model when free VRAM < ${floor} MiB for ${debounce}x${interval}s"

armed=0   # only evict after we've seen the model resident with free >= floor
low=0     # consecutive sub-floor polls
while :; do
  running=$(curl -fsS -H "Authorization: Bearer ${key}" "${url}/running" 2>/dev/null || true)
  case "${running}" in
    *'"state":"ready"'*) ;;                              # a model is fully loaded — guard it
    *) armed=0; low=0; sleep "${interval}"; continue ;;  # nothing ready → nothing to guard
  esac

  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -dc '0-9')
  if [ -z "${free}" ]; then sleep "${interval}"; continue; fi

  if [ "${free}" -ge "${floor}" ]; then
    armed=1; low=0                                       # healthy headroom seen → arm
  elif [ "${armed}" -eq 1 ]; then
    low=$((low + 1))
    if [ "${low}" -ge "${debounce}" ]; then
      log "free VRAM ${free} MiB < ${floor} MiB — unloading model to free the GPU"
      if curl -fsS -H "Authorization: Bearer ${key}" "${url}/unload" >/dev/null 2>&1; then
        log "unloaded (~12 GB freed); next request reloads the model"
      else
        log "unload request failed"
      fi
      armed=0; low=0
      sleep $((interval * 5))                            # back off while the new workload ramps
      continue
    fi
  fi
  sleep "${interval}"
done
