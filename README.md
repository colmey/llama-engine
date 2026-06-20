# Local inference engine

A small, self-hosted **OpenAI-compatible** inference endpoint for local LLMs, built on
[llama.cpp](https://github.com/ggml-org/llama.cpp) (CUDA) + [llama-swap](https://github.com/mostlygeek/llama-swap).
One stable URL serves many models; clients pick the model by name and llama-swap loads it
on demand (and unloads it when idle). Intended as a backend for agents and tools.

Ships configured for **GLM-4.7-Flash** (30B-A3B MoE) on a 16 GB GPU as the worked example,
but adding any GGUF is a few lines of config.

```
client ──HTTP / OpenAI API──► ${LLM_HOST_PORT:-8081}  (0.0.0.0 = LAN-accessible)
                                     │
                              ┌──────▼──────────────────---┐
                              │ container (llm-inference)  │
                              │  llama-swap  — auth +      │
                              │   picks model by name      │
                              │     └─ llama-server (CUDA) │  loads the requested GGUF on the GPU
                              └────────────────────────────┘
                                 mounts ./models (ro) + the API key from .env
```

Everything runs in Docker, so a host only needs the NVIDIA driver and Container Toolkit —
no CUDA toolkit, no building llama.cpp by hand.

## Deploy on a new machine

**Host prerequisites:** an NVIDIA GPU, driver ≥ 580, Docker + Compose, and the **NVIDIA
Container Toolkit** (the one piece not bundled in the image):

```bash
# NVIDIA Container Toolkit (Ubuntu/Debian)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# This stack requests the GPU via CDI (Container Device Interface), which survives a
# `systemctl daemon-reload`. The legacy `gpus: all` hook does NOT — a daily apt timer's
# reload silently strips the container's GPU access, dropping inference to CPU. Generate
# the CDI spec (re-run after each NVIDIA driver upgrade):
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

docker run --rm --gpus all nvidia/cuda:13.0.1-runtime-ubuntu24.04 nvidia-smi   # verify driver+toolkit
```

**Deploy:**

```bash
# 1. API key (a secret you choose; both files are git-ignored)
openssl rand -hex 32 > .api-key && chmod 600 .api-key
echo "LLM_API_KEY=$(cat .api-key)" > .env

# 2. a model GGUF into ./models  (example: GLM-4.7-Flash)
pip install -U "huggingface_hub[cli]"
hf download unsloth/GLM-4.7-Flash-GGUF --include "*UD-Q4_K_XL*" --local-dir models

# 3. build the image (compiles llama.cpp for your GPU — ~10–20 min first time, then cached) and start
docker compose up -d --build

# 4. confirm it's up
curl -s -H "Authorization: Bearer $(cat .api-key)" http://127.0.0.1:8081/v1/models
```

Lifecycle: `docker compose logs -f` · `docker compose restart` · `docker compose down`.
The first request to a model loads it (~10–30 s); after that it stays hot until idle (`ttl`)
or another model is requested.

**Portability:** the image is portable across NVIDIA-Linux hosts. The CPU backend is built
generic (AVX2), but the **GPU code is compiled for one architecture** — `CUDA_ARCH: "120"`
(Blackwell / sm_120) in [docker-compose.yml](docker-compose.yml). For a different GPU, change
it (e.g. `"89;120"` for Ada + Blackwell) and rebuild.

## Talk to the model(s)

The `model` field selects which GGUF llama-swap loads; `GET /v1/models` lists what's defined.

**curl:**
```bash
KEY=$(cat .api-key)
curl -s http://127.0.0.1:8081/v1/chat/completions \
  -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' \
  -d '{"model":"glm-4.7-flash","messages":[{"role":"user","content":"hi"}],"max_tokens":512}'
```

**Bundled client** ([scripts/chat.py](scripts/chat.py)) — stdlib only, no `pip`. Streams
tokens and shows reasoning separately from the answer:
```bash
python3 scripts/chat.py                    # interactive REPL
python3 scripts/chat.py --prompt "hello"   # one-shot smoke test
python3 scripts/chat.py --show-reasoning   # also stream the model's chain-of-thought
```
It reads the key from `.api-key` and defaults to `http://127.0.0.1:8081/v1`.

**OpenAI SDK** (incl. tool calling, for agents):
```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8081/v1", api_key=open(".api-key").read().strip())

resp = client.chat.completions.create(
    model="glm-4.7-flash",                       # name selects/swaps the model
    messages=[{"role": "user", "content": "What's the weather in Denver? Use the tool."}],
    tools=[{"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    temperature=0.7, top_p=1.0, max_tokens=512,   # reasoning models burn tokens — leave room
    extra_body={"min_p": 0.01},
)
msg = resp.choices[0].message
print(msg.tool_calls or msg.content)
```

> GLM-4.7-Flash reasons before answering: the chain-of-thought arrives in `reasoning_content`
> and the answer in `content`. Budget `max_tokens ≥ 512` or you may get only reasoning back.

## How API keying works

1. You generate a secret into **`.api-key`** (git-ignored).
2. `.env` exposes it as **`LLM_API_KEY`**; Compose passes that into the container.
3. At startup [docker/entrypoint.sh](docker/entrypoint.sh) substitutes it into llama-swap's
   `apiKeys:` list (writing a filled copy to `/run`), so the key lives only in memory at
   runtime — never baked into the image or the committed config.
4. Clients send it as `Authorization: Bearer <key>`. llama-swap validates the token and
   **strips** it before forwarding to llama-server. No key → `401`.

Rotate by regenerating: `openssl rand -hex 32 > .api-key && echo "LLM_API_KEY=$(cat .api-key)" > .env && docker compose up -d`.

## Add a model

1. Drop the GGUF into `./models/`.
2. Copy a block under `models:` in [docker/llama-swap.yaml](docker/llama-swap.yaml) (there's a
   commented template), set `-m /models/<file>.gguf`, and tune `--n-cpu-moe` for its size.
3. `docker compose restart`. Clients use the new `model` name. Only one model occupies VRAM
   at a time (llama-swap unloads the previous one), so many models can live on disk.

## Tune / change config

All runtime flags live in the model's `cmd:` block in
[docker/llama-swap.yaml](docker/llama-swap.yaml). Edit, then `docker compose restart` (the
config is mounted — **no rebuild**); the model relaunches with the new flags on the next request.

| Lever | What it does |
|---|---|
| `--n-cpu-moe N` | Main VRAM ↔ speed knob: keeps the top N layers' experts on CPU. **Lower N = more on GPU = faster**, until you OOM. Measured (GLM UD-Q4_K_XL, 32k ctx): N=22 → ~12.6 GB / ~72 tok/s; N=20 → ~13 GB / ~76 tok/s. |
| `ttl` | Idle seconds before the model unloads and frees VRAM. `0` = stay resident; `1800` = warm for 30 min then release. |
| `--cache-reuse 256` | Reuses a cached prompt prefix (e.g. the system prompt) across requests → much faster time-to-first-token. |
| `-c` | Context length. Bigger ⇒ more KV-cache VRAM, so raise `--n-cpu-moe` (or quantize KV with `--cache-type-k/v q8_0`) to fit. |

## Sharing the GPU with a desktop

On a workstation where the GPU also drives your desktop, a model tuned to nearly fill VRAM will
OOM when a game or other GPU-heavy app needs the card. Two safety nets ship with the stack:

- **Auto-evict sidecar** ([scripts/vram-guard.sh](scripts/vram-guard.sh)) — runs inside the
  container (started by the compose entrypoint), polls free VRAM every 2 s, and unloads the
  resident model the moment free VRAM drops below `VRAM_GUARD_FLOOR_MB` (default 700 MiB). The GPU
  frees automatically; the next request cold-reloads the model (~3 s). It only arms after seeing a
  model loaded with healthy headroom, so it never false-evicts at rest. Tune `VRAM_GUARD_FLOOR_MB` /
  `VRAM_GUARD_INTERVAL` in [docker-compose.yml](docker-compose.yml), or set `VRAM_GUARD_DISABLE=1`.
- **Manual unload** ([scripts/llm-unload.sh](scripts/llm-unload.sh)) — frees the GPU now via
  `GET /unload`. Doubles as a launcher wrapper that frees VRAM the instant a game starts, e.g. as a
  Steam launch option: `/abs/path/scripts/llm-unload.sh %command%`.

This matters most for models tuned to fill VRAM (e.g. `gpt-oss-20b` at `--n-cpu-moe 4` leaves only
~1.4 GB free). For a bigger permanent cushion instead, raise `--n-cpu-moe` in the model's config.

## Security

- **LAN-accessible** — published as `0.0.0.0:${LLM_HOST_PORT:-8081}` by default; any device
  on the network can reach it. Bind to `127.0.0.1` only by setting `LLM_HOST_PORT` in `.env`
  and adjusting the port mapping in [docker-compose.yml](docker-compose.yml) to
  `"127.0.0.1:${LLM_HOST_PORT:-8081}:${LLM_CONTAINER_PORT:-8080}"`.
- **API key required** — even locally, so other users/processes can't use your GPU.
- **Models read-only** — mounted `:ro`; the container can't modify your GGUFs.
- **Secrets git-ignored** — `.api-key`, `.env`, and `models/` never leave the box.
- **Remote access → tunnel** (SSH / WireGuard / Tailscale), don't expose the port.

## Files

| Path | What it is |
|---|---|
| `docker-compose.yml` | What you run — GPU, the `127.0.0.1:8081` bind, mounts, `.env`. |
| `docker/Dockerfile` | Builds llama.cpp (CUDA) + bundles llama-swap. |
| `docker/entrypoint.sh` | Injects the API key into the config at startup. |
| `docker/llama-swap.yaml` | **Your config** — which models exist and their tuned flags. |
| `.env` / `.env.example` | API key for Compose (`.env` is git-ignored). |
| `.api-key` | The secret itself (git-ignored). |
| `scripts/chat.py` | Dependency-free streaming chat client / smoke test. |
| `scripts/llm-unload.sh` | Free GPU VRAM now (`GET /unload`); |
| `scripts/vram-guard.sh` | Sidecar that auto-evicts the model under VRAM pressure. |
| `models/` | Your GGUF files (git-ignored data). |
| `CLAUDE.md` | Hardware/build notes for this box. |
