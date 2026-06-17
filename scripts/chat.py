#!/usr/bin/env python3
"""Minimal streaming chat client for the local llama-swap endpoint.

Stdlib only — no pip, no openai package. Talks to the OpenAI-compatible API,
streams tokens live, and shows the model's reasoning separately from the answer.

Usage:
    python scripts/chat.py                      # interactive REPL
    python scripts/chat.py --prompt "hello"     # one-shot (good for smoke tests)
    echo "hello" | python scripts/chat.py       # one-shot from stdin
    python scripts/chat.py --model glm-4.7-flash --show-reasoning

Connection (defaults work with this repo):
    --base-url  http://127.0.0.1:8081/v1   (or $LLM_BASE_URL)
    --key       from $LLM_API_KEY, else ./.api-key
    --model     glm-4.7-flash              (or $LLM_MODEL_ID)

REPL commands:  /reset  /system <text>  /model <name>  /models  /help  /quit
"""
import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent

# ANSI styling, disabled when output isn't a terminal.
_tty = sys.stdout.isatty()
def _c(code): return code if _tty else ""
DIM, CYAN, GREEN, YELLOW, RED, RESET = (
    _c("\033[2m"), _c("\033[36m"), _c("\033[32m"), _c("\033[33m"), _c("\033[31m"), _c("\033[0m"))


def resolve_key(cli_key):
    if cli_key:
        return cli_key
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    f = ROOT / ".api-key"
    if f.exists():
        return f.read_text().strip()
    sys.exit(f"{RED}No API key. Set $LLM_API_KEY or create .api-key.{RESET}")


def _request(base_url, key, path, payload=None):
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}"}
    if data:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers)


def list_models(base_url, key):
    try:
        with urllib.request.urlopen(_request(base_url, key, "/models")) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception as e:
        return _explain(e)


def _explain(e):
    """Turn a connection/HTTP error into a friendly string."""
    if isinstance(e, urllib.error.HTTPError):
        body = e.read().decode(errors="replace")
        if e.code == 401:
            return f"{RED}401 Unauthorized — wrong/missing API key.{RESET}"
        if e.code == 503:
            return f"{YELLOW}503 — model is still loading, try again in a few seconds.{RESET}"
        return f"{RED}HTTP {e.code}: {body[:300]}{RESET}"
    if isinstance(e, urllib.error.URLError):
        return (f"{RED}Can't reach the server ({e.reason}). Is it up?{RESET}\n"
                f"{DIM}  docker compose up -d{RESET}")
    return f"{RED}{type(e).__name__}: {e}{RESET}"


def stream_reply(base_url, key, model, messages, show_reasoning):
    """Stream one assistant turn. Returns the answer text (content only) or None on error."""
    payload = {
        "model": model, "messages": messages, "stream": True,
        "temperature": 0.7, "top_p": 1.0, "min_p": 0.01,
        "repeat_penalty": 1.0, "max_tokens": 2048,
    }
    answer, in_reasoning = [], False
    try:
        with urllib.request.urlopen(_request(base_url, key, "/chat/completions", payload)) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                delta = json.loads(data)["choices"][0].get("delta", {})

                reasoning = delta.get("reasoning_content")
                if reasoning and show_reasoning:
                    if not in_reasoning:
                        sys.stdout.write(f"{DIM}[thinking] ")
                        in_reasoning = True
                    sys.stdout.write(f"{DIM}{reasoning}{RESET}" if not _tty else reasoning)
                    sys.stdout.flush()

                content = delta.get("content")
                if content:
                    if in_reasoning:
                        sys.stdout.write(f"{RESET}\n\n")
                        in_reasoning = False
                    sys.stdout.write(content)
                    sys.stdout.flush()
                    answer.append(content)
        sys.stdout.write(RESET + "\n")
        return "".join(answer)
    except Exception as e:
        sys.stdout.write(RESET + "\n" + _explain(e) + "\n")
        return None


def main():
    ap = argparse.ArgumentParser(description="Streaming chat client for the local GLM endpoint.")
    ap.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8081/v1"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL_ID", "glm-4.7-flash"))
    ap.add_argument("--key", default=None)
    ap.add_argument("--system", default=None, help="System prompt.")
    ap.add_argument("--prompt", default=None, help="One-shot prompt, then exit.")
    ap.add_argument("--show-reasoning", action="store_true", help="Stream GLM's reasoning too.")
    args = ap.parse_args()

    key = resolve_key(args.key)
    base_url, model = args.base_url, args.model
    history = [{"role": "system", "content": args.system}] if args.system else []

    # One-shot mode: --prompt or piped stdin.
    oneshot = args.prompt if args.prompt is not None else (None if sys.stdin.isatty() else sys.stdin.read().strip())
    if oneshot:
        history.append({"role": "user", "content": oneshot})
        reply = stream_reply(base_url, key, model, history, show_reasoning=True)
        sys.exit(0 if reply is not None else 1)

    # Interactive REPL.
    models = list_models(base_url, key)
    if isinstance(models, str):           # error string
        print(models)
        sys.exit(1)
    print(f"{CYAN}Connected to {base_url}{RESET}  models: {GREEN}{', '.join(models) or '(none)'}{RESET}")
    print(f"{DIM}Model: {model}. Commands: /reset /system <t> /model <n> /models /help /quit{RESET}\n")

    while True:
        try:
            user = input(f"{GREEN}you ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in ("/quit", "/exit", "/q"):
            break
        if user == "/help":
            print(f"{DIM}/reset  /system <text>  /model <name>  /models  /quit{RESET}")
            continue
        if user == "/reset":
            history = [h for h in history if h["role"] == "system"]
            print(f"{DIM}(history cleared){RESET}")
            continue
        if user.startswith("/system"):
            sysmsg = user[len("/system"):].strip()
            history = [{"role": "system", "content": sysmsg}] + [h for h in history if h["role"] != "system"]
            print(f"{DIM}(system prompt set){RESET}")
            continue
        if user.startswith("/model"):
            arg = user[len("/model"):].strip()
            if arg:
                model = arg
                print(f"{DIM}(model → {model}){RESET}")
            continue
        if user == "/models":
            print(f"{DIM}{', '.join(list_models(base_url, key))}{RESET}")
            continue

        history.append({"role": "user", "content": user})
        print(f"{CYAN}{model} ▸ {RESET}", end="", flush=True)
        reply = stream_reply(base_url, key, model, history, show_reasoning=args.show_reasoning)
        if reply:
            history.append({"role": "assistant", "content": reply})
        else:
            history.pop()                 # drop the user turn that failed


if __name__ == "__main__":
    main()
