#!/usr/bin/env python3
"""Benchmark cold-load latency and warm generation speed through llama-swap.

The script is dependency-free and uses the same connection settings as chat.py.
"Cold TTFT" is measured at the client from immediately before the request until
the first generated content or reasoning chunk. It therefore includes model
startup, prompt processing, network overhead, and first-token generation.

Examples:
    python3 scripts/benchmark.py
    python3 scripts/benchmark.py --model qwen3.5-9b --runs 5
    python3 scripts/benchmark.py --max-tokens 256 --json benchmark-results.json
"""
import argparse
import datetime as dt
import json
import os
import pathlib
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_BASE_URL = "http://127.0.0.1:8081/v1"
DEFAULT_PROMPT = "Write exactly 128 words explaining why reproducible benchmarks matter."


class BenchmarkError(RuntimeError):
    pass


def resolve_key(cli_key):
    if cli_key:
        return cli_key
    if os.environ.get("LLM_API_KEY"):
        return os.environ["LLM_API_KEY"]
    key_file = ROOT / ".api-key"
    if key_file.exists():
        return key_file.read_text().strip()
    raise BenchmarkError("No API key. Set $LLM_API_KEY or create .api-key.")


def request(base_url, key, path, payload=None, method=None):
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Authorization": f"Bearer {key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    return urllib.request.Request(url, data=data, headers=headers, method=method)


def read_json(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise BenchmarkError(f"HTTP {exc.code}: {body[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(str(exc)) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise BenchmarkError(f"Invalid JSON response: {exc}") from exc


def list_models(base_url, key, timeout):
    data = read_json(request(base_url, key, "/models"), timeout)
    try:
        return [item["id"] for item in data.get("data", [])]
    except (AttributeError, KeyError, TypeError) as exc:
        raise BenchmarkError("Invalid model-list response") from exc


def root_url(base_url):
    url = base_url.rstrip("/")
    return url[:-3] if url.endswith("/v1") else url


def _open(req, timeout):
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise BenchmarkError(f"HTTP {exc.code}: {body[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(str(exc)) from exc


def unload_all(base_url, key, timeout):
    """Unload models, preferring the route already used by this repository."""
    root = root_url(base_url)
    attempts = [
        request(root, key, "/unload"),
        request(root, key, "/api/models/unload", payload={}, method="POST"),
    ]
    errors = []
    for req in attempts:
        try:
            _open(req, timeout)
            return
        except BenchmarkError as exc:
            errors.append(str(exc))
    raise BenchmarkError("Unable to unload models: " + "; ".join(errors))


def parse_sse(response, clock=time.perf_counter):
    """Consume an OpenAI SSE response and return client-side timing data."""
    started = clock()
    first_token_at = None
    usage = None
    for raw in response:
        line = raw.decode("utf-8").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError as exc:
            raise BenchmarkError(f"Malformed SSE data: {exc}") from exc
        if event.get("usage"):
            usage = event["usage"]
        for choice in event.get("choices", []):
            delta = choice.get("delta", {})
            if first_token_at is None and (delta.get("content") or delta.get("reasoning_content")):
                first_token_at = clock()
    ended = clock()
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    ttft = first_token_at - started if first_token_at is not None else None
    generation_seconds = ended - first_token_at if first_token_at is not None else None
    tokens_per_second = None
    if completion_tokens is not None and generation_seconds and generation_seconds > 0:
        tokens_per_second = completion_tokens / generation_seconds
    return {
        "ttft_seconds": ttft,
        "total_seconds": ended - started,
        "generation_seconds": generation_seconds,
        "completion_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
    }


def run_completion(base_url, key, model, prompt, max_tokens, timeout):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    req = request(base_url, key, "/chat/completions", payload)
    requested_at = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            result = parse_sse(response)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise BenchmarkError(f"HTTP {exc.code}: {body[:300]}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(str(exc)) from exc
    # parse_sse starts after headers arrive; include proxy/model startup in TTFT and total.
    headers_received_at = time.perf_counter() - result["total_seconds"]
    startup_seconds = headers_received_at - requested_at
    result["ttft_seconds"] = (result["ttft_seconds"] + startup_seconds
                              if result["ttft_seconds"] is not None else None)
    result["total_seconds"] += startup_seconds
    if result["generation_seconds"] is not None and result["completion_tokens"] is not None:
        result["tokens_per_second"] = result["completion_tokens"] / result["generation_seconds"]
    return result


def average(values):
    present = [value for value in values if value is not None]
    return statistics.fmean(present) if present else None


def aggregate(cold, warm):
    successful = ([cold] if cold and "error" not in cold else []) + [r for r in warm if "error" not in r]
    return {
        "successful_runs": len(successful),
        "cold_ttft_seconds": cold.get("ttft_seconds") if cold and "error" not in cold else None,
        "cold_total_seconds": cold.get("total_seconds") if cold and "error" not in cold else None,
        "warm_ttft_seconds": average([r.get("ttft_seconds") for r in warm if "error" not in r]),
        "warm_total_seconds": average([r.get("total_seconds") for r in warm if "error" not in r]),
        "warm_tokens_per_second": average([r.get("tokens_per_second") for r in warm if "error" not in r]),
    }


def benchmark_model(args, key, model):
    cold = None
    warm = []
    try:
        unload_all(args.base_url, key, args.timeout)
        cold = run_completion(args.base_url, key, model, args.prompt, args.max_tokens, args.timeout)
    except BenchmarkError as exc:
        cold = {"error": str(exc)}
    for _ in range(args.runs):
        try:
            warm.append(run_completion(args.base_url, key, model, args.prompt, args.max_tokens, args.timeout))
        except BenchmarkError as exc:
            warm.append({"error": str(exc)})
    return {"model": model, "cold": cold, "warm": warm, "summary": aggregate(cold, warm)}


def fmt(value, suffix=""):
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def render_table(results):
    headers = ["MODEL", "COLD TTFT", "COLD TOTAL", "WARM TTFT", "WARM TOTAL", "WARM TOK/S", "RUNS"]
    rows = []
    for result in results:
        summary = result["summary"]
        rows.append([
            result["model"], fmt(summary["cold_ttft_seconds"], "s"),
            fmt(summary["cold_total_seconds"], "s"), fmt(summary["warm_ttft_seconds"], "s"),
            fmt(summary["warm_total_seconds"], "s"), fmt(summary["warm_tokens_per_second"]),
            str(summary["successful_runs"]),
        ])
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = ["  ".join(value.ljust(widths[i]) for i, value in enumerate(headers)),
             "  ".join("-" * width for width in widths)]
    lines.extend("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)) for row in rows)
    return "\n".join(lines)


def render_errors(results):
    lines = []
    for result in results:
        if "error" in result["cold"]:
            lines.append(f"{result['model']} cold: {result['cold']['error']}")
        for index, run in enumerate(result["warm"], 1):
            if "error" in run:
                lines.append(f"{result['model']} warm {index}: {run['error']}")
    return "\n".join(lines)


def build_parser():
    parser = argparse.ArgumentParser(description="Benchmark models served by llama-swap.")
    parser.add_argument("--base-url", default=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--key", default=None, help="API key (otherwise $LLM_API_KEY or .api-key).")
    parser.add_argument("--model", action="append", help="Model to benchmark; repeat to select several.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--runs", type=int, default=3, help="Number of warm runs per model (default: 3).")
    parser.add_argument("--timeout", type=float, default=600, help="Per-request timeout in seconds.")
    parser.add_argument("--json", type=pathlib.Path, metavar="PATH", help="Write detailed JSON results.")
    return parser


def validate_args(parser, args):
    if args.max_tokens < 1:
        parser.error("--max-tokens must be at least 1")
    if args.runs < 1:
        parser.error("--runs must be at least 1")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than 0")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(parser, args)
    try:
        key = resolve_key(args.key)
        available = list_models(args.base_url, key, args.timeout)
    except BenchmarkError as exc:
        print(f"benchmark: {exc}", file=sys.stderr)
        return 1
    selected = args.model or available
    unknown = [model for model in selected if model not in available]
    if unknown:
        print("benchmark: unknown model(s): " + ", ".join(unknown), file=sys.stderr)
        return 1
    if not selected:
        print("benchmark: no models were returned by the server", file=sys.stderr)
        return 1

    results = []
    for model in selected:
        print(f"Benchmarking {model}...", file=sys.stderr)
        results.append(benchmark_model(args, key, model))
    print(render_table(results))
    errors = render_errors(results)
    if errors:
        print("\nFailures:\n" + errors, file=sys.stderr)

    document = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "configuration": {
            "base_url": args.base_url, "models": selected, "prompt": args.prompt,
            "max_tokens": args.max_tokens, "warm_runs": args.runs, "timeout_seconds": args.timeout,
        },
        "results": results,
    }
    if args.json:
        try:
            args.json.write_text(json.dumps(document, indent=2) + "\n")
        except OSError as exc:
            print(f"benchmark: cannot write {args.json}: {exc}", file=sys.stderr)
            return 1
    return 0 if all(result["summary"]["successful_runs"] > 0 for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
