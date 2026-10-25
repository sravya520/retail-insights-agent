"""Observability: one JSON line per LLM call, one per request, SQL repair reasons, optional Langfuse.

  logs/llm_calls.jsonl   every LLM call: node, model, input, output and cached tokens, latency, cost, cache hit
  logs/traces.jsonl      every request: question, route, trajectory, verified flag, totals (tokens, cost, latency)
  logs/sql_repairs.jsonl every SQL repair attempt with the reason it was needed

Langfuse is used only when LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are both set and the package is installed.
Otherwise the local files are all there is, and nothing fails.
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

from ria import config

LLM_CALLS_PATH = config.ROOT / "logs" / "llm_calls.jsonl"
TRACES_PATH = config.ROOT / "logs" / "traces.jsonl"
SQL_REPAIRS_PATH = config.ROOT / "logs" / "sql_repairs.jsonl"

_scope: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("ria_request_scope", default=None)


def usage_from_response(response: Any) -> dict[str, int]:
    """Read token counts from LangChain usage metadata. Missing fields become 0.

    output_tokens already includes reasoning (thinking) tokens.
    """
    meta = getattr(response, "usage_metadata", None) or {}
    details_in = meta.get("input_token_details") or {}
    details_out = meta.get("output_token_details") or {}
    return {
        "input_tokens": int(meta.get("input_tokens", 0)),
        "output_tokens": int(meta.get("output_tokens", 0)),
        "cached_tokens": int(details_in.get("cache_read", 0)),
        "reasoning_tokens": int(details_out.get("reasoning", 0)),
    }


def cost_usd(model: str, usage: dict[str, int]) -> float:
    """Cost of one call from config.PRICES (USD per 1M tokens). Unknown model gives 0.0.

    Uncached input tokens are billed at the input price, cached input tokens at the cached price,
    and output tokens (thinking included) at the output price.
    """
    price = config.PRICES.get(model)
    if not price:
        return 0.0
    cached = usage.get("cached_tokens", 0)
    uncached = max(usage.get("input_tokens", 0) - cached, 0)
    return (uncached * price["input"] + cached * price["cached_input"] + usage.get("output_tokens", 0) * price["output"]) / 1e6


@contextlib.contextmanager
def request_scope() -> Iterator[list[dict[str, Any]]]:
    """Collect the call records made while handling one request, to compute per-request totals."""
    records: list[dict[str, Any]] = []
    token = _scope.set(records)
    try:
        yield records
    finally:
        _scope.reset(token)


def log_llm_call(node: str, model: str, latency_s: float, attempts: int, usage: dict[str, int],
                 error: str | None = None, path: Path | None = None, wait_s: float = 0.0,
                 cache_hit: bool | None = None) -> dict[str, Any]:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "node": node, "model": model,
        "latency_s": round(latency_s, 3), "wait_s": round(wait_s, 3), "attempts": attempts, **usage,
        # A cache hit costs nothing. The tokens stored with it are those of the original call.
        "cost_usd": 0.0 if cache_hit else round(cost_usd(model, usage), 8),
    }
    if cache_hit is not None:
        record["cache_hit"] = cache_hit  # absent when the cache is switched off
    if error:
        record["error"] = error[:300]
    target = path or LLM_CALLS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    scope = _scope.get()
    if scope is not None:
        scope.append(record)
    return record


def totals(calls: list[dict[str, Any]]) -> dict[str, Any]:
    """Per-request totals. Cache hits are counted separately and add no tokens, cost or latency."""
    real = [c for c in calls if not c.get("cache_hit")]
    return {
        "llm_calls": len(real), "cache_hits": len(calls) - len(real),
        "input_tokens": sum(c.get("input_tokens", 0) for c in real),
        "output_tokens": sum(c.get("output_tokens", 0) for c in real),
        "cached_tokens": sum(c.get("cached_tokens", 0) for c in real),
        "total_tokens": sum(c.get("input_tokens", 0) + c.get("output_tokens", 0) for c in real),
        "cost_usd": round(sum(c.get("cost_usd", 0.0) for c in real), 8),
        "llm_latency_s": round(sum(c.get("latency_s", 0.0) for c in real), 3),
    }


def log_request_trace(record: dict[str, Any], path: Path | None = None) -> None:
    """One JSON line per request: thread, question, route, trajectory, verified flag, totals, latency."""
    target = path or TRACES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}, default=str) + "\n")


def log_sql_repair(entry: dict[str, Any], path: Path | None = None) -> None:
    """One line per SQL repair attempt: the question, the failed SQL, the stage (validation or execution) and the error."""
    target = path or SQL_REPAIRS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **entry}) + "\n")


_langfuse_warned = False


def langfuse_callbacks() -> list[Any]:
    """Langfuse callback handlers for LangChain, or an empty list. Never raises."""
    global _langfuse_warned
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return []
    try:
        try:
            from langfuse.langchain import CallbackHandler  # langfuse 3.x
        except ImportError:
            from langfuse.callback import CallbackHandler  # langfuse 2.x
        return [CallbackHandler()]
    except Exception as exc:  # noqa: BLE001 - tracing must never break the app
        if not _langfuse_warned:
            print(f"Langfuse keys are set but Langfuse could not start ({type(exc).__name__}). Using local tracing only.")
            _langfuse_warned = True
        return []
