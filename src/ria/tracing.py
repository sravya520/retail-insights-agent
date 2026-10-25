"""Per-call LLM logging. Phase 5 extends this with cost, per-request totals and Langfuse.

Every LLM call appends one JSON line to logs/llm_calls.jsonl with node, model, tokens and latency.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ria import config

LLM_CALLS_PATH = config.ROOT / "logs" / "llm_calls.jsonl"


def usage_from_response(response: Any) -> dict[str, int]:
    """Read token counts from LangChain usage metadata. Missing fields become 0."""
    meta = getattr(response, "usage_metadata", None) or {}
    details_in = meta.get("input_token_details") or {}
    details_out = meta.get("output_token_details") or {}
    return {
        "input_tokens": int(meta.get("input_tokens", 0)),
        "output_tokens": int(meta.get("output_tokens", 0)),
        "cached_tokens": int(details_in.get("cache_read", 0)),
        "reasoning_tokens": int(details_out.get("reasoning", 0)),
    }


def log_llm_call(node: str, model: str, latency_s: float, attempts: int, usage: dict[str, int],
                 error: str | None = None, path: Path | None = None, wait_s: float = 0.0,
                 cache_hit: bool | None = None) -> dict[str, Any]:
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "node": node, "model": model,
        "latency_s": round(latency_s, 3), "wait_s": round(wait_s, 3), "attempts": attempts, **usage,
    }
    if cache_hit is not None:
        record["cache_hit"] = cache_hit  # absent when the cache is switched off
    if error:
        record["error"] = error[:300]
    target = path or LLM_CALLS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


TRACES_PATH = config.ROOT / "logs" / "traces.jsonl"


def log_request_trace(record: dict[str, Any], path: Path | None = None) -> None:
    """One JSON line per request: thread, question, route, trajectory, verifier result, latency.

    Phase 5 adds token and cost totals.
    """
    target = path or TRACES_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **record}, default=str) + "\n")
