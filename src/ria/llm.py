"""Gemini client with Tenacity retry, a local response cache and per-call logging.

* Only 429 and 5xx are retried. A 429 for the daily quota is not retried. Other errors fail fast.
* Cache (development only): key = model + full prompt + settings (+ schema for structured calls),
  stored as JSON files on disk. Controlled by one flag, config.LLM_CACHE_ENABLED.
  Cache hits are logged in logs/llm_calls.jsonl with cache_hit=true.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ria import config
from ria.tracing import langfuse_callbacks, log_llm_call, usage_from_response

_pace_lock = threading.Lock()
_last_call_end = 0.0


def _pace() -> None:
    """Wait so that calls start at least MIN_SECONDS_BETWEEN_CALLS apart (a simple client-side rate limit)."""
    global _last_call_end
    with _pace_lock:
        wait = config.MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_end)
        if wait > 0:
            time.sleep(wait)
        _last_call_end = time.monotonic()


class LLMError(RuntimeError):
    """A Gemini call failed in a way that retrying will not fix."""


_STATUS_RE = re.compile(r"\b(429|5\d\d)\b")


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits (429) and server errors (5xx).

    A 429 that says the DAILY quota is used up is not retryable: waiting seconds cannot fix it.
    """
    if re.search(r"PerDay|per day", str(exc)):
        return False
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or 500 <= code < 600
    name = type(exc).__name__
    if name in {"ResourceExhausted", "ServiceUnavailable", "InternalServerError", "DeadlineExceeded"}:
        return True
    return bool(_STATUS_RE.search(str(exc)))


def get_llm(role: str = "default", model: str | None = None, temperature: float | None = None) -> Any:
    """Build a Gemini chat model. `role` picks the thinking level from config.THINKING_LEVELS."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GOOGLE_API_KEY:
        raise LLMError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    kwargs: dict[str, Any] = {}
    level = config.THINKING_LEVELS.get(role)
    if level:
        kwargs["thinking_level"] = level
    return ChatGoogleGenerativeAI(
        model=model or config.GEMINI_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=1,  # Tenacity owns the retry policy
        callbacks=langfuse_callbacks() or None,  # Langfuse only when keys are set
        **kwargs,
    )


# ---------------------------------------------------------------- cache

def _normalise_messages(messages: Any) -> list[list[str]]:
    if isinstance(messages, str):
        return [["human", messages]]
    out = []
    for m in messages:
        if isinstance(m, (tuple, list)):
            role, content = m[0], m[1]
        else:
            role, content = getattr(m, "type", "human"), getattr(m, "content", str(m))
        out.append([str(role), content if isinstance(content, str) else json.dumps(content, sort_keys=True)])
    return out


def cache_key(llm: Any, messages: Any, schema_name: str = "") -> str:
    """Key = model + full prompt + settings (+ schema). Any change to any of them gives a new key."""
    payload = {
        "model": getattr(llm, "model", "") or getattr(llm, "model_name", ""),
        "temperature": getattr(llm, "temperature", None),
        "thinking": getattr(llm, "reasoning_effort", None),
        "schema": schema_name,
        "messages": _normalise_messages(messages),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_path(key: str) -> Path:
    return config.LLM_CACHE_DIR / f"{key}.json"


def _cache_get(key: str) -> dict[str, Any] | None:
    path = _cache_path(key)
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (OSError, ValueError):
        return None


def _cache_put(key: str, value: dict[str, Any]) -> None:
    config.LLM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _cache_path(key).write_text(json.dumps(value), encoding="utf-8")


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


# ---------------------------------------------------------------- core call

def _call(llm: Any, invoke: Callable[[], Any], node: str, key: str,
          encode: Callable[[Any], dict[str, Any]], decode: Callable[[dict[str, Any]], Any],
          usage_of: Callable[[Any], dict[str, int]]) -> Any:
    """Cache lookup, then retry loop, then logging. latency_s is model time only. wait_s is pacing and backoff."""
    model = getattr(llm, "model", config.GEMINI_MODEL)
    use_cache = config.LLM_CACHE_ENABLED
    if use_cache:
        started = time.perf_counter()
        hit = _cache_get(key)
        if hit is not None:
            log_llm_call(node, model, time.perf_counter() - started, 0, hit.get("usage", {}), cache_hit=True)
            return decode(hit)

    attempts = 0
    busy = 0.0

    @retry(
        retry=retry_if_exception(is_retryable),
        wait=wait_random_exponential(multiplier=config.LLM_BACKOFF_MULTIPLIER, max=60),  # exponential backoff with jitter
        stop=stop_after_attempt(config.LLM_MAX_ATTEMPTS),
        reraise=True,
    )
    def _invoke() -> Any:
        nonlocal attempts, busy
        attempts += 1
        _pace()
        started = time.perf_counter()
        try:
            return invoke()
        finally:
            busy += time.perf_counter() - started

    start = time.perf_counter()
    try:
        response = _invoke()
    except Exception as exc:  # noqa: BLE001
        log_llm_call(node, model, busy, attempts, {}, error=f"{type(exc).__name__}: {exc}",
                     wait_s=time.perf_counter() - start - busy, cache_hit=False if use_cache else None)
        raise LLMError(f"Gemini call failed ({type(exc).__name__}): {exc}") from exc
    usage = usage_of(response)
    log_llm_call(node, model, busy, attempts, usage, wait_s=time.perf_counter() - start - busy,
                 cache_hit=False if use_cache else None)
    if use_cache:
        _cache_put(key, {**encode(response), "usage": usage})
    return response


def invoke_with_retry(llm: Any, messages: Any, node: str = "llm") -> Any:
    """Plain text call. Returns an object with .content (text) and .usage_metadata."""
    from langchain_core.messages import AIMessage

    reply = _call(
        llm, lambda: llm.invoke(messages), node, cache_key(llm, messages),
        encode=lambda r: {"content": _text(r.content)}, decode=lambda d: AIMessage(content=d["content"]),
        usage_of=usage_from_response)
    if not isinstance(reply.content, str):  # Gemini can return a list of content parts: always give callers text
        reply = AIMessage(content=_text(reply.content))  # usage is already logged by _call
    return reply


def invoke_structured(llm: Any, messages: Any, schema: Any, node: str = "llm") -> Any:
    """Structured output call. Returns an instance of the Pydantic class `schema`."""
    structured = llm.with_structured_output(schema, include_raw=True)

    def run() -> Any:
        result = structured.invoke(messages)
        if result.get("parsed") is None:
            raise ValueError(f"Model output did not match {schema.__name__}: {result.get('parsing_error')}")
        return result

    def usage_of(result: Any) -> dict[str, int]:
        return usage_from_response(result["raw"]) if result.get("raw") is not None else {}

    result = _call(
        llm, run, node, cache_key(llm, messages, schema_name=json.dumps(schema.model_json_schema(), sort_keys=True)),
        encode=lambda r: {"parsed": r["parsed"].model_dump()},
        decode=lambda d: {"parsed": schema(**d["parsed"]), "raw": None}, usage_of=usage_of)
    return result["parsed"]
