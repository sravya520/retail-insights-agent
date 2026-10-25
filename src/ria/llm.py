"""Gemini client with Tenacity retry and per-call logging.

Only 429 and 5xx are retried, all other errors fail fast with a clear message.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ria import config
from ria.tracing import log_llm_call, usage_from_response


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
        **kwargs,
    )


def invoke_with_retry(llm: Any, messages: Any, node: str = "llm") -> Any:
    """Call the model with retry, and log node, tokens and latency.

    latency_s is time spent inside the model call (all attempts). wait_s is pacing and retry backoff time.
    """
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
            return llm.invoke(messages)
        finally:
            busy += time.perf_counter() - started

    model = getattr(llm, "model", config.GEMINI_MODEL)
    start = time.perf_counter()
    try:
        response = _invoke()
    except Exception as exc:  # noqa: BLE001
        log_llm_call(node, model, busy, attempts, {}, error=f"{type(exc).__name__}: {exc}",
                     wait_s=time.perf_counter() - start - busy)
        raise LLMError(f"Gemini call failed ({type(exc).__name__}): {exc}") from exc
    log_llm_call(node, model, busy, attempts, usage_from_response(response), wait_s=time.perf_counter() - start - busy)
    return response
