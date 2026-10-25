"""Gemini client with Tenacity retry. Only 429 and 5xx are retried, all other errors fail fast."""
from __future__ import annotations

import re
from typing import Any

from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

from ria import config


class LLMError(RuntimeError):
    """A Gemini call failed in a way that retrying will not fix."""


_STATUS_RE = re.compile(r"\b(429|5\d\d)\b")


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits (429) and server errors (5xx)."""
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if isinstance(code, int):
        return code == 429 or 500 <= code < 600
    name = type(exc).__name__
    if name in {"ResourceExhausted", "ServiceUnavailable", "InternalServerError", "DeadlineExceeded"}:
        return True
    return bool(_STATUS_RE.search(str(exc)))


def get_llm(model: str | None = None, temperature: float | None = None) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GOOGLE_API_KEY:
        raise LLMError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key.")
    return ChatGoogleGenerativeAI(
        model=model or config.GEMINI_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        google_api_key=config.GOOGLE_API_KEY,
        max_retries=1,  # Tenacity owns the retry policy
    )


@retry(
    retry=retry_if_exception(is_retryable),
    wait=wait_random_exponential(multiplier=1, max=30),  # exponential backoff with jitter
    stop=stop_after_attempt(config.LLM_MAX_ATTEMPTS),
    reraise=True,
)
def _invoke(llm: Any, messages: Any) -> Any:
    return llm.invoke(messages)


def invoke_with_retry(llm: Any, messages: Any) -> Any:
    try:
        return _invoke(llm, messages)
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"Gemini call failed ({type(exc).__name__}): {exc}") from exc
