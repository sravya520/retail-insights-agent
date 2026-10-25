"""How the Streamlit app reaches the agent.

  * RIA_API_URL set   -> HttpBackend: calls the FastAPI app (used in the Docker image, where both run together)
  * RIA_API_URL unset -> LocalBackend: runs the graph inside the Streamlit process (used on Streamlit Community Cloud)

Both raise BackendError(kind, message), so the UI has one error path.
Kinds: limit (a usage limit stopped the request), quota (the model's daily quota is used up),
model (the model call failed), invalid (bad question), unavailable (the API cannot be reached).
"""
from __future__ import annotations

import os
from typing import Any

from ria import config


class BackendError(Exception):
    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind, self.message = kind, message


class LocalBackend:
    def __init__(self) -> None:
        from ria.service import AskService
        self.service = AskService()

    def ask(self, question: str, thread_id: str | None, session_id: str) -> dict[str, Any]:
        from ria.service import InvalidQuestion, LimitReached, UpstreamError, UpstreamQuotaError
        try:
            return self.service.ask(question, thread_id, session_id)
        except InvalidQuestion as exc:
            raise BackendError("invalid", str(exc)) from exc
        except LimitReached as exc:
            raise BackendError("limit", exc.decision.message) from exc
        except UpstreamQuotaError as exc:
            raise BackendError("quota", "The language model's free daily quota is used up. Live questions are paused until it resets.") from exc
        except UpstreamError as exc:
            raise BackendError("model", f"The language model call failed: {str(exc)[:200]}") from exc

    def limits(self, session_id: str) -> dict[str, Any]:
        return self.service.limit_info(session_id)

    def examples(self) -> list[dict[str, Any]]:
        from ria.service import load_examples
        return load_examples()


class HttpBackend:
    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")

    def _client(self):
        import httpx
        return httpx.Client(timeout=120.0)

    def ask(self, question: str, thread_id: str | None, session_id: str) -> dict[str, Any]:
        import httpx
        try:
            with self._client() as client:
                r = client.post(f"{self.url}/ask", json={"question": question, "thread_id": thread_id},
                                headers={"X-Session-Id": session_id})
        except httpx.HTTPError as exc:
            raise BackendError("unavailable", f"Cannot reach the API at {self.url}: {exc}") from exc
        if r.status_code == 200:
            return r.json()
        detail = r.json().get("detail", {}) if r.headers.get("content-type", "").startswith("application/json") else {}
        message = detail.get("message") if isinstance(detail, dict) else str(detail)
        kind = {429: "limit", 503: "quota", 502: "model", 422: "invalid"}.get(r.status_code, "unavailable")
        raise BackendError(kind, message or f"The API returned status {r.status_code}.")

    def limits(self, session_id: str) -> dict[str, Any]:
        try:
            with self._client() as client:
                return client.get(f"{self.url}/limits", headers={"X-Session-Id": session_id}).json()
        except Exception:  # noqa: BLE001 - the limits display must never break the page
            return {}

    def examples(self) -> list[dict[str, Any]]:
        from ria.service import load_examples
        return load_examples()  # same file as the API serves, read directly so opening examples needs no request


_backend: Any = None


def get_backend() -> Any:
    global _backend
    if _backend is None:
        url = os.getenv("RIA_API_URL", "").strip()
        _backend = HttpBackend(url) if url else LocalBackend()
    return _backend
