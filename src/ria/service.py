"""The service layer shared by the FastAPI app and the Streamlit app.

AskService owns the LangGraph app (built on first use), the usage limiter, and the shape of an answer. Both front
ends call `ask()` and get the same dictionary back: answer, route, trajectory, SQL, citations, verified flag,
tokens, cost and latency.
"""
from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from ria import config
from ria.limits import LimitDecision, UsageLimiter, limiter_from_config
from ria.llm import LLMError
from ria.rag_agent import parse_citations

MAX_ROWS_RETURNED = 20


class LimitReached(Exception):
    """A usage limit stopped the request. `decision` says which one and why."""

    def __init__(self, decision: LimitDecision) -> None:
        super().__init__(decision.message)
        self.decision = decision


class UpstreamQuotaError(Exception):
    """The language model API reported that its daily quota is used up."""


class UpstreamError(Exception):
    """The language model call failed for another reason."""


class InvalidQuestion(ValueError):
    pass


def validate_question(question: str) -> str:
    q = (question or "").strip()
    if not q:
        raise InvalidQuestion("Please type a question.")
    if len(q) > config.MAX_QUESTION_CHARS:
        raise InvalidQuestion(f"Please keep the question under {config.MAX_QUESTION_CHARS} characters.")
    return q


def shape_answer(state: dict[str, Any]) -> dict[str, Any]:
    """Turn the graph state (from graph.ask) into the dictionary the API and the UI show."""
    sql, rag = state.get("sql"), state.get("rag")
    answer = state.get("answer", "")
    chunk_text = {(c["doc"], c["clause_id"]): c for c in ((rag or {}).get("chunks") or [])}
    citations = []
    for cite in parse_citations(answer):
        chunk = chunk_text.get((cite.doc, cite.clause_id))
        citations.append({"doc": cite.doc, "clause_id": cite.clause_id, "page": cite.page,
                          "text": chunk["text"] if chunk else ""})
    usage = state.get("usage") or {}
    return {
        "answer": answer,
        "route": state.get("route"),
        "trajectory": state.get("trajectory") or [],
        "standalone_question": state.get("standalone_question"),
        "sql": (sql or {}).get("sql") or None,
        "sql_columns": (sql or {}).get("columns") or [],
        "sql_rows": [list(r) for r in ((sql or {}).get("rows") or [])[:MAX_ROWS_RETURNED]],
        "sql_row_count": len((sql or {}).get("rows") or []),
        "sql_repaired": bool((sql or {}).get("repaired")),
        "citations": citations,
        "verified": state.get("verified"),
        "removed": (state.get("verifier") or {}).get("removed", []),
        "tokens": {"input": usage.get("input_tokens", 0), "output": usage.get("output_tokens", 0),
                   "cached": usage.get("cached_tokens", 0), "total": usage.get("total_tokens", 0)},
        "cost_usd": usage.get("cost_usd", 0.0),
        "latency_s": state.get("latency_s", 0.0),
        "model_latency_s": usage.get("llm_latency_s", 0.0),
        "llm_calls": usage.get("llm_calls", 0),
        "model": config.GEMINI_MODEL,
        "source": "live",
    }


class AskService:
    def __init__(self, limiter: UsageLimiter | None = None, app_factory: Callable[[], Any] | None = None) -> None:
        self.limiter = limiter or limiter_from_config()
        self._app_factory = app_factory
        self._app: Any = None
        self._lock = threading.Lock()

    def _get_app(self) -> Any:
        with self._lock:
            if self._app is None:
                if self._app_factory is not None:
                    self._app = self._app_factory()
                else:
                    from ria.bootstrap import ensure_data
                    from ria.graph import build_graph, default_services
                    ensure_data()
                    self._app = build_graph(default_services())
            return self._app

    def ask(self, question: str, thread_id: str | None = None, session_id: str = "anonymous") -> dict[str, Any]:
        """Answer one live question. Raises InvalidQuestion, LimitReached, UpstreamQuotaError or UpstreamError."""
        from ria.graph import ask as graph_ask

        q = validate_question(question)
        decision = self.limiter.acquire(session_id)
        if not decision.allowed:
            raise LimitReached(decision)
        thread = thread_id or f"{session_id}-{uuid.uuid4().hex[:8]}"
        try:
            state = graph_ask(self._get_app(), q, thread_id=thread, label="demo")
        except LLMError as exc:
            text = str(exc)
            if "PerDay" in text or "per day" in text:
                self.limiter.pause_until_reset()
                raise UpstreamQuotaError(text) from exc
            self.limiter.refund(session_id)
            raise UpstreamError(text) from exc
        except Exception:
            self.limiter.refund(session_id)
            raise
        result = shape_answer(state)
        result["thread_id"] = thread
        result["limits"] = self.limit_info(session_id)
        return result

    def limit_info(self, session_id: str) -> dict[str, Any]:
        d = self.limiter.status(session_id)
        return {"session_remaining": d.session_remaining, "daily_remaining": d.daily_remaining,
                "session_limit": self.limiter.session_limit, "daily_limit": self.limiter.daily_limit,
                "live_available": d.allowed, "reason": d.reason, "message": d.message if not d.allowed else "",
                "resets_at_utc": d.resets_at_utc}


def load_examples(path: Path | None = None) -> list[dict[str, Any]]:
    """The pre-computed answers from real eval-v1 runs. They cost no quota."""
    path = path or config.EXAMPLES_PATH
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["examples"]
