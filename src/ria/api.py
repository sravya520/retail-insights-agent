"""FastAPI app.

  POST /ask      question + thread_id -> answer, route, trajectory, SQL, citations, verified flag, tokens, cost, latency
  GET  /health   liveness plus what data and settings are present (no model call)
  GET  /limits   how many live questions are left for the caller and for the demo
  GET  /examples the pre-computed answers from real eval-v1 runs (cost no quota)

Run: uvicorn ria.api:app --port 8000
The caller is identified by the X-Session-Id header, or by its IP address when the header is missing.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ria import config
from ria.service import (AskService, InvalidQuestion, LimitReached, UpstreamError, UpstreamQuotaError,
                         load_examples)
from ria.sql_db import DB_PATH


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=config.MAX_QUESTION_CHARS, description="A question in plain English")
    thread_id: str | None = Field(default=None, max_length=64, description="Reuse the same id to ask follow-up questions")


class Citation(BaseModel):
    doc: str
    clause_id: str
    page: int
    text: str = ""


class Tokens(BaseModel):
    input: int
    output: int
    cached: int
    total: int


class AskResponse(BaseModel):
    answer: str
    route: str | None
    trajectory: list[str]
    standalone_question: str | None = None
    sql: str | None = None
    sql_columns: list[str] = []
    sql_rows: list[list[Any]] = []
    sql_row_count: int = 0
    sql_repaired: bool = False
    explanation_failed: bool = False
    citations: list[Citation] = []
    verified: bool | None
    removed: list[dict[str, Any]] = []
    tokens: Tokens
    cost_usd: float
    latency_s: float
    model_latency_s: float = 0.0
    llm_calls: int = 0
    model: str
    source: str = "live"
    thread_id: str
    limits: dict[str, Any]


def create_app(service: AskService | None = None) -> FastAPI:
    app = FastAPI(title="Retail Insights Agent", version="1.0",
                  description="Ask plain-English questions about a retail company's sales data and policy documents. "
                              "The policy documents are synthetic. Live questions are limited to protect a free quota.")
    holder: dict[str, AskService | None] = {"service": service}

    def get_service() -> AskService:
        if holder["service"] is None:
            holder["service"] = AskService()
        return holder["service"]

    def caller(request: Request, x_session_id: str | None) -> str:
        return (x_session_id or (request.client.host if request.client else "anonymous"))[:64]

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "model": config.GEMINI_MODEL, "api_key_set": bool(config.GOOGLE_API_KEY),
                "database_present": DB_PATH.exists(), "vector_index_present": config.CHUNKS_PATH.exists(),
                "examples_present": config.EXAMPLES_PATH.exists()}

    @app.get("/limits")
    def limits(request: Request, x_session_id: str | None = Header(default=None)) -> dict[str, Any]:
        return get_service().limit_info(caller(request, x_session_id))

    @app.get("/examples")
    def examples() -> dict[str, Any]:
        return {"examples": load_examples()}

    @app.post("/ask", response_model=AskResponse)
    def ask(body: AskRequest, request: Request, x_session_id: str | None = Header(default=None)) -> dict[str, Any]:
        try:
            return get_service().ask(body.question, body.thread_id, caller(request, x_session_id))
        except InvalidQuestion as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_question", "message": str(exc)})
        except LimitReached as exc:
            d = exc.decision
            raise HTTPException(status_code=429, detail={"code": "limit_reached", "reason": d.reason, "message": d.message,
                                                         "resets_at_utc": d.resets_at_utc})
        except UpstreamQuotaError:
            raise HTTPException(status_code=503, detail={"code": "model_quota_exhausted", "message":
                                "The language model's free daily quota is used up. Live questions are paused until it resets."})
        except UpstreamError as exc:
            raise HTTPException(status_code=502, detail={"code": "model_error", "message": f"The language model call failed: {str(exc)[:200]}"})

    return app


app = create_app()
