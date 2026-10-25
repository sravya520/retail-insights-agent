"""LLM judge for faithfulness and answer relevance. Fixed rubric, scores 1 to 5, temperature 0.

The judge sees the question, the answer and the evidence the system had (SQL rows and or policy passages).
It does not see any gold answer. A judge model that is different from the answering model is preferred. The
model that scored each item is stored with the score, because the pool falls back to the next model when a
model runs out of free daily quota.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel, Field

from ria import config
from ria.llm import LLMError, get_llm, invoke_structured

FAITHFUL_THRESHOLD = 4  # judge says "faithful" when faithfulness >= 4 (fixed before any human labels exist)

RUBRIC = """You are a strict grader of answers from a retail question answering system.
You see the question, the answer, and the evidence the system had: SQL result rows and or policy passages.
Grade only against the evidence. Ignore your own knowledge.

faithfulness (is every claim in the answer supported by the evidence?)
  5 every claim and every number is directly supported by the evidence
  4 supported, with only trivial wording additions
  3 one minor detail is not supported by the evidence
  2 several claims are unsupported, or one important claim is
  1 mostly unsupported, or it contradicts the evidence

relevance (does the answer address the question that was asked?)
  5 answers exactly what was asked, nothing missing, no extra cases
  4 answers it with a small gap or a little extra
  3 answers only part of the question
  2 mostly off topic
  1 does not answer the question

List any unsupported claims in unsupported_claims (empty text if none)."""


class JudgeScores(BaseModel):
    faithfulness: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    unsupported_claims: str = ""


@dataclass
class JudgeResult:
    faithfulness: int
    relevance: int
    unsupported_claims: str
    model: str

    @property
    def faithful(self) -> bool:
        return self.faithfulness >= FAITHFUL_THRESHOLD


def evidence_text(sql: dict[str, Any] | None, chunks: list[dict[str, Any]] | None, max_rows: int = 30) -> str:
    """Plain text of the evidence the system had. Used by the judge and in the human label template."""
    parts = []
    if sql:
        rows = sql.get("rows") or []
        header = " | ".join(sql.get("columns") or [])
        body = "\n".join(" | ".join(str(v) for v in r) for r in rows[:max_rows])
        parts.append(f"SQL result ({len(rows)} rows):\n{header}\n{body}" if rows else "SQL result: no rows")
    for c in chunks or []:
        parts.append(f"Policy passage [{c['doc']} §{c['clause_id']}, p.{c['page']}]: {c['text']}")
    return "\n\n".join(parts) or "(no evidence)"


def is_unavailable(exc: BaseException) -> bool:
    """True when a judge model should be dropped: daily quota used up, unknown model, or no access."""
    return bool(re.search(r"PerDay|per day|NOT_FOUND|no longer available|PERMISSION_DENIED", str(exc)))


@dataclass
class JudgePool:
    """Tries judge models in order. A model that is unavailable is dropped for the rest of the run."""
    models: list[str]
    factory: Callable[..., Any] = get_llm
    dead: set[str] = field(default_factory=set)

    def active(self) -> list[str]:
        return [m for m in self.models if m not in self.dead]

    def score(self, question: str, answer: str, evidence: str) -> JudgeResult:
        messages = [("system", RUBRIC),
                    ("human", f"Question:\n{question}\n\nAnswer:\n{answer}\n\nEvidence:\n{evidence}")]
        last: Exception | None = None
        for model in self.active():
            try:
                llm = self.factory("judge", model=model, temperature=0.0)
                scores = invoke_structured(llm, messages, JudgeScores, node="judge")
                return JudgeResult(scores.faithfulness, scores.relevance, scores.unsupported_claims, model)
            except LLMError as exc:
                last = exc
                if is_unavailable(exc):
                    self.dead.add(model)
                    continue
                raise
        raise LLMError(f"No judge model is available (daily quota used up on: {sorted(self.dead)}). Last error: {last}")
