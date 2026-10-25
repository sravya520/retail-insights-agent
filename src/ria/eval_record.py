"""Turn a graph result (the dict returned by graph.ask) into a saved record. Shared by the eval scripts."""
from __future__ import annotations

from typing import Any

GOLD_FIELDS = ("id", "split", "question", "expected_route", "expected_trajectory", "ordered", "gold_answer",
               "gold_citations", "gold_text")


def record_from_state(q: dict[str, Any], out: dict[str, Any]) -> dict[str, Any]:
    sql, rag = out.get("sql"), out.get("rag")
    rec = {k: q.get(k) for k in GOLD_FIELDS}
    rec.update(
        standalone_question=out.get("standalone_question"), data_question=out.get("data_question"),
        policy_question=out.get("policy_question"), route=out.get("route"), trajectory=out.get("trajectory"),
        answer=out.get("answer", ""), draft=out.get("draft", ""), verified=out.get("verified"),
        removed=(out.get("verifier") or {}).get("removed", []), latency_s=out.get("latency_s"), error=None,
        usage=out.get("usage"), calls=out.get("calls"),
        sql={k: sql.get(k) for k in ("sql", "columns", "rows", "error", "repaired", "repair_log")} if sql else None,
        rag={"chunks": rag["chunks"], "citations": rag["citations"]} if rag else None)
    return rec
