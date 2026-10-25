"""Pure metric and gate functions for the Phase 4 eval. No model calls here, so they are unit-tested.

A record is one dict per question (see evals/run_evals.py). Keys used here:
  id, split, expected_route, expected_trajectory, route, trajectory, answer, standalone_question,
  gold_answer, ordered, gold_citations, sql {rows, error}, rag {chunks}, verified, removed, judge, error, latency_s
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any, Iterable

from ria.graph import REFUSAL_MESSAGE
from ria.sql_agent import SQLValidationError, validate_sql
from ria.sql_compare import results_match
from ria.verifier import verify_answer

Record = dict[str, Any]


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def wilson_lower(successes: int, n: int, z: float = 1.96) -> float | None:
    """Lower end of the 95% Wilson interval. Shows how little a perfect score on a small set proves."""
    if not n:
        return None
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (centre - margin) / denom


def router_accuracy(records: Iterable[Record]) -> tuple[int, int]:
    rs = list(records)
    return sum(r.get("route") == r["expected_route"] for r in rs), len(rs)


def trajectory_accuracy(records: Iterable[Record]) -> tuple[int, int, Counter]:
    """Exact match of the node sequence. Also returns a Counter of the wrong paths (as strings)."""
    rs = list(records)
    ok = 0
    wrong: Counter = Counter()
    for r in rs:
        if r.get("trajectory") == r["expected_trajectory"]:
            ok += 1
        else:
            got = " > ".join(r["trajectory"]) if r.get("trajectory") else f"(no result: {r.get('error', 'unknown')[:40]})"
            wrong[f"expected {' > '.join(r['expected_trajectory'])} | got {got}"] += 1
    return ok, len(rs), wrong


def sql_execution_accuracy(records: Iterable[Record]) -> tuple[int, int]:
    """Questions with a gold SQL answer (data and both). Agent rows must match gold within 0.5%."""
    rs = [r for r in records if r.get("gold_answer") is not None]
    ok = 0
    for r in rs:
        sql = r.get("sql")
        if sql and not sql.get("error") and results_match(r["gold_answer"], sql["rows"], ordered=r.get("ordered", False)):
            ok += 1
    return ok, len(rs)


def retrieval_recall_at_5(records: Iterable[Record]) -> tuple[int, int]:
    """Policy and both questions with gold citations: any gold clause among the 5 chunks the RAG agent retrieved."""
    rs = [r for r in records if r.get("gold_citations")]
    ok = 0
    for r in rs:
        gold = {(g["doc"], g["clause_id"]) for g in r["gold_citations"]}
        got = {(c["doc"], c["clause_id"]) for c in ((r.get("rag") or {}).get("chunks") or [])}
        ok += bool(gold & got)
    return ok, len(rs)


def refusal_rate(records: Iterable[Record]) -> tuple[int, int]:
    rs = [r for r in records if r["expected_route"] == "out_of_scope"]
    return sum(r.get("route") == "out_of_scope" and r.get("answer") == REFUSAL_MESSAGE for r in rs), len(rs)


def judge_means(records: Iterable[Record]) -> dict[str, Any]:
    judged = [r["judge"] for r in records if r.get("judge")]
    if not judged:
        return {"n": 0, "faithfulness": None, "relevance": None, "faithful_share": None}
    return {"n": len(judged),
            "faithfulness": statistics.mean(j["faithfulness"] for j in judged),
            "relevance": statistics.mean(j["relevance"] for j in judged),
            "faithful_share": sum(j["faithfulness"] >= 4 for j in judged) / len(judged)}


def verifier_stats(records: Iterable[Record]) -> dict[str, int]:
    rs = [r for r in records if r.get("verified") is not None]
    removed = [x for r in rs for x in (r.get("removed") or [])]
    return {"answers_checked": len(rs), "answers_changed": sum(r["verified"] is False for r in rs),
            "sentences_removed": len(removed),
            "removed_unsupported_number": sum(x["reason"] == "unsupported_number" for x in removed),
            "removed_invalid_citation": sum(x["reason"] == "invalid_citation" for x in removed)}


def median_latency(records: Iterable[Record]) -> float | None:
    values = [r["latency_s"] for r in records if r.get("latency_s") is not None and not r.get("error")]
    return statistics.median(values) if values else None


# ---------------------------------------------------------------- gates

def gate_no_unsupported_content(records: Iterable[Record]) -> tuple[int, int, list[str]]:
    """GATE 1. Re-check every final answer against its own evidence. Passing means 0 violations.

    A violation is a final answer in which the verifier, run again, still finds an unsupported number or a citation
    that matches no retrieved chunk. This uses the verifier's own rules, so it shows the verifier output is stable.
    It does not prove the answer is true (see the limits in docs/results.md).
    """
    bad, checked = [], 0
    for r in records:
        if r.get("route") in (None, "out_of_scope") or not r.get("answer") or r.get("error"):
            continue
        checked += 1
        again = verify_answer(r["answer"], question=r.get("standalone_question") or r["question"],
                              sql_rows=(r.get("sql") or {}).get("rows") or [],
                              chunks=(r.get("rag") or {}).get("chunks") or [])
        if again.removed:
            bad.append(r["id"])
    return len(bad), checked, bad


def gate_out_of_scope_refused(records: Iterable[Record]) -> tuple[int, int, list[str]]:
    """GATE 2. Every out of scope question must be refused: route out_of_scope, trajectory [router, refuse], fixed message."""
    rs = [r for r in records if r["expected_route"] == "out_of_scope"]
    bad = [r["id"] for r in rs if not (r.get("route") == "out_of_scope" and r.get("trajectory") == ["router", "refuse"]
                                      and r.get("answer") == REFUSAL_MESSAGE)]
    return len(bad), len(rs), bad


def gate_unsafe_sql_blocked(unsafe_queries: list[str], schema: dict[str, list[str]]) -> tuple[int, int, list[str]]:
    """GATE 3. validate_sql must raise for every unsafe query. No model involved."""
    leaked = []
    for sql in unsafe_queries:
        try:
            validate_sql(sql, schema)
            leaked.append(sql)
        except SQLValidationError:
            pass
    return len(leaked), len(unsafe_queries), leaked


def gold_queries_not_blocked(gold_sqls: list[str], schema: dict[str, list[str]]) -> list[str]:
    """Observation: gold queries the guardrail wrongly blocks (should be empty)."""
    blocked = []
    for sql in gold_sqls:
        try:
            validate_sql(sql, schema)
        except SQLValidationError:
            blocked.append(sql)
    return blocked


# ---------------------------------------------------------------- human label template and judge agreement

def select_for_labeling(records: list[Record], n: int = 20, seed: int = 7) -> list[Record]:
    """Pick n answered, judged records for hand labelling.

    Order of preference: answers the verifier changed, then answers the judge scored below 5 on faithfulness
    (borderline cases), then a mix of routes, preferring held-out questions. The final order is shuffled with a fixed seed so the order does not hint at the route.
    """
    import random
    rng = random.Random(seed)
    pool = [r for r in records if r.get("judge") and r.get("answer") and r.get("route") not in (None, "out_of_scope")]
    changed = [r for r in pool if r.get("verified") is False]
    rng.shuffle(changed)
    chosen = changed[: n // 2]
    borderline = [r for r in pool if r not in chosen and r["judge"]["faithfulness"] < 5]
    rng.shuffle(borderline)
    chosen += borderline[: n // 2 - len(chosen) + n // 4]
    rest = [r for r in pool if r not in chosen]
    rng.shuffle(rest)
    rest.sort(key=lambda r: r["split"] != "heldout")  # held-out first, stable within each group
    by_route: dict[str, list[Record]] = {"both": [], "policy": [], "data": []}
    for r in rest:
        by_route.setdefault(r["route"], []).append(r)
    order = ["both", "policy", "data"]
    while len(chosen) < n and any(by_route.get(k) for k in order):
        for k in order:
            if by_route.get(k) and len(chosen) < n:
                chosen.append(by_route[k].pop(0))
    rng.shuffle(chosen)
    return chosen


def label_template_row(r: Record, run_id: str) -> dict[str, Any]:
    from ria.judge import evidence_text
    sql, chunks = r.get("sql"), (r.get("rag") or {}).get("chunks")
    return {"id": r["id"], "run_id": run_id, "route": r["route"], "question": r["question"], "answer": r["answer"],
            "evidence_sql": ({"sql": sql["sql"], "columns": sql["columns"], "rows": sql["rows"][:10]} if sql else None),
            "evidence_chunks": [{"doc": c["doc"], "clause_id": c["clause_id"], "page": c["page"], "text": c["text"]} for c in chunks] if chunks else None,
            "faithful": None, "notes": ""}


def judge_agreement(labels: list[dict[str, Any]], judge_faithful: dict[str, bool]) -> dict[str, Any]:
    """Agreement between human labels (faithful true or false) and the judge (faithfulness >= 4)."""
    pairs = [(bool(l["faithful"]), judge_faithful[l["id"]], l["id"]) for l in labels if l["id"] in judge_faithful]
    n = len(pairs)
    tp = sum(h and j for h, j, _ in pairs)
    tn = sum((not h) and (not j) for h, j, _ in pairs)
    fp = sum((not h) and j for h, j, _ in pairs)   # judge says faithful, human says not
    fn = sum(h and (not j) for h, j, _ in pairs)
    agree = (tp + tn) / n if n else None
    expected = (((tp + fn) * (tp + fp)) + ((tn + fp) * (tn + fn))) / (n * n) if n else 0
    kappa = (agree - expected) / (1 - expected) if n and expected < 1 else None
    return {"n": n, "agreement": agree, "kappa": kappa, "judge_faithful_human_not": fp, "judge_not_human_faithful": fn,
            "both_faithful": tp, "both_not_faithful": tn, "disagreements": [i for h, j, i in pairs if h != j]}
