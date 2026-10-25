"""Token and cost statistics from saved request records. Pure functions, no model calls.

A request record has: route, usage {llm_calls, cache_hits, input_tokens, output_tokens, cached_tokens, total_tokens,
cost_usd, llm_latency_s}, calls [{node, input_tokens, output_tokens, cached_tokens, latency_s, cost_usd}],
and for policy answers rag {chunks}. See ria/eval_record.py.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable

Record = dict[str, Any]
ROUTES = ("data", "policy", "both", "out_of_scope")


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarise(records: Iterable[Record]) -> dict[str, Any]:
    """Totals and averages for a set of requests."""
    rs = [r for r in records if r.get("usage")]
    n = len(rs)
    inp = sum(r["usage"]["input_tokens"] for r in rs)
    out = sum(r["usage"]["output_tokens"] for r in rs)
    cached = sum(r["usage"]["cached_tokens"] for r in rs)
    cost = sum(r["usage"]["cost_usd"] for r in rs)
    call_lat = [c["latency_s"] for r in rs for c in r.get("calls", []) if not c.get("cache_hit") and c.get("latency_s") is not None]
    return {
        "requests": n, "llm_calls": sum(r["usage"]["llm_calls"] for r in rs),
        "input_tokens": inp, "output_tokens": out, "cached_tokens": cached,
        "cached_share_of_input": (cached / inp) if inp else 0.0,
        "avg_input_per_query": inp / n if n else 0.0, "avg_output_per_query": out / n if n else 0.0,
        "cost_per_query": cost / n if n else 0.0, "cost_per_1000": 1000 * cost / n if n else 0.0,
        "median_call_latency_s": statistics.median(call_lat) if call_lat else 0.0,
        "median_query_model_latency_s": statistics.median([r["usage"]["llm_latency_s"] for r in rs]) if rs else 0.0,
    }


def by_route(records: Iterable[Record]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        if r.get("usage") and r.get("route"):
            groups[r["route"]].append(r)
    return {route: summarise(groups[route]) for route in ROUTES if route in groups}


def by_node(records: Iterable[Record]) -> dict[str, dict[str, float]]:
    """Average input and output tokens per call, per node."""
    groups: dict[str, list[Record]] = defaultdict(list)
    for r in records:
        for c in r.get("calls", []):
            if not c.get("cache_hit"):
                groups[c["node"]].append(c)
    return {node: {"calls": len(cs), "avg_input": _mean([c["input_tokens"] for c in cs]),
                   "avg_output": _mean([c["output_tokens"] for c in cs])} for node, cs in sorted(groups.items())}


def component_shares(records: Iterable[Record], sizes: dict[str, int], context_tokens: dict[str, int]) -> dict[str, Any]:
    """Share of all tokens spent on the schema, few-shot examples and retrieved context.

    sizes: token counts of fixed prompt parts: schema, few_shot, sql_system (whole SQL system prompt).
    context_tokens: request id -> tokens of the retrieved policy passages that were sent to the RAG agent.
    A fixed part is sent once in every sql_agent.write and sql_agent.repair call. The retrieved context is sent in
    every rag_agent call of that request. Counts are approximate (parts are counted separately from the whole prompt).
    """
    rs = [r for r in records if r.get("usage")]
    total_in = sum(r["usage"]["input_tokens"] for r in rs)
    total_all = total_in + sum(r["usage"]["output_tokens"] for r in rs)
    sql_calls = sum(1 for r in rs for c in r.get("calls", []) if c["node"] in ("sql_agent.write", "sql_agent.repair") and not c.get("cache_hit"))
    schema = sql_calls * sizes["schema"]
    few_shot = sql_calls * sizes["few_shot"]
    sql_rest = sql_calls * max(sizes["sql_system"] - sizes["schema"] - sizes["few_shot"], 0)
    retrieved = 0
    for r in rs:
        rag_calls = sum(1 for c in r.get("calls", []) if c["node"] == "rag_agent" and not c.get("cache_hit"))
        retrieved += rag_calls * context_tokens.get(r["id"], 0)

    def pair(x: int) -> dict[str, float]:
        return {"tokens": x, "share_of_input": x / total_in if total_in else 0.0, "share_of_all_tokens": x / total_all if total_all else 0.0}

    return {"total_input_tokens": total_in, "total_tokens": total_all, "schema_description": pair(schema),
            "few_shot_examples": pair(few_shot), "other_sql_instructions": pair(sql_rest),
            "retrieved_policy_context": pair(retrieved)}
