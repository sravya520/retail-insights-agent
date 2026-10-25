import pytest

from ria import config
from ria import token_stats as ts
from ria.prompting import build_messages


def call(node, inp, out, cached=0, lat=1.0, cost=0.001, hit=False):
    return {"node": node, "input_tokens": inp, "output_tokens": out, "cached_tokens": cached, "latency_s": lat,
            "cost_usd": cost, "cache_hit": hit}


def rec(id, route, calls):
    real = [c for c in calls if not c["cache_hit"]]
    return {"id": id, "route": route, "calls": calls, "usage": {
        "llm_calls": len(real), "cache_hits": len(calls) - len(real),
        "input_tokens": sum(c["input_tokens"] for c in real), "output_tokens": sum(c["output_tokens"] for c in real),
        "cached_tokens": sum(c["cached_tokens"] for c in real), "total_tokens": 0,
        "cost_usd": sum(c["cost_usd"] for c in real), "llm_latency_s": sum(c["latency_s"] for c in real)}}


DATA = rec("a", "data", [call("router", 400, 30), call("sql_agent.write", 900, 50), call("sql_agent.explain", 160, 40)])
POLICY = rec("b", "policy", [call("router", 400, 30), call("rag_agent", 550, 700)])
OOS = rec("c", "out_of_scope", [call("router", 380, 20)])


def test_summarise_and_by_route():
    s = ts.summarise([DATA, POLICY, OOS])
    assert s["requests"] == 3 and s["llm_calls"] == 6
    assert s["input_tokens"] == 400 + 900 + 160 + 400 + 550 + 380
    assert s["cost_per_query"] == pytest.approx(0.002) and s["cost_per_1000"] == pytest.approx(2.0)
    routes = ts.by_route([DATA, POLICY, OOS])
    assert list(routes) == ["data", "policy", "out_of_scope"]
    assert routes["data"]["avg_input_per_query"] == 1460 and routes["out_of_scope"]["llm_calls"] == 1


def test_cached_share_and_cache_hits_are_not_counted():
    cached = rec("d", "data", [call("sql_agent.write", 1000, 50, cached=400), call("router", 400, 30, hit=True)])
    s = ts.summarise([cached])
    assert s["cached_tokens"] == 400 and s["cached_share_of_input"] == pytest.approx(0.4)
    assert s["llm_calls"] == 1  # the cache hit adds nothing
    assert ts.by_node([cached])["sql_agent.write"]["calls"] == 1 and "router" not in ts.by_node([cached])


def test_component_shares():
    sizes = {"schema": 500, "few_shot": 200, "sql_system": 876}
    shares = ts.component_shares([DATA, POLICY, OOS], sizes, {"b": 300})
    assert shares["schema_description"]["tokens"] == 500  # one sql write call
    assert shares["few_shot_examples"]["tokens"] == 200
    assert shares["other_sql_instructions"]["tokens"] == 176
    assert shares["retrieved_policy_context"]["tokens"] == 300  # one rag call with 300 context tokens
    total_in = 400 + 900 + 160 + 400 + 550 + 380
    assert shares["schema_description"]["share_of_input"] == pytest.approx(500 / total_in)
    assert shares["total_input_tokens"] == total_in


def test_repair_calls_resend_the_schema():
    repaired = rec("e", "data", [call("sql_agent.write", 900, 50), call("sql_agent.repair", 1000, 50)])
    shares = ts.component_shares([repaired], {"schema": 500, "few_shot": 200, "sql_system": 876}, {})
    assert shares["schema_description"]["tokens"] == 1000


def test_prompt_layouts(monkeypatch):
    monkeypatch.setattr(config, "PROMPT_LAYOUT", "stable_first")
    assert build_messages("STABLE", "VAR") == [("system", "STABLE"), ("human", "VAR")]
    monkeypatch.setattr(config, "PROMPT_LAYOUT", "question_first")
    assert build_messages("STABLE", "VAR") == [("human", "VAR\n\nSTABLE")]
    assert config.PROMPT_LAYOUT == "question_first"
    monkeypatch.undo()
    assert config.PROMPT_LAYOUT == "stable_first"  # production default
