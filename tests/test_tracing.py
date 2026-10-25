import json
import sys

import duckdb
import pytest

from ria import config, tracing
from ria.llm import invoke_with_retry
from ria.results_md import FrozenSectionError, update_section
from ria.sql_agent import SQLAgent


def lines(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_cost_math_uses_cached_price_and_unknown_model_is_free():
    usage = {"input_tokens": 1000, "output_tokens": 100, "cached_tokens": 400}
    expected = (600 * 0.30 + 400 * 0.03 + 100 * 2.50) / 1e6
    assert tracing.cost_usd("gemini-3.5-flash-lite", usage) == pytest.approx(expected)
    assert tracing.cost_usd("no-such-model", usage) == 0.0
    assert tracing.cost_usd("gemini-3.5-flash-lite", {}) == 0.0


def test_price_table_has_the_three_prices_for_every_model():
    for model, price in config.PRICES.items():
        assert set(price) == {"input", "output", "cached_input"} and price["cached_input"] < price["input"] < price["output"], model


class _Reply:
    def __init__(self, text, usage):
        self.content = text
        self.usage_metadata = usage


class UsageLLM:
    model = "gemini-3.5-flash-lite"

    def invoke(self, messages):
        return _Reply("ok", {"input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100,
                             "input_token_details": {"cache_read": 400}, "output_token_details": {"reasoning": 60}})


def test_call_log_has_tokens_cost_and_request_totals_exclude_cache_hits(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", True)
    with tracing.request_scope() as calls:
        invoke_with_retry(UsageLLM(), [("human", "q")], node="n1")
        invoke_with_retry(UsageLLM(), [("human", "q")], node="n1")  # cache hit
        invoke_with_retry(UsageLLM(), [("human", "other")], node="n2")
    first = lines(tracing.LLM_CALLS_PATH)[0]
    assert first["input_tokens"] == 1000 and first["cached_tokens"] == 400 and first["reasoning_tokens"] == 60
    assert first["cost_usd"] == pytest.approx((600 * 0.30 + 400 * 0.03 + 100 * 2.5) / 1e6, rel=1e-4)
    t = tracing.totals(calls)
    assert t["llm_calls"] == 2 and t["cache_hits"] == 1 and t["input_tokens"] == 2000 and t["cached_tokens"] == 800
    assert t["total_tokens"] == 2200 and t["cost_usd"] == pytest.approx(2 * first["cost_usd"], rel=1e-4)


def test_calls_outside_a_scope_are_not_collected(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)
    invoke_with_retry(UsageLLM(), [("human", "q")], node="n")
    with tracing.request_scope() as calls:
        pass
    assert calls == []


def test_langfuse_is_skipped_without_keys(monkeypatch):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert tracing.langfuse_callbacks() == []


def test_langfuse_failure_never_raises(monkeypatch, capsys):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "langfuse", None)  # makes "import langfuse..." fail
    monkeypatch.setattr(tracing, "_langfuse_warned", False)
    assert tracing.langfuse_callbacks() == []
    assert "local tracing only" in capsys.readouterr().out


class Scripted:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        return _Reply(self.replies.pop(0), {})


def make_con():
    con = duckdb.connect()
    con.execute("CREATE TABLE sales AS SELECT 'I1' AS invoice_no, 6.0 AS revenue, 2010 AS invoice_year")
    return con


def test_sql_repair_reason_is_logged_for_validation_failure_that_then_succeeds(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)
    llm = Scripted(["SELECT 28 AS window_days FROM sales", "SELECT SUM(revenue) AS total FROM sales", "Total is 6."])
    res = SQLAgent(con=make_con(), llm=llm).run("What is revenue?")
    assert res.ok and res.repaired and len(res.repair_log) == 1
    entry = res.repair_log[0]
    assert entry["stage"] == "validation" and "Numeric constants" in entry["error"] and entry["repair_succeeded"] is True
    logged = lines(tracing.SQL_REPAIRS_PATH)
    assert len(logged) == 1 and logged[0]["failed_sql"] == "SELECT 28 AS window_days FROM sales"


def test_sql_repair_reason_is_logged_for_execution_failure_that_fails_again(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)
    llm = Scripted(["SELECT CAST(invoice_no AS INTEGER) AS x FROM sales", "SELECT CAST(invoice_no AS INTEGER) AS y FROM sales"])
    res = SQLAgent(con=make_con(), llm=llm).run("q")
    assert not res.ok and res.repair_log[0]["stage"] == "execution"
    assert res.repair_log[0]["repair_succeeded"] is False
    assert lines(tracing.SQL_REPAIRS_PATH)[0]["repair_succeeded"] is False


def test_no_repair_means_no_repair_log(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)
    llm = Scripted(["SELECT SUM(revenue) AS total FROM sales", "Total is 6."])
    res = SQLAgent(con=make_con(), llm=llm).run("q")
    assert res.ok and not res.repaired and res.repair_log == []
    assert not tracing.SQL_REPAIRS_PATH.exists()


def test_frozen_results_sections_cannot_be_overwritten(tmp_path):
    path = tmp_path / "results.md"
    path.write_text("# Results\n", encoding="utf-8")
    with pytest.raises(FrozenSectionError):
        update_section("eval_v1", "new numbers", path=path)
    update_section("eval_v2", "fine", path=path)
    assert "fine" in path.read_text(encoding="utf-8")
