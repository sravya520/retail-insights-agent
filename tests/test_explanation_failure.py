"""A failed model call must never put raw API error text into an answer.

Covers the bug seen on q10 in run-dev-after-fixes: the explanation step hit the daily quota and the error text became
the answer. Now the answer is a safe message plus the SQL rows, the run is marked explanation_failed, and the error
text appears only in the trace log.
"""
import json

import duckdb

from ria import tracing
from ria.graph import Services, ask, build_graph
from ria.service import shape_answer
from ria.sql_agent import QUERY_FAILED_ANSWER, SQLAgent, rows_answer
from ria.verifier import verify_answer

QUOTA_ERROR = ("429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota. "
               "For more information on this error, head to: https://ai.google.dev/gemini-api/docs/rate-limits. "
               "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
               "limit: 500, model: gemini-3.5-flash-lite', 'quotaId': 'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}}")
ERROR_MARKERS = ("429", "RESOURCE_EXHAUSTED", "http", "quota", "Gemini call failed", "limit: 500")


class _Reply:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {}


class ScriptedLLM:
    """Replies in order. A reply that is an Exception is raised instead (a failed model call)."""
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)

    def invoke(self, messages):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return _Reply(reply)


class FakeRouter:
    model = "fake-router"

    def __init__(self, route, question):
        self.route, self.question = route, question

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class R:
            def invoke(self, messages):
                return {"raw": None, "parsed": schema(route=outer.route, standalone_question=outer.question, reason="t"),
                        "parsing_error": None}
        return R()


def con():
    c = duckdb.connect()
    c.execute("CREATE TABLE sales AS SELECT 18102 AS customer_id, 335993.22 AS revenue, 2010 AS invoice_year")
    return c


def assert_no_error_text(text):
    for marker in ERROR_MARKERS:
        assert marker not in text, f"error text leaked: {marker!r} in {text!r}"


def test_explanation_failure_gives_safe_answer_with_rows_and_keeps_error_for_the_trace():
    llm = ScriptedLLM(["SELECT customer_id, SUM(revenue) AS spend FROM sales GROUP BY customer_id", Exception(QUOTA_ERROR)])
    res = SQLAgent(con=con(), llm=llm).run("Which customer spent the most in 2010?")
    assert res.ok and res.explanation_failed
    assert "429" in res.explanation_error  # kept for the trace log
    assert_no_error_text(res.answer)
    assert "customer_id: 18102" in res.answer and "spend: 335,993.22" in res.answer


def test_the_verifier_never_sees_error_text_and_the_trace_gets_it():
    sql_agent = SQLAgent(con=con(), llm=ScriptedLLM(["SELECT customer_id, SUM(revenue) AS spend FROM sales GROUP BY customer_id",
                                                     Exception(QUOTA_ERROR)]))
    services = Services(FakeRouter("data", "Which customer spent the most in 2010?"), sql_agent, lambda q: None, ScriptedLLM([]))
    out = ask(build_graph(services), "Which customer spent the most in 2010?", "t-expl")
    assert_no_error_text(out["draft"])   # what the verifier checked
    assert_no_error_text(out["answer"])  # what the user sees
    assert out["verified"] is True and out["verifier"]["removed"] == []  # every number came from the rows
    assert out["sql"]["explanation_failed"] is True
    trace = json.loads(tracing.TRACES_PATH.read_text(encoding="utf-8").splitlines()[-1])
    assert trace["explanation_failed"] is True and "429" in trace["explanation_error"]


def test_api_answer_shows_the_flag_but_not_the_error():
    sql_agent = SQLAgent(con=con(), llm=ScriptedLLM(["SELECT customer_id, SUM(revenue) AS spend FROM sales GROUP BY customer_id",
                                                     Exception(QUOTA_ERROR)]))
    services = Services(FakeRouter("data", "q"), sql_agent, lambda q: None, ScriptedLLM([]))
    shaped = shape_answer(ask(build_graph(services), "q", "t-api"))
    assert shaped["explanation_failed"] is True
    assert_no_error_text(json.dumps(shaped))


def test_failed_query_answer_has_no_error_text():
    llm = ScriptedLLM(["SELECT nope FROM sales", Exception(QUOTA_ERROR)])  # bad SQL, then the repair call fails
    res = SQLAgent(con=con(), llm=llm).run("q")
    assert not res.ok and res.answer == QUERY_FAILED_ANSWER
    assert "429" in res.error  # kept for the trace log
    assert_no_error_text(res.answer)


def test_rows_answer_passes_the_verifier_and_handles_empty_and_long_results():
    rows = [("United Kingdom", 8276953.103998788), ("Netherlands", 276661.86), ("EIRE", 273420.70)]
    text = rows_answer(["country", "revenue"], rows)
    assert verify_answer(text, "Top countries by revenue", sql_rows=rows).removed == []
    assert "no rows" in rows_answer(["x"], [])
    long = rows_answer(["n"], [(i,) for i in range(15)], limit=10)
    assert long.count("\n- ") == 10 and "More rows were returned" in long
