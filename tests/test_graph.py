import duckdb
import pytest
from pydantic import ValidationError

from ria.graph import REFUSAL_MESSAGE, RouterDecision, Services, ask, build_graph
from ria.ingest_docs import Chunk
from ria.rag_agent import Citation, RagAnswer
from ria.retrieval import Hit
from ria.sql_agent import SQLAgent


class _Raw:
    usage_metadata = {"input_tokens": 10, "output_tokens": 5}


class FakeRouterLLM:
    """Mimics llm.with_structured_output(schema, include_raw=True). Replies are scripted and the prompts are recorded."""
    model = "fake-router"

    def __init__(self, decisions):
        self.decisions = list(decisions)
        self.prompts = []

    def with_structured_output(self, schema, include_raw=True):
        outer = self

        class Runner:
            def invoke(self, messages):
                outer.prompts.append(messages[-1][1])
                route, standalone, *parts = outer.decisions.pop(0)
                extra = {"data_question": parts[0], "policy_question": parts[1]} if parts else {}
                return {"raw": _Raw(), "parsed": schema(route=route, standalone_question=standalone, reason="test", **extra),
                        "parsing_error": None}
        return Runner()


class _Reply:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {}


class ScriptedLLM:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages)
        return _Reply(self.replies.pop(0))


def make_sql_agent(replies):
    con = duckdb.connect()
    con.execute("CREATE TABLE returns AS SELECT 'C1' AS invoice_no, 2010 AS invoice_year, 12 AS invoice_month, "
                "5.0 AS revenue, 1 AS quantity, 'A' AS stock_code")
    con.execute("INSERT INTO returns VALUES ('C2', 2010, 12, 7.5, 1, 'B')")
    return SQLAgent(con=con, llm=ScriptedLLM(replies))


CHUNK = Chunk("r:2.1", "Returns and Refunds", "2. Standard return window", "2.1", "Retail return window", 1,
              "2.1 Retail return window. A retail customer may return an unused item within 30 days of the delivery date.")


def make_rag(answer):
    return lambda q: RagAnswer(q, answer, [Hit(CHUNK, 1.0, "hybrid")], [Citation("Returns and Refunds", "2.1", 1)], [])


def services(decisions, sql_replies=(), rag_answer="", combiner_replies=()):
    return Services(FakeRouterLLM(decisions), make_sql_agent(list(sql_replies)), make_rag(rag_answer),
                    ScriptedLLM(list(combiner_replies)))


def test_router_output_schema():
    ok = RouterDecision(route="data", standalone_question="q", reason="r")
    assert ok.route == "data"
    for bad_route in ("sql", "", "DATA"):
        with pytest.raises(ValidationError):
            RouterDecision(route=bad_route, standalone_question="q", reason="r")
    with pytest.raises(ValidationError):
        RouterDecision(route="data")  # standalone_question and reason are required
    assert set(RouterDecision.model_json_schema()["properties"]) == {"route", "standalone_question", "data_question", "policy_question", "reason"}


def test_data_route_end_to_end():
    s = services([("data", "How many orders were returned in December 2010?")],
                 sql_replies=["SELECT COUNT(DISTINCT invoice_no) AS n FROM returns", "2 orders were returned in December 2010."])
    out = ask(build_graph(s), "How many orders were returned in December 2010?", "t1")
    assert out["trajectory"] == ["router", "sql_agent", "combiner", "verifier"]
    assert out["route"] == "data" and out["verified"] is True
    assert out["answer"] == "2 orders were returned in December 2010."


def test_policy_route_end_to_end():
    s = services([("policy", "How long to return an item?")],
                 rag_answer="Retail customers have 30 days [Returns and Refunds §2.1, p.1].")
    out = ask(build_graph(s), "How long to return an item?", "t2")
    assert out["trajectory"] == ["router", "rag_agent", "combiner", "verifier"]
    assert out["verified"] is True and "30 days" in out["answer"]


def test_both_route_runs_both_agents_then_combines():
    s = services([("both", "Returned orders in December 2010 and the return window?")],
                 sql_replies=["SELECT COUNT(DISTINCT invoice_no) AS n FROM returns", "There were 2 returned orders."],
                 rag_answer="Retail customers have 30 days [Returns and Refunds §2.1, p.1].",
                 combiner_replies=["There were 2 returned orders. Retail customers have 30 days [Returns and Refunds §2.1, p.1]."])
    out = ask(build_graph(s), "Returned orders in December 2010 and the return window?", "t3")
    assert out["trajectory"] == ["router", "sql_agent", "rag_agent", "combiner", "verifier"]
    assert out["verified"] is True
    assert "2 returned orders" in out["answer"] and "§2.1" in out["answer"]


def test_out_of_scope_is_refused_without_any_agent():
    s = services([("out_of_scope", "What is the weather in London?")])
    out = ask(build_graph(s), "What is the weather in London?", "t4")
    assert out["trajectory"] == ["router", "refuse"]
    assert out["answer"] == REFUSAL_MESSAGE


def test_verifier_node_removes_fabricated_number_and_fake_citation():
    s = services([("policy", "How long to return an item?")],
                 rag_answer=("Retail customers have 30 days [Returns and Refunds §2.1, p.1]. "
                             "They also get 99 days in December [Returns and Refunds §2.1, p.1]. "
                             "Wholesale gets 14 days [Returns and Refunds §9.9, p.2]."))
    out = ask(build_graph(s), "How long to return an item?", "t5")
    assert out["verified"] is False
    assert out["answer"] == "Retail customers have 30 days [Returns and Refunds §2.1, p.1]."
    reasons = sorted(r["reason"] for r in out["verifier"]["removed"])
    assert reasons == ["invalid_citation", "unsupported_number"]


def test_memory_follow_up_uses_same_thread_and_threads_are_separate(tmp_path):
    s = services([("data", "How many orders were returned in December 2010?"),
                  ("data", "How many orders were returned in January 2011?"),
                  ("data", "A fresh question")],
                 sql_replies=["SELECT COUNT(DISTINCT invoice_no) AS n FROM returns", "2 orders were returned in December 2010.",
                              "SELECT COUNT(DISTINCT invoice_no) AS n FROM returns WHERE invoice_month = 1", "0 orders were returned in January 2011.",
                              "SELECT COUNT(*) AS n FROM returns", "2 rows."])
    app = build_graph(s)
    ask(app, "How many orders were returned in December 2010?", "thread-A")
    second = ask(app, "And in January?", "thread-A")
    assert "How many orders were returned in December 2010?" in s.router_llm.prompts[1]  # history reached the router
    assert "And in January?" in s.router_llm.prompts[1]
    assert second["standalone_question"] == "How many orders were returned in January 2011?"
    assert second["trajectory"] == ["router", "sql_agent", "combiner", "verifier"]  # trajectory is reset per request
    ask(app, "Something new", "thread-B")
    assert "How many orders" not in s.router_llm.prompts[2]  # a different thread sees no history


def test_trace_line_written(tmp_path):
    import json
    from ria import tracing
    s = services([("out_of_scope", "x")])
    ask(build_graph(s), "x", "t9")
    line = json.loads(tracing.TRACES_PATH.read_text(encoding="utf-8").splitlines()[-1])
    assert line["trajectory"] == ["router", "refuse"] and line["thread_id"] == "t9"


def test_both_route_gives_each_agent_only_its_own_part():
    seen = {}

    def rag(q):
        seen["policy"] = q
        return make_rag("Retail customers have 30 days [Returns and Refunds §2.1, p.1].")(q)

    s = Services(FakeRouterLLM([("both", "Orders returned in December 2010 and the retail return window?",
                                 "How many orders were returned in December 2010?",
                                 "How long do retail customers have to return an item?")]),
                 make_sql_agent(["SELECT COUNT(DISTINCT invoice_no) AS n FROM returns", "There were 2 returned orders."]),
                 rag, ScriptedLLM(["There were 2 returned orders. Retail customers have 30 days [Returns and Refunds §2.1, p.1]."]))
    out = ask(build_graph(s), "Orders returned in December 2010 and the retail return window?", "t10")
    assert seen["policy"] == "How long do retail customers have to return an item?"
    sql_prompt = s.sql_agent.llm.prompts[0][-1][1]
    assert "How many orders were returned in December 2010?" in sql_prompt and "retail return window" not in sql_prompt
    assert out["verified"] is True


def test_router_schema_has_split_fields_with_defaults():
    d = RouterDecision(route="policy", standalone_question="q", reason="r")
    assert d.data_question == "" and d.policy_question == ""
