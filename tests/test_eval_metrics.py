import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from ria import eval_metrics as m
from ria.graph import REFUSAL_MESSAGE
from ria.judge import JudgePool, JudgeScores, evidence_text, is_unavailable
from ria.llm import LLMError

DATA_TRAJ = ["router", "sql_agent", "combiner", "verifier"]
POLICY_TRAJ = ["router", "rag_agent", "combiner", "verifier"]
CHUNK = {"doc": "Returns and Refunds", "clause_id": "2.1", "page": 1,
         "text": "2.1 Retail return window. A retail customer may return an unused item within 30 days."}
SCHEMA = {"sales": ["revenue", "invoice_year", "country", "quantity"], "returns": ["revenue", "invoice_no"]}


def rec(id, expected, route, traj, **kw):
    base = dict(id=id, split="heldout", question="q", expected_route=expected,
                expected_trajectory={"data": DATA_TRAJ, "policy": POLICY_TRAJ, "out_of_scope": ["router", "refuse"]}[expected],
                route=route, trajectory=traj, answer="a", error=None)
    base.update(kw)
    return base


def test_router_and_trajectory_accuracy_and_wrong_paths():
    rs = [rec("a", "data", "data", DATA_TRAJ), rec("b", "policy", "data", DATA_TRAJ),
          rec("c", "policy", "policy", ["router", "rag_agent", "verifier"]), rec("d", "data", "data", DATA_TRAJ)]
    assert m.router_accuracy(rs) == (3, 4)
    ok, n, wrong = m.trajectory_accuracy(rs)
    assert (ok, n) == (2, 4) and sum(wrong.values()) == 2
    assert any("got router > rag_agent > verifier" in k for k in wrong)


def test_sql_execution_accuracy_uses_tolerance_and_order():
    good = rec("a", "data", "data", DATA_TRAJ, gold_answer=[["UK", 100.0]], ordered=False,
               sql={"rows": [["UK", 100.4]], "error": None})
    off = rec("b", "data", "data", DATA_TRAJ, gold_answer=[["UK", 100.0]], ordered=False,
              sql={"rows": [["UK", 103.0]], "error": None})
    errored = rec("c", "data", "data", DATA_TRAJ, gold_answer=[[1]], sql={"rows": [], "error": "boom"})
    no_gold = rec("d", "policy", "policy", POLICY_TRAJ)
    assert m.sql_execution_accuracy([good, off, errored, no_gold]) == (1, 3)


def test_retrieval_recall_any_gold_in_top5():
    hit = rec("a", "policy", "policy", POLICY_TRAJ, gold_citations=[{"doc": "Returns and Refunds", "clause_id": "2.1"}], rag={"chunks": [CHUNK]})
    miss = rec("b", "policy", "policy", POLICY_TRAJ, gold_citations=[{"doc": "Shipping and Delivery", "clause_id": "3.1"}], rag={"chunks": [CHUNK]})
    assert m.retrieval_recall_at_5([hit, miss]) == (1, 2)


def test_gate_out_of_scope_needs_exact_refusal_path():
    ok = rec("a", "out_of_scope", "out_of_scope", ["router", "refuse"], answer=REFUSAL_MESSAGE)
    wrong_route = rec("b", "out_of_scope", "data", DATA_TRAJ)
    wrong_text = rec("c", "out_of_scope", "out_of_scope", ["router", "refuse"], answer="Sure, here you go")
    bad, n, ids = m.gate_out_of_scope_refused([ok, wrong_route, wrong_text])
    assert (bad, n, ids) == (2, 3, ["b", "c"])
    assert m.refusal_rate([ok, wrong_route]) == (1, 2)


def test_gate_no_unsupported_content_flags_fabricated_number_and_fake_citation():
    clean = rec("a", "data", "data", DATA_TRAJ, answer="Revenue was 100.50.", sql={"rows": [[100.5]], "error": None},
                standalone_question="q")
    fabricated = rec("b", "data", "data", DATA_TRAJ, answer="Revenue was 999.", sql={"rows": [[100.5]], "error": None},
                     standalone_question="q")
    fake_cite = rec("c", "policy", "policy", POLICY_TRAJ, answer="30 days [Returns and Refunds §9.9, p.1].",
                    rag={"chunks": [CHUNK]}, standalone_question="q")
    good_cite = rec("d", "policy", "policy", POLICY_TRAJ, answer="30 days [Returns and Refunds §2.1, p.1].",
                    rag={"chunks": [CHUNK]}, standalone_question="q")
    refused = rec("e", "out_of_scope", "out_of_scope", ["router", "refuse"], answer=REFUSAL_MESSAGE)
    bad, checked, ids = m.gate_no_unsupported_content([clean, fabricated, fake_cite, good_cite, refused])
    assert (bad, checked, ids) == (2, 4, ["b", "c"])


def test_unsafe_sql_suite_is_fully_blocked_and_gold_is_not():
    unsafe = json.loads((Path(__file__).resolve().parents[1] / "evals" / "unsafe_sql.json").read_text(encoding="utf-8"))
    assert len(unsafe) >= 40
    leaked, n, which = m.gate_unsafe_sql_blocked(unsafe, SCHEMA)
    assert leaked == 0 and n == len(unsafe), which
    assert m.gate_unsafe_sql_blocked(["SELECT SUM(revenue) FROM sales"], SCHEMA)[0] == 1  # a safe query is not "blocked"
    assert m.gold_queries_not_blocked(["SELECT SUM(revenue) FROM sales WHERE invoice_year = 2010"], SCHEMA) == []
    assert len(m.gold_queries_not_blocked(["DROP TABLE sales"], SCHEMA)) == 1


def test_judge_means_and_verifier_stats():
    rs = [rec("a", "data", "data", DATA_TRAJ, judge={"faithfulness": 5, "relevance": 4}, verified=True, removed=[]),
          rec("b", "data", "data", DATA_TRAJ, judge={"faithfulness": 3, "relevance": 5}, verified=False,
              removed=[{"reason": "unsupported_number", "sentence": "x", "detail": "1"}])]
    jm = m.judge_means(rs)
    assert jm["faithfulness"] == 4 and jm["relevance"] == 4.5 and jm["faithful_share"] == 0.5
    vs = m.verifier_stats(rs)
    assert vs["answers_changed"] == 1 and vs["removed_unsupported_number"] == 1


def test_select_for_labeling_prefers_verifier_changed_and_mixes_routes():
    records = []
    for i in range(30):
        route = ["data", "policy", "both"][i % 3]
        records.append(rec(f"x{i}", "data", route, DATA_TRAJ, judge={"faithfulness": 5, "relevance": 5, "model": "m"},
                           verified=(i % 7 != 0), split="heldout" if i < 12 else "dev"))
    chosen = m.select_for_labeling(records, n=20)
    assert len(chosen) == 20 and len({r["id"] for r in chosen}) == 20
    changed_all = [r["id"] for r in records if r["verified"] is False]
    assert all(i in {r["id"] for r in chosen} for i in changed_all)  # all changed answers (5 of them) are included
    assert {r["route"] for r in chosen} == {"data", "policy", "both"}
    assert m.select_for_labeling(records, n=20) == chosen  # deterministic


def test_label_template_hides_judge_scores_and_has_empty_faithful_field():
    r = rec("a", "policy", "policy", POLICY_TRAJ, answer="30 days [Returns and Refunds §2.1, p.1].",
            judge={"faithfulness": 5, "relevance": 5, "model": "m"}, rag={"chunks": [CHUNK]}, sql=None)
    row = m.label_template_row(r, "run-1")
    assert row["faithful"] is None and row["evidence_chunks"][0]["clause_id"] == "2.1"
    assert "judge" not in json.dumps(row).lower()


def test_judge_agreement_math():
    labels = [{"id": "a", "faithful": True}, {"id": "b", "faithful": True}, {"id": "c", "faithful": False},
              {"id": "d", "faithful": False}]
    judge = {"a": True, "b": False, "c": True, "d": False}
    res = m.judge_agreement(labels, judge)
    assert res["n"] == 4 and res["agreement"] == 0.5
    assert res["judge_faithful_human_not"] == 1 and res["judge_not_human_faithful"] == 1
    assert res["disagreements"] == ["b", "c"] and res["kappa"] == 0.0
    perfect = m.judge_agreement(labels, {"a": True, "b": True, "c": False, "d": False})
    assert perfect["agreement"] == 1.0 and perfect["kappa"] == 1.0


def test_judge_scores_schema_bounds():
    assert JudgeScores(faithfulness=5, relevance=1).unsupported_claims == ""
    for bad in (0, 6):
        with pytest.raises(ValidationError):
            JudgeScores(faithfulness=bad, relevance=3)


def test_evidence_text_contains_rows_and_chunk_ids():
    text = evidence_text({"columns": ["country", "revenue"], "rows": [["UK", 1.5]]}, [CHUNK])
    assert "country | revenue" in text and "UK | 1.5" in text and "[Returns and Refunds §2.1, p.1]" in text


def test_judge_pool_falls_back_when_a_model_has_no_daily_quota(monkeypatch):
    calls = []

    def fake_invoke(llm, messages, schema, node="judge"):
        calls.append(llm.model)
        if llm.model == "model-a":
            raise LLMError("Gemini call failed: 429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")
        return schema(faithfulness=4, relevance=5, unsupported_claims="")

    class FakeLLM:
        def __init__(self, model):
            self.model = model

    monkeypatch.setattr("ria.judge.invoke_structured", fake_invoke)
    pool = JudgePool(["model-a", "model-b"], factory=lambda role, model=None, temperature=None: FakeLLM(model))
    first = pool.score("q", "a", "e")
    second = pool.score("q", "a", "e")
    assert first.model == second.model == "model-b" and first.faithful
    assert calls == ["model-a", "model-b", "model-b"]  # model-a is dropped after its first failure
    assert pool.dead == {"model-a"}


def test_judge_pool_raises_when_every_model_is_out_of_quota(monkeypatch):
    def fake_invoke(llm, messages, schema, node="judge"):
        raise LLMError("429 per day quota")

    class FakeLLM:
        model = "x"

    monkeypatch.setattr("ria.judge.invoke_structured", fake_invoke)
    pool = JudgePool(["only"], factory=lambda role, model=None, temperature=None: FakeLLM())
    with pytest.raises(LLMError, match="No judge model is available"):
        pool.score("q", "a", "e")
    assert is_unavailable(Exception("NOT_FOUND no longer available"))
    assert not is_unavailable(Exception("some parse error"))


def test_wilson_lower_bound_for_perfect_scores_on_small_sets():
    assert m.wilson_lower(20, 20) == pytest.approx(0.839, abs=0.002)
    assert m.wilson_lower(10, 10) == pytest.approx(0.722, abs=0.002)
    assert m.wilson_lower(0, 0) is None
    assert m.wilson_lower(5, 10) < 0.5


def test_select_for_labeling_includes_borderline_judge_scores():
    records = []
    for i in range(30):
        records.append(rec(f"y{i}", "data", ["data", "policy", "both"][i % 3], DATA_TRAJ,
                           judge={"faithfulness": 4 if i in (3, 8, 13) else 5, "relevance": 5, "model": "m"},
                           verified=True, split="dev"))
    chosen = {r["id"] for r in m.select_for_labeling(records, n=20)}
    assert {"y3", "y8", "y13"} <= chosen
