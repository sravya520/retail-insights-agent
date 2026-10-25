import json
from pathlib import Path

from ria import config
from ria import eval_metrics as m
from ria.graph import ROUTER_PROMPT
from ria.ingest_docs import Chunk
from ria.rag_agent import NOT_COVERED, SYSTEM_PROMPT, answer_policy_question, rewrite_query
from ria.retrieval import Hit, Retriever, rrf_fuse

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    model = "fake"

    def __init__(self, text):
        self.text, self.prompts = text, []

    def invoke(self, messages):
        self.prompts.append(messages)

        class R:
            content = self.text
            usage_metadata = {}
        return R()


CHUNK = Chunk("r:2.1", "Returns and Refunds", "2. Window", "2.1", "Window", 1, "2.1 Retail return window. 30 days.")


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, mode="hybrid", top_k=5, rerank=False):
        self.calls.append(("search", [query]))
        return [Hit(CHUNK, 1.0, "hybrid")]

    def search_multi(self, queries, top_k=5):
        self.calls.append(("search_multi", list(queries)))
        return [Hit(CHUNK, 1.0, "hybrid-multi")]


def test_query_rewrite_is_off_by_default_in_production():
    assert config.QUERY_REWRITE is False  # the dev experiment did not meet the adoption rule


def test_answer_without_rewrite_uses_plain_search_and_makes_one_llm_call():
    retriever, rag = FakeRetriever(), FakeLLM("30 days [Returns and Refunds §2.1, p.1].")
    res = answer_policy_question("How long?", retriever=retriever, llm=rag, rewrite=False)
    assert retriever.calls == [("search", ["How long?"])] and res.search_query == ""
    assert len(rag.prompts) == 1


def test_answer_with_rewrite_searches_with_both_queries():
    retriever, rag, rw = FakeRetriever(), FakeLLM("30 days [Returns and Refunds §2.1, p.1]."), FakeLLM('"return window days"\nextra line')
    res = answer_policy_question("How long do I get?", retriever=retriever, llm=rag, rewrite=True, rewrite_llm=rw)
    assert retriever.calls == [("search_multi", ["How long do I get?", "return window days"])]
    assert res.search_query == "return window days"  # first line, quotes stripped
    assert res.citations and not res.invalid_citations


def test_rewrite_falls_back_to_the_question_when_the_reply_is_empty():
    assert rewrite_query("original question", FakeLLM("   ")) == "original question"


def test_rrf_over_several_queries_rewards_agreement():
    fused = rrf_fuse([["a", "b"], ["b", "c"], ["b", "a"]])
    assert fused[0][0] == "b"


def test_search_multi_on_the_real_index():
    index = config.CHUNKS_PATH
    if not index.exists():
        import pytest
        pytest.skip("run python -m ria.ingest_docs first")
    r = Retriever()
    hits = r.search_multi(["return window change of mind", "how many days to return an item"], top_k=5)
    assert len(hits) == 5 and len({h.chunk.id for h in hits}) == 5
    assert any(h.chunk.clause_id == "2.1" and h.chunk.doc == "Returns and Refunds" for h in hits)


def test_rag_prompt_no_longer_tells_the_model_to_say_for_a_retail_customer():
    assert 'say "for a retail customer"' not in SYSTEM_PROMPT
    assert "Never add a customer type that neither the question nor the passage mentions" in SYSTEM_PROMPT
    assert NOT_COVERED in SYSTEM_PROMPT


def test_router_prompt_has_disambiguation_rules_and_examples():
    assert "Decide by what is being asked" in ROUTER_PROMPT
    assert "choose both only when" in ROUTER_PROMPT
    assert "-> policy" in ROUTER_PROMPT and "-> data" in ROUTER_PROMPT and "-> both" in ROUTER_PROMPT


def test_router_prompt_examples_are_not_copied_from_eval_questions():
    questions = []
    for name in ("questions.jsonl", "questions_v2.jsonl"):
        questions += [json.loads(l)["question"].lower() for l in (ROOT / "evals" / name).read_text(encoding="utf-8").splitlines() if l.strip()]
    example_lines = [ln.split('"')[1].lower() for ln in ROUTER_PROMPT.splitlines() if ln.strip().startswith('"') and "->" in ln]
    assert len(example_lines) == 5
    for ex in example_lines:
        assert ex not in questions


def test_not_covered_metric():
    ok = {"expect_not_covered": True, "route": "policy", "answer": NOT_COVERED}
    cited = {"expect_not_covered": True, "route": "policy", "answer": "Open 9 to 5 [Returns and Refunds §1.1, p.1]."}
    wrong_route = {"expect_not_covered": True, "route": "out_of_scope", "answer": NOT_COVERED}
    normal = {"expect_not_covered": None, "route": "policy", "answer": "x"}
    assert m.not_covered_accuracy([ok, cited, wrong_route, normal]) == (1, 3)


def test_v2_question_file_is_well_formed_and_distinct_from_v1():
    v1 = [json.loads(l) for l in (ROOT / "evals" / "questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    v2 = [json.loads(l) for l in (ROOT / "evals" / "questions_v2.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(v2) == 20 and all(q["split"] == "heldout" for q in v2)
    assert {q["id"] for q in v2}.isdisjoint({q["id"] for q in v1})
    assert {q["question"] for q in v2}.isdisjoint({q["question"] for q in v1})
    routes = {r: sum(q["expected_route"] == r for q in v2) for r in ("data", "policy", "both", "out_of_scope")}
    assert routes == {"data": 6, "policy": 6, "both": 4, "out_of_scope": 4}
    assert sum(bool(q.get("expect_not_covered")) for q in v2) == 2
    for q in v2:
        if q["expected_route"] in ("data", "both"):
            assert q["gold_answer"]
        if q["expected_route"] in ("policy", "both") and not q.get("expect_not_covered"):
            assert q["gold_citations"]
