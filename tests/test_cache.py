import json

import pytest

from ria import config, tracing
from ria.llm import cache_key, invoke_structured, invoke_with_retry
from ria.graph import RouterDecision


class _Reply:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"input_tokens": 7, "output_tokens": 3}


class CountingLLM:
    model = "fake-model"
    temperature = 0.0
    reasoning_effort = "low"

    def __init__(self):
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _Reply(f"answer {self.calls}")


def log_lines():
    return [json.loads(l) for l in tracing.LLM_CALLS_PATH.read_text(encoding="utf-8").splitlines()]


def test_second_identical_call_is_a_cache_hit_and_is_logged_separately(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", True)
    llm = CountingLLM()
    first = invoke_with_retry(llm, [("system", "s"), ("human", "q")], node="n")
    second = invoke_with_retry(llm, [("system", "s"), ("human", "q")], node="n")
    assert llm.calls == 1 and first.content == second.content == "answer 1"
    lines = log_lines()
    assert [l["cache_hit"] for l in lines] == [False, True]
    assert lines[1]["attempts"] == 0 and lines[1]["latency_s"] < 0.5


def test_cache_off_flag_always_calls_the_model(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)
    llm = CountingLLM()
    invoke_with_retry(llm, [("human", "q")], node="n")
    invoke_with_retry(llm, [("human", "q")], node="n")
    assert llm.calls == 2
    assert all("cache_hit" not in l for l in log_lines())  # no cache field when the cache is off
    assert not config.LLM_CACHE_DIR.exists()


def test_key_changes_with_model_prompt_and_settings():
    llm = CountingLLM()
    base = cache_key(llm, [("human", "q")])
    assert cache_key(llm, [("human", "q2")]) != base
    other_model = CountingLLM()
    other_model.model = "other"
    assert cache_key(other_model, [("human", "q")]) != base
    other_think = CountingLLM()
    other_think.reasoning_effort = "high"
    assert cache_key(other_think, [("human", "q")]) != base
    assert cache_key(llm, [("human", "q")], schema_name="X") != base
    assert cache_key(llm, [("human", "q")]) == base  # stable


def test_structured_call_is_cached(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", True)

    class Raw:
        usage_metadata = {"input_tokens": 1, "output_tokens": 1}

    class StructLLM(CountingLLM):
        def with_structured_output(self, schema, include_raw=True):
            outer = self

            class R:
                def invoke(self, messages):
                    outer.calls += 1
                    return {"raw": Raw(), "parsed": schema(route="policy", standalone_question="q", reason="r"),
                            "parsing_error": None}
            return R()

    llm = StructLLM()
    a = invoke_structured(llm, [("human", "q")], RouterDecision, node="router")
    b = invoke_structured(llm, [("human", "q")], RouterDecision, node="router")
    assert llm.calls == 1 and a == b and b.route == "policy"
    assert [l["cache_hit"] for l in log_lines()] == [False, True]


def test_unparsable_structured_output_raises_clear_error(monkeypatch):
    from ria.llm import LLMError
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)

    class BadLLM(CountingLLM):
        def with_structured_output(self, schema, include_raw=True):
            class R:
                def invoke(self, messages):
                    return {"raw": None, "parsed": None, "parsing_error": "bad json"}
            return R()

    with pytest.raises(LLMError, match="did not match RouterDecision"):
        invoke_structured(BadLLM(), [("human", "q")], RouterDecision, node="router")


def test_list_content_is_returned_as_text(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)

    class PartsLLM(CountingLLM):
        def invoke(self, messages):
            return _Reply([{"type": "text", "text": "Hello "}, {"type": "text", "text": "world"}])

    assert invoke_with_retry(PartsLLM(), [("human", "q")], node="n").content == "Hello world"
