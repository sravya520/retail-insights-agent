import pytest


@pytest.fixture(autouse=True)
def _isolated_call_log(tmp_path, monkeypatch):
    """Tests must never write to the real logs/llm_calls.jsonl."""
    monkeypatch.setattr("ria.tracing.LLM_CALLS_PATH", tmp_path / "llm_calls.jsonl")


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr("ria.config.MIN_SECONDS_BETWEEN_CALLS", 0)
