import pytest


@pytest.fixture(autouse=True)
def _isolated_call_log(tmp_path, monkeypatch):
    """Tests must never write to the real logs/llm_calls.jsonl."""
    monkeypatch.setattr("ria.tracing.LLM_CALLS_PATH", tmp_path / "llm_calls.jsonl")
    monkeypatch.setattr("ria.tracing.TRACES_PATH", tmp_path / "traces.jsonl")
    monkeypatch.setattr("ria.tracing.SQL_REPAIRS_PATH", tmp_path / "sql_repairs.jsonl")
    monkeypatch.setattr("ria.config.LLM_CACHE_DIR", tmp_path / "llm_cache")


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr("ria.config.MIN_SECONDS_BETWEEN_CALLS", 0)
