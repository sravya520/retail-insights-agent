import os
from pathlib import Path

# Tests must not wait on the network. If the embedding model is already in the local Hugging Face cache, use it offline.
# (Set before any Hugging Face import. On a machine without the cache, for example a fresh CI runner, this is skipped.)
_HF_CACHE = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub" / "models--BAAI--bge-small-en-v1.5"
if _HF_CACHE.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

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
