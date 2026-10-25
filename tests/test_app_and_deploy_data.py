import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ria import config, ui_backend
from ria.service import load_examples
from ria.sql_compare import results_match
from ria.sql_db import connect_readonly, ensure_db, get_schema

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")


class FakeBackend:
    def __init__(self, live_available=True, error=None):
        self.live_available, self.error, self.asked = live_available, error, []

    def ask(self, question, thread_id, session_id):
        self.asked.append(question)
        if self.error:
            raise self.error
        return {**load_examples()[0], "source": "live", "answer": "LIVE ANSWER", "thread_id": thread_id}

    def limits(self, session_id):
        return {"session_remaining": 4 if self.live_available else 0, "session_limit": 5, "daily_remaining": 50 if self.live_available else 0,
                "daily_limit": 60, "live_available": self.live_available, "reason": "ok" if self.live_available else "session",
                "message": "" if self.live_available else "You have used your 5 live questions for this session."}

    def examples(self):
        return load_examples()


def run_app(monkeypatch, backend):
    monkeypatch.setattr(ui_backend, "get_backend", lambda: backend)
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def texts(at):
    return " ".join(x.value for x in at.markdown) + " " + " ".join(x.value for x in at.caption)


def test_app_starts_and_offers_limited_live_questions_and_free_examples(monkeypatch):
    at = run_app(monkeypatch, FakeBackend())
    assert not at.exception
    assert any("Pre-computed examples" in s.value for s in at.sidebar.subheader)
    assert any("Live questions" in s.value for s in at.sidebar.subheader)
    assert len(at.sidebar.button) >= 5  # 4 live questions + open example + clear


def test_opening_a_precomputed_example_shows_every_detail_and_uses_no_quota(monkeypatch):
    backend = FakeBackend()
    at = run_app(monkeypatch, backend)
    at.sidebar.selectbox(key="example_route").select("both")
    at.run()
    at.sidebar.button(key="open-example").click()
    at.run()
    assert not at.exception and backend.asked == []  # no live call was made
    body = texts(at)
    assert "Pre-computed example" in body
    expander = at.expander[0]
    assert expander.label == "How this was answered"
    inside = " ".join(x.value for x in expander.markdown) + " " + " ".join(x.value for x in expander.caption)
    assert "Agent trajectory" in inside and "router" in inside and "SQL used" in inside and "Citations" in inside
    assert any("Pre-computed, not live" in i.value for i in expander.info)
    assert any("Verified" in s.value for s in expander.success)
    table = " ".join(x.value for x in expander.markdown)
    for column in ("Tokens in", "Tokens out", "Cached in", "Cost (USD)", "Latency", "Model calls"):
        assert column in table, column
    assert len(expander.code) == 1 and expander.code[0].language == "sql"


def test_live_question_is_answered_and_labelled_live(monkeypatch):
    backend = FakeBackend()
    at = run_app(monkeypatch, backend)
    at.chat_input[0].set_value("How many orders in 2010?").run()
    assert backend.asked == ["How many orders in 2010?"]
    assert "LIVE ANSWER" in texts(at) and "Live answer" in texts(at)


def test_limit_reached_shows_a_clear_warning_and_disables_live_buttons(monkeypatch):
    backend = FakeBackend(live_available=False, error=ui_backend.BackendError("limit", "You have used your 5 live questions for this session."))
    at = run_app(monkeypatch, backend)
    assert any("You have used your 5 live questions" in w.value for w in at.sidebar.warning)
    live_buttons = [b for b in at.sidebar.button if b.key and b.key.startswith("live-")]
    assert live_buttons and all(b.disabled for b in live_buttons)
    at.chat_input[0].set_value("one more").run()
    assert any("5 live questions" in w.value for w in at.warning)
    assert not at.exception


# ------------------------------------------------------------------ deploy data

def test_database_rebuilt_from_parquet_gives_the_stored_gold_answers(tmp_path):
    """The deployed database is rebuilt from committed parquet files. Every gold query must still give its gold answer."""
    db = ensure_db(tmp_path / "rebuilt.duckdb", config.DEPLOY_DIR)
    con = connect_readonly(db)
    assert {"sales", "returns", "products", "customers"} <= set(get_schema(con))
    checked = 0
    for name in ("questions.jsonl", "questions_v2.jsonl"):
        for line in (ROOT / "evals" / name).read_text(encoding="utf-8").splitlines():
            q = json.loads(line)
            if q.get("gold_sql"):
                rows = con.execute(q["gold_sql"]).fetchall()
                assert results_match(q["gold_answer"], rows, ordered=q.get("ordered", False)), q["id"]
                checked += 1
    assert checked == 36  # 15 dev + b01 + 10 held-out v1 + 10 v2 gold queries


def test_ensure_db_explains_what_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="parquet"):
        ensure_db(tmp_path / "x.duckdb", tmp_path / "no-parquet-here")


def test_deploy_parquet_files_are_small_enough_to_commit():
    sizes = [p.stat().st_size for p in config.DEPLOY_DIR.glob("*.parquet")]
    assert sizes and sum(sizes) < 25_000_000
