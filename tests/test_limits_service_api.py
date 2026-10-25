import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

from ria import config
from ria.api import create_app
from ria.limits import UsageLimiter, next_reset_utc, quota_day
from ria.llm import LLMError
from ria.service import AskService, InvalidQuestion, LimitReached, UpstreamError, UpstreamQuotaError, load_examples, validate_question


class Clock:
    def __init__(self, t):
        self.t = t

    def __call__(self):
        return self.t


# 2026-10-25 20:00 UTC is 13:00 on the 25th in US Pacific (PDT)
T0 = datetime(2026, 10, 25, 20, 0, tzinfo=timezone.utc).timestamp()


# ------------------------------------------------------------------ limiter

def test_session_limit_then_message():
    lim = UsageLimiter(session_limit=2, daily_limit=10, clock=Clock(T0))
    assert lim.acquire("a").allowed and lim.acquire("a").allowed
    d = lim.acquire("a")
    assert not d.allowed and d.reason == "session" and "2 live questions" in d.message and "pre-computed examples" in d.message
    assert lim.acquire("b").allowed  # another visitor is not affected


def test_daily_limit_applies_to_everyone_and_gives_reset_time():
    lim = UsageLimiter(session_limit=5, daily_limit=3, clock=Clock(T0))
    for s in ("a", "b", "c"):
        assert lim.acquire(s).allowed
    d = lim.acquire("d")
    assert not d.allowed and d.reason == "daily" and "3 live questions" in d.message and "UTC" in d.message
    assert d.daily_remaining == 0


def test_limits_reset_when_the_us_pacific_day_changes():
    clock = Clock(T0)
    lim = UsageLimiter(1, 1, clock=clock)
    assert lim.acquire("a").allowed and not lim.acquire("a").allowed
    assert next_reset_utc(T0).strftime("%Y-%m-%d %H:%M") == "2026-10-26 07:00"  # midnight PDT = 07:00 UTC
    clock.t = datetime(2026, 10, 26, 7, 1, tzinfo=timezone.utc).timestamp()
    assert quota_day(clock.t) == "2026-10-26"
    assert lim.acquire("a").allowed


def test_pause_until_reset_blocks_everything_and_refund_gives_a_request_back():
    clock = Clock(T0)
    lim = UsageLimiter(5, 10, clock=clock)
    lim.acquire("a")
    lim.refund("a")
    assert lim.status("a").session_remaining == 5 and lim.status("a").daily_remaining == 10
    lim.pause_until_reset()
    d = lim.acquire("b")
    assert not d.allowed and d.reason == "paused" and "quota" in d.message
    clock.t += 24 * 3600
    assert lim.acquire("b").allowed


def test_state_survives_a_restart_but_not_a_new_day(tmp_path):
    path = tmp_path / "usage.json"
    clock = Clock(T0)
    UsageLimiter(5, 10, path=path, clock=clock).acquire("a")
    again = UsageLimiter(5, 10, path=path, clock=clock)
    assert again.status("a").session_remaining == 4 and again.status("a").daily_remaining == 9
    clock.t += 24 * 3600
    assert UsageLimiter(5, 10, path=path, clock=clock).status("a").daily_remaining == 10


def test_unwritable_state_file_does_not_break_the_limiter(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder")
    lim = UsageLimiter(5, 10, path=blocker / "usage.json", clock=Clock(T0))
    assert lim.acquire("a").allowed and lim.status("a").session_remaining == 4


# ------------------------------------------------------------------ service

class FakeGraphApp:
    def __init__(self, state=None, error=None):
        self.state, self.error, self.calls = state, error, 0

    def invoke(self, payload, config=None):
        self.calls += 1
        if self.error:
            raise self.error
        return dict(self.state)


STATE = {"answer": "Retail customers have 30 days [Returns and Refunds §2.1, p.1].", "route": "policy",
         "trajectory": ["router", "rag_agent", "combiner", "verifier"], "standalone_question": "How long?",
         "verified": True, "verifier": {"removed": []}, "sql": None,
         "rag": {"chunks": [{"doc": "Returns and Refunds", "clause_id": "2.1", "page": 1, "text": "2.1 Retail return window. 30 days."}],
                 "citations": []}}


def make_service(app, session_limit=3, daily_limit=10):
    return AskService(limiter=UsageLimiter(session_limit, daily_limit, clock=Clock(T0)), app_factory=lambda: app)


def test_validate_question():
    assert validate_question("  hi  ") == "hi"
    for bad in ("", "   ", "x" * (config.MAX_QUESTION_CHARS + 1)):
        with pytest.raises(InvalidQuestion):
            validate_question(bad)


def test_answer_has_every_field_the_ui_needs():
    res = make_service(FakeGraphApp(STATE)).ask("How long do I have?", session_id="s1")
    for key in ("answer", "route", "trajectory", "sql", "citations", "verified", "tokens", "cost_usd", "latency_s", "model", "thread_id", "limits"):
        assert key in res, key
    assert res["route"] == "policy" and res["verified"] is True and res["source"] == "live"
    assert res["citations"] == [{"doc": "Returns and Refunds", "clause_id": "2.1", "page": 1, "text": "2.1 Retail return window. 30 days."}]
    assert set(res["tokens"]) == {"input", "output", "cached", "total"}
    assert res["limits"]["session_remaining"] == 2


def test_limit_blocks_before_any_model_call():
    app = FakeGraphApp(STATE)
    svc = make_service(app, session_limit=1)
    svc.ask("q1", session_id="s")
    with pytest.raises(LimitReached) as exc:
        svc.ask("q2", session_id="s")
    assert exc.value.decision.reason == "session" and app.calls == 1


def test_upstream_daily_quota_pauses_the_demo_and_other_errors_refund():
    svc = make_service(FakeGraphApp(error=LLMError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel")))
    with pytest.raises(UpstreamQuotaError):
        svc.ask("q", session_id="s")
    with pytest.raises(LimitReached) as exc:
        svc.ask("q", session_id="other")
    assert exc.value.decision.reason == "paused"

    svc2 = make_service(FakeGraphApp(error=LLMError("500 internal")))
    with pytest.raises(UpstreamError):
        svc2.ask("q", session_id="s")
    assert svc2.limit_info("s")["session_remaining"] == 3  # refunded


def test_load_examples_reads_the_committed_file():
    examples = load_examples()
    assert len(examples) == 20 and all(e["source"] == "precomputed" for e in examples)


# ------------------------------------------------------------------ API

@pytest.fixture()
def client():
    return TestClient(create_app(make_service(FakeGraphApp(STATE), session_limit=2)))


def test_health_needs_no_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == config.GEMINI_MODEL and "database_present" in body


def test_ask_returns_the_documented_fields(client):
    r = client.post("/ask", json={"question": "How long do I have to return an item?", "thread_id": "abc"}, headers={"X-Session-Id": "u1"})
    assert r.status_code == 200
    body = r.json()
    assert body["route"] == "policy" and body["trajectory"][0] == "router" and body["citations"][0]["clause_id"] == "2.1"
    assert body["verified"] is True and body["tokens"]["total"] == 0 and "cost_usd" in body and "latency_s" in body


def test_ask_validation_errors_are_422(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 1000}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_limit_reached_is_429_with_a_clear_message(client):
    h = {"X-Session-Id": "u2"}
    assert client.post("/ask", json={"question": "q1"}, headers=h).status_code == 200
    assert client.post("/ask", json={"question": "q2"}, headers=h).status_code == 200
    r = client.post("/ask", json={"question": "q3"}, headers=h)
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert detail["code"] == "limit_reached" and detail["reason"] == "session" and "pre-computed examples" in detail["message"]
    assert client.post("/ask", json={"question": "q3"}, headers={"X-Session-Id": "u3"}).status_code == 200  # other caller fine
    limits = client.get("/limits", headers=h).json()
    assert limits["live_available"] is False and limits["session_remaining"] == 0


def test_model_quota_is_503_and_model_error_is_502():
    quota = TestClient(create_app(make_service(FakeGraphApp(error=LLMError("PerDay quota")))))
    r = quota.post("/ask", json={"question": "q"})
    assert r.status_code == 503 and r.json()["detail"]["code"] == "model_quota_exhausted"
    err = TestClient(create_app(make_service(FakeGraphApp(error=LLMError("503 unavailable")))))
    assert err.post("/ask", json={"question": "q"}).status_code == 502


def test_examples_endpoint(client):
    body = client.get("/examples").json()
    assert len(body["examples"]) == 20


# ------------------------------------------------------------------ the committed examples file

def test_examples_are_labelled_real_and_complete():
    data = json.loads(config.EXAMPLES_PATH.read_text(encoding="utf-8"))
    assert data["source_run"] == "caching-stable_first.jsonl" and "real run" in data["examples"][0]["source_note"].lower()
    for e in data["examples"]:
        assert e["source"] == "precomputed" and e["question"] and e["answer"] and e["trajectory"]
        assert set(e["tokens"]) == {"input", "output", "cached", "total"} and e["tokens"]["total"] > 0
        assert e["cost_usd"] > 0 and e["latency_s"] > 0
        if e["route"] in ("data", "both"):
            assert e["sql"] and e["sql_rows"]
        if e["route"] in ("policy", "both"):
            assert e["citations"] and all(c["text"] for c in e["citations"])
    assert any(e["matches_gold"] is False for e in data["examples"])  # the one wrong answer is kept and labelled
