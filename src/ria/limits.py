"""Request limits that protect the free Gemini quota of a public demo.

Three layers:
  * per session: each visitor session may ask SESSION_LIMIT live questions per quota day
  * per day: the whole demo may answer DAILY_LIMIT live questions per quota day
  * paused: if Gemini itself reports that its daily quota is used up, the demo stops asking until the day rolls over

The quota day is the US Pacific calendar day, because that is when the Gemini free-tier daily quotas reset.
State is kept in a small JSON file (or in memory when no path is given), so limits survive restarts of the app
process but not a wipe of the disk. Pre-computed examples never touch this module.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

QUOTA_TZ = ZoneInfo("America/Los_Angeles")


@dataclass
class LimitDecision:
    allowed: bool
    reason: str            # ok, session, daily or paused
    message: str
    session_remaining: int
    daily_remaining: int
    resets_at_utc: str     # ISO time of the next quota reset


def quota_day(now: float) -> str:
    return datetime.fromtimestamp(now, QUOTA_TZ).strftime("%Y-%m-%d")


def next_reset_utc(now: float) -> datetime:
    local = datetime.fromtimestamp(now, QUOTA_TZ)
    midnight = (local + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return midnight.astimezone(ZoneInfo("UTC"))


class UsageLimiter:
    def __init__(self, session_limit: int, daily_limit: int, path: Path | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.session_limit, self.daily_limit, self.path, self.clock = session_limit, daily_limit, path, clock
        self._lock = threading.Lock()
        self._state = self._load()

    # ---- state
    def _fresh(self, day: str) -> dict:
        return {"day": day, "daily_count": 0, "sessions": {}, "paused": False}

    def _load(self) -> dict:
        day = quota_day(self.clock())
        if self.path and self.path.exists():
            try:
                state = json.loads(self.path.read_text(encoding="utf-8"))
                if state.get("day") == day:
                    return state
            except (OSError, ValueError):
                pass
        return self._fresh(day)

    def _save(self) -> None:
        if not self.path:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._state), encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError:
            pass  # limits still work in memory if the disk is read-only

    def _roll(self) -> None:
        day = quota_day(self.clock())
        if self._state.get("day") != day:
            self._state = self._fresh(day)

    # ---- decisions
    def _decision(self, session_id: str) -> LimitDecision:
        used = self._state["sessions"].get(session_id, 0)
        s_left = max(self.session_limit - used, 0)
        d_left = max(self.daily_limit - self._state["daily_count"], 0)
        reset = next_reset_utc(self.clock()).strftime("%Y-%m-%d %H:%M UTC")
        if self._state["paused"]:
            return LimitDecision(False, "paused", "The language model's free daily quota is used up, so live questions are paused. "
                                 f"They resume after the quota resets ({reset}). The pre-computed examples still work.", s_left, 0, reset)
        if d_left == 0:
            return LimitDecision(False, "daily", f"The demo has reached its limit of {self.daily_limit} live questions for today. "
                                 f"It resets at {reset}. The pre-computed examples still work and cost nothing.", s_left, 0, reset)
        if s_left == 0:
            return LimitDecision(False, "session", f"You have used your {self.session_limit} live questions for this session. "
                                 "This keeps the free demo available for other visitors. The pre-computed examples still work, "
                                 "or you can run the project yourself with your own Gemini key (see the README).", 0, d_left, reset)
        return LimitDecision(True, "ok", "ok", s_left, d_left, reset)

    def status(self, session_id: str) -> LimitDecision:
        with self._lock:
            self._roll()
            return self._decision(session_id)

    def acquire(self, session_id: str) -> LimitDecision:
        """Check the limits and, if allowed, count one request. Check and count happen under one lock."""
        with self._lock:
            self._roll()
            decision = self._decision(session_id)
            if decision.allowed:
                self._state["daily_count"] += 1
                self._state["sessions"][session_id] = self._state["sessions"].get(session_id, 0) + 1
                self._save()
                decision = self._decision(session_id)
                decision.allowed, decision.reason, decision.message = True, "ok", "ok"
            return decision

    def refund(self, session_id: str) -> None:
        """Give a request back (for example when the call failed before the model answered)."""
        with self._lock:
            self._roll()
            if self._state["daily_count"] > 0:
                self._state["daily_count"] -= 1
            if self._state["sessions"].get(session_id, 0) > 0:
                self._state["sessions"][session_id] -= 1
            self._save()

    def pause_until_reset(self) -> None:
        """Gemini reported its daily quota is used up: stop live questions until the quota day rolls over."""
        with self._lock:
            self._roll()
            self._state["paused"] = True
            self._save()


def limiter_from_config(path: Path | None = None) -> UsageLimiter:
    from ria import config
    return UsageLimiter(config.DEMO_SESSION_LIMIT, config.DEMO_DAILY_LIMIT, path or config.USAGE_STATE_PATH)
