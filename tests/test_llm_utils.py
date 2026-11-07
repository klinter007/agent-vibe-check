"""Tests for adaptive retry logic in llm_utils."""

from __future__ import annotations

import pytest

from agent_vibe_check.pipeline.llm_utils import (
    _classify_error,
    call_with_retries,
)


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Make all call_with_retries sleeps instant so tests stay snappy."""
    monkeypatch.setattr(
        "agent_vibe_check.pipeline.llm_utils.time.sleep", lambda *_: None
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class FakeHeaders(dict):
    """Works like requests.Response.headers."""


class FakeResponse:
    def __init__(self, headers: dict):
        self.headers = FakeHeaders(headers)


class RateLimitError(Exception):
    """Mimics anthropic.RateLimitError."""

    def __init__(self, msg: str, headers: dict | None = None):
        super().__init__(msg)
        if headers is not None:
            self.response = FakeResponse(headers)


class AuthenticationError(Exception):
    pass


class InternalServerError(Exception):
    pass


def test_classify_detects_rate_limit_by_class_name():
    cat, wait = _classify_error(RateLimitError("quota exceeded"))
    assert cat == "rate_limit"
    assert wait is None


def test_classify_reads_retry_after_header():
    err = RateLimitError("rate limited", headers={"retry-after": "42"})
    cat, wait = _classify_error(err)
    assert cat == "rate_limit"
    assert wait == 42.0


def test_classify_fatal_for_auth():
    cat, wait = _classify_error(AuthenticationError("401 unauthorized"))
    assert cat == "fatal"


def test_classify_fatal_for_bad_status_in_message():
    cat, _ = _classify_error(RuntimeError("HTTP 400 bad prompt"))
    assert cat == "fatal"


def test_classify_retryable_for_5xx():
    cat, _ = _classify_error(InternalServerError("500 server error"))
    assert cat == "retryable"


def test_classify_retryable_for_unknown():
    cat, _ = _classify_error(RuntimeError("weird network thing"))
    assert cat == "retryable"


# ---------------------------------------------------------------------------
# Retry loop
# ---------------------------------------------------------------------------


def test_retries_transient_then_succeeds():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise InternalServerError("500")
        return "ok"

    result, err = call_with_retries(flaky, max_attempts=5)
    assert result == "ok"
    assert err is None
    assert calls["n"] == 3


def test_rate_limit_honors_retry_after(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "agent_vibe_check.pipeline.llm_utils.time.sleep",
        lambda s: sleeps.append(s),
    )
    calls = {"n": 0}

    def limited():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("slow down", headers={"retry-after": "90"})
        return "ok"

    result, err = call_with_retries(limited, max_attempts=5)
    assert result == "ok"
    # First sleep should respect the 90s retry-after (or >= 60 floor).
    assert sleeps[0] >= 60.0
    assert sleeps[0] == 90.0


def test_rate_limit_without_header_uses_floor(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "agent_vibe_check.pipeline.llm_utils.time.sleep",
        lambda s: sleeps.append(s),
    )
    calls = {"n": 0}

    def limited():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RateLimitError("slow down")  # no header
        return "ok"

    result, _ = call_with_retries(limited, max_attempts=3)
    assert result == "ok"
    assert sleeps[0] == 60.0  # floor


def test_fatal_errors_fail_immediately():
    calls = {"n": 0}

    def auth_failed():
        calls["n"] += 1
        raise AuthenticationError("401 unauthorized")

    result, err = call_with_retries(auth_failed, max_attempts=5)
    assert result is None
    assert isinstance(err, AuthenticationError)
    # Fatal errors should NOT retry.
    assert calls["n"] == 1


def test_max_delay_caps_backoff(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(
        "agent_vibe_check.pipeline.llm_utils.time.sleep",
        lambda s: sleeps.append(s),
    )

    def always_fails():
        raise InternalServerError("500")

    call_with_retries(always_fails, max_attempts=8,
                      base_delay=3.0, max_delay=30.0)
    # 3, 6, 12, 24, 30, 30, 30 — never exceeds max_delay
    assert max(sleeps) <= 30.0
    assert sleeps[-1] == 30.0


def test_on_retry_callback_fires():
    events: list[tuple[str, float, int]] = []

    def hook(category, wait, attempt, err):
        events.append((category, wait, attempt))

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise InternalServerError("boom")
        return "done"

    call_with_retries(flaky, max_attempts=5, on_retry=hook)
    assert len(events) == 2
    assert events[0][0] == "retryable"


def test_on_retry_exception_doesnt_crash_retry():
    """A broken UI hook shouldn't kill the pipeline."""

    def boom_hook(*_args, **_kw):
        raise RuntimeError("ui bug")

    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise InternalServerError("boom")
        return "still works"

    result, err = call_with_retries(flaky, max_attempts=3, on_retry=boom_hook)
    assert result == "still works"
    assert err is None
