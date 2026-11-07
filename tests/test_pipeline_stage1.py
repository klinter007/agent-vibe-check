"""
Hermetic tests for Stage 1 (session extractor + session render).

No network, no real LLM. We use a FakeProvider that returns canned JSON, so we
can verify caching, error paths, and validator integration without spending a
single token.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_vibe_check.parser import Message, Session, ToolCall
from agent_vibe_check.pipeline import session_extractor as se
from agent_vibe_check.pipeline.schemas import SCHEMA_VERSION, SessionFacts
from agent_vibe_check.pipeline.session_render import render_session
from agent_vibe_check.providers import ProviderInfo, LLMProvider


# --- helpers ----------------------------------------------------------------


class FakeInfo:
    name = "fake"
    display_name = "Fake"
    env_var = "FAKE_KEY"
    default_model = "fake-model-1"
    signup_url = "https://example.com"


class FakeProvider:
    """Returns a canned string (or raises) on each `analyze()` call."""
    info = FakeInfo()

    def __init__(self, responses: list[str] | None = None, raises: Exception | None = None):
        self.responses = list(responses or [])
        self.raises = raises
        self.calls: list[tuple[str, str | None]] = []

    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        self.calls.append((prompt, model))
        if self.raises is not None:
            raise self.raises
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


def _msg(role: str, text: str, tool_calls=None) -> Message:
    return Message(
        role=role,
        text_blocks=[text] if text else [],
        tool_calls=tool_calls or [],
    )


def _session(sid="s-1", folder="Users-me-projects-demo", n_messages: int = 8) -> Session:
    msgs = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        if role == "assistant" and i == 3:
            msgs.append(_msg(role, "running a command", tool_calls=[
                ToolCall(name="Shell", input={"cmd": "pytest -q"})
            ]))
        else:
            msgs.append(_msg(role, f"message-{i} content about pytest and fixtures"))
    return Session(
        session_id=sid,
        project_folder=folder,
        transcript_path=None,
        messages=msgs,
        source="jsonl",
        _mtime_ns=1_700_000_000_000_000_000,
        _byte_size=len(msgs) * 50,
    )


def _valid_response(msg_count: int = 8) -> str:
    return json.dumps(
        {
            "topic": "test topic",
            "summary": "session summary sentence",
            "topics_touched": ["pytest"],
            "skills_observed": [
                {
                    "dimension": "steering",
                    "level": "working",
                    "confidence": "med",
                    "evidence": [{"msg_idx": 1, "behavior": "used pytest"}],
                }
            ],
            "decisions": [],
            "problems_solved": [],
            "open_questions": [],
            "user_communication": {"rating": "good", "notes": ""},
            "vibing_moments": [],
            "ai_failures": [],
        }
    )


@pytest.fixture(autouse=True)
def isolated_vibe_home(tmp_path, monkeypatch):
    """Point $VIBE_CHECK_HOME at tmp for the whole test. Atomic."""
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path))
    # paths module caches nothing; env is read at call time.
    yield tmp_path


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    """Tests exercise retry paths; don't actually sleep the backoff."""
    monkeypatch.setattr("agent_vibe_check.pipeline.llm_utils.time.sleep", lambda *_: None)


# --- session_render ---------------------------------------------------------


def test_render_includes_msg_idx_prefixes():
    s = _session(n_messages=4)
    r = render_session(s)
    assert "[0] USER:" in r.text
    assert "[1] AI:" in r.text
    assert r.truncated is False
    assert r.message_count == 4


def test_render_includes_tool_call_summary():
    s = _session(n_messages=6)
    r = render_session(s)
    assert "<<tool:Shell>>" in r.text


def test_render_truncates_over_budget():
    # Build a session with huge messages to force truncation
    big = "X" * 20_000
    msgs = [_msg("user" if i % 2 == 0 else "assistant", big) for i in range(20)]
    s = Session(
        session_id="big", project_folder="projects-x", messages=msgs,
        source="jsonl", _mtime_ns=1, _byte_size=10,
    )
    r = render_session(s, max_chars=50_000)
    assert r.truncated is True
    assert "messages omitted" in r.text
    assert r.msg_idx_range == (0, 19)
    assert len(r.text) <= 52_000  # a little slack for the gap marker


# --- Stage 1 extract_session -----------------------------------------------


def test_extract_session_happy_path_writes_facts_and_index():
    s = _session()
    provider = FakeProvider(responses=[_valid_response()])
    result = se.extract_session(s, provider)
    assert result.error is None
    assert result.cached is False
    assert result.facts is not None
    assert result.facts.topic == "test topic"
    # Canonical metadata was filled from session
    assert result.facts.session_id == "s-1"
    assert result.facts.project_short == "demo"
    assert result.facts.source == "jsonl"
    assert result.facts.schema_version == SCHEMA_VERSION
    # Skill survived the taxonomy filter
    assert result.facts.skills_observed[0].dimension == "steering"
    # Files on disk
    facts_path = se._sessions_dir("demo") / "s-1.json"
    idx_path = se._sessions_dir("demo") / se.INDEX_FILE
    assert facts_path.exists()
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text())
    assert "s-1" in idx["entries"]
    assert idx["entries"]["s-1"]["status"] in ("ok", "ok_with_warnings")


def test_extract_session_skips_bad_dimension():
    s = _session()
    bad = json.loads(_valid_response())
    bad["skills_observed"].append(
        {"dimension": "totally-made-up", "level": "expert",
         "confidence": "high", "evidence": [{"msg_idx": 0, "behavior": "claimed it"}]}
    )
    provider = FakeProvider(responses=[json.dumps(bad)])
    result = se.extract_session(s, provider)
    assert result.facts is not None
    # Only the valid dimension survives
    dims = [x.dimension for x in result.facts.skills_observed]
    assert "totally-made-up" not in dims
    assert "steering" in dims


def test_extract_session_cache_hit_skips_llm():
    s = _session()
    provider = FakeProvider(responses=[_valid_response()])
    # First call hits the LLM
    r1 = se.extract_session(s, provider)
    assert r1.cached is False
    assert len(provider.calls) == 1
    # Second call should be a pure cache read, no LLM
    r2 = se.extract_session(s, provider)
    assert r2.cached is True
    assert r2.facts is not None
    assert r2.facts.session_id == "s-1"
    assert len(provider.calls) == 1, "provider must not be called again on cache hit"


def test_extract_session_cache_miss_on_fingerprint_change():
    s = _session()
    provider = FakeProvider(responses=[_valid_response(), _valid_response()])
    se.extract_session(s, provider)
    assert len(provider.calls) == 1

    s.messages.append(_msg("user", "added a message"))
    s._byte_size += 100  # fingerprint changes
    r2 = se.extract_session(s, provider)
    assert r2.cached is False, "fingerprint change should invalidate cache"
    assert len(provider.calls) == 2


def test_extract_session_force_bypasses_cache():
    s = _session()
    provider = FakeProvider(responses=[_valid_response(), _valid_response()])
    se.extract_session(s, provider)
    r2 = se.extract_session(s, provider, force=True)
    assert r2.cached is False
    assert len(provider.calls) == 2


def test_extract_session_skips_too_short():
    s = _session(n_messages=2)
    provider = FakeProvider(responses=[_valid_response()])
    result = se.extract_session(s, provider)
    assert result.facts is None
    assert result.skipped_reason is not None
    assert "too_short" in result.skipped_reason
    assert len(provider.calls) == 0, "should not call LLM for too-short sessions"


def test_extract_session_bad_json_saves_raw_and_records_failure():
    s = _session()
    provider = FakeProvider(responses=["not json at all, just prose"])
    result = se.extract_session(s, provider)
    assert result.facts is None
    assert result.error is not None
    assert "bad_json" in result.error
    raw = se._sessions_dir("demo") / "s-1.raw.txt"
    assert raw.exists()
    idx = se._load_index("demo")
    assert "s-1" in idx
    assert idx["s-1"].status.startswith("failed:")


def test_extract_session_provider_exception_records_failure():
    s = _session()
    provider = FakeProvider(raises=RuntimeError("429 too many requests"))
    result = se.extract_session(s, provider)
    assert result.facts is None
    assert result.error is not None
    assert "RuntimeError" in result.error
    idx = se._load_index("demo")
    assert idx["s-1"].status.startswith("failed:")


def test_extract_project_sessions_aggregates_stats(monkeypatch):
    sessions = [
        _session(sid="s-a", n_messages=6),
        _session(sid="s-b", n_messages=6),
        _session(sid="s-tiny", n_messages=2),  # skipped
    ]
    provider = FakeProvider(responses=[_valid_response(), _valid_response()])
    stats = se.extract_project_sessions("demo", sessions, provider)
    assert stats.total_sessions == 3
    assert stats.ok == 2
    assert stats.skipped == 1
    assert stats.failed == 0


def test_load_project_session_facts_returns_written_files():
    s = _session()
    provider = FakeProvider(responses=[_valid_response()])
    se.extract_session(s, provider)
    facts_list = se.load_project_session_facts("demo")
    assert len(facts_list) == 1
    assert facts_list[0].session_id == "s-1"


def test_progress_callback_invoked():
    s = _session()
    provider = FakeProvider(responses=[_valid_response()])
    seen: list[tuple[int, int]] = []
    def on_progress(done, total, _result):
        seen.append((done, total))
    se.extract_project_sessions("demo", [s], provider, on_progress=on_progress)
    assert seen == [(1, 1)]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])


# --- v2: vibes / friction / outcome, system prompt, batch -------------------


def _v2_response() -> str:
    return json.dumps({
        "topic": "fix login", "goal": "fix the login bug", "summary": "fixed it",
        "driver": "agent", "task_type": "fix-bug",
        "outcome": {"status": "shipped", "confidence": "high",
                    "evidence": [{"msg_idx": 6, "signal": "user confirms"}]},
        "vibes": [{"signal": "verification", "level": "strong", "confidence": "med",
                   "evidence": [{"msg_idx": 2, "behavior": "asked for test output"}]},
                  {"signal": "python", "level": "strong", "confidence": "med", "evidence": []}],
        "friction": [{"type": "loop", "cause": "missing-context", "msg_idx_range": [2, 5],
                      "cost_messages": 4, "what_happened": "retried", "prevention": "say which venv"},
                     {"type": "made-up", "cause": "nonsense", "cost_messages": "x"}],
        "wins": [{"description": "first try", "msg_idxs": [1]}],
        "rule_candidates": [{"rule": "Run the tests first.", "scope": "general", "msg_idx": 3},
                            {"rule": "Use the uv venv in ./.venv.", "scope": "project"}],
    })


class _SystemCapturingProvider(FakeProvider):
    def __init__(self, responses):
        super().__init__(responses)
        self.systems = []

    def analyze(self, prompt, *, model=None, system=None):
        self.systems.append(system)
        return super().analyze(prompt, model=model)


def test_v2_response_is_mapped_and_sanitized():
    prov = _SystemCapturingProvider([_v2_response()])
    res = se.extract_session(_session(), prov)
    f = res.facts
    assert f.schema_version == SCHEMA_VERSION
    assert f.driver == "agent" and f.task_type == "fix-bug"
    assert f.outcome.status == "shipped"
    assert [s.dimension for s in f.skills_observed] == ["verification"]  # unknown signal dropped
    assert f.friction[0].type == "loop" and f.friction[0].cost_messages == 4
    assert f.friction[1].type == "correction" and f.friction[1].cause == "unknown"  # clamped
    assert f.friction_cost == 4
    assert f.vibing_moments[0].description == "first try"
    assert {r.scope for r in f.rule_candidates} == {"general", "project"}
    # Static instructions go in the cacheable system prompt, transcript in the user prompt.
    assert "Lens 1: Vibes" in prov.systems[0]
    assert "# Transcript" in prov.calls[0][0] and "Lens 1" not in prov.calls[0][0]


def test_old_schema_cache_is_not_reused():
    prov = FakeProvider([_v2_response(), _v2_response()])
    s = _session()
    se.extract_session(s, prov)
    short = "demo"
    p = se._sessions_dir(short) / f"{s.session_id}.json"
    raw = json.loads(p.read_text())
    raw["schema_version"] = 1
    p.write_text(json.dumps(raw))
    res = se.extract_session(s, prov)
    assert not res.cached and len(prov.calls) == 2


class _BatchProvider(FakeProvider):
    supports_batch = True

    def __init__(self, responses):
        super().__init__(responses)
        self.batches = []

    def batch_analyze(self, items, *, model=None, on_status=None, should_stop=None):
        self.batches.append([cid for cid, _, _ in items])
        return {cid: (self.responses.pop(0) if self.responses else RuntimeError("boom"))
                for cid, _, _ in items}


def test_extract_sessions_batch_sends_only_pending_once():
    sessions = {"demo": [_session("a"), _session("b"), _session("short", n_messages=2)],
                "other": [_session("c", folder="Users-me-projects-other")]}
    prov = _BatchProvider([_v2_response(), _v2_response(), _v2_response()])
    stats = se.extract_sessions_batch(sessions, prov)
    assert len(prov.batches) == 1 and len(prov.batches[0]) == 3
    assert stats["demo"].ok == 2 and stats["demo"].skipped == 1
    assert stats["other"].ok == 1
    # Second run: everything cached, no batch submitted.
    stats2 = se.extract_sessions_batch(sessions, prov)
    assert len(prov.batches) == 1  # nothing pending -> no batch call
    assert stats2["demo"].cached == 2


def test_extract_sessions_batch_records_errors():
    prov = _BatchProvider([])  # every request errors
    stats = se.extract_sessions_batch({"demo": [_session("a")]}, prov)
    assert stats["demo"].failed == 1 and stats["demo"].errors


class _ResumableProvider(FakeProvider):
    """Fake Claude-style batch API with a server-side store of results."""

    def __init__(self, crash_on_wait=False):
        super().__init__([])
        self.crash_on_wait = crash_on_wait
        self.server: dict[str, list[str]] = {}
        self.submitted = 0

    def batch_submit(self, items, *, model=None):
        self.submitted += 1
        bid = f"b{len(self.server)}"
        self.server[bid] = [cid for cid, _, _ in items]
        return bid

    def batch_wait(self, batch_id, *, on_status=None, should_stop=None):
        if self.crash_on_wait:
            raise KeyboardInterrupt("terminal closed")
        return "ended"

    def batch_results(self, batch_id, items_by_id=None, model=None):
        return {cid: _v2_response() for cid in self.server[batch_id]}


def test_batch_id_survives_a_crash_and_is_resumed():
    sessions = {"demo": [_session("a"), _session("b")]}
    p1 = _ResumableProvider(crash_on_wait=True)
    with pytest.raises(KeyboardInterrupt):
        se.extract_sessions_batch(sessions, p1)
    assert len(se._load_pending_batches()) == 1

    p2 = _ResumableProvider()
    p2.server = p1.server  # same account: the batch still exists server-side
    stats = se.extract_sessions_batch(sessions, p2)
    assert p2.submitted == 0            # nothing paid twice
    assert stats["demo"].ok == 2
    assert se._load_pending_batches() == []


def test_batch_custom_ids_are_stable_and_valid():
    import re
    a, b = se._batch_custom_id(_session("a")), se._batch_custom_id(_session("a"))
    assert a == b and re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", a)
    assert se._batch_custom_id(_session("b")) != a



def test_gone_batch_is_dropped_and_sessions_resent():
    from agent_vibe_check.providers import BatchGoneError
    sessions = {"demo": [_session("a")]}
    p1 = _ResumableProvider(crash_on_wait=True)
    with pytest.raises(KeyboardInterrupt):
        se.extract_sessions_batch(sessions, p1)

    class _Gone(_ResumableProvider):
        first = True

        def batch_wait(self, batch_id, *, on_status=None, should_stop=None):
            if self.first:              # the earlier batch no longer exists server-side
                self.first = False
                raise BatchGoneError("expired")
            return "ended"

    p2 = _Gone()
    stats = se.extract_sessions_batch(sessions, p2)
    assert p2.submitted == 1 and stats["demo"].ok == 1
    assert se._load_pending_batches() == []


def test_unreadable_batch_is_kept_and_not_paid_twice():
    sessions = {"demo": [_session("a")]}
    p1 = _ResumableProvider(crash_on_wait=True)
    with pytest.raises(KeyboardInterrupt):
        se.extract_sessions_batch(sessions, p1)

    class _Down(_ResumableProvider):
        def batch_wait(self, batch_id, *, on_status=None, should_stop=None):
            raise ConnectionError("network down")

    p2 = _Down()
    stats = se.extract_sessions_batch(sessions, p2)
    assert p2.submitted == 0                      # held, not resubmitted
    assert len(se._load_pending_batches()) == 1   # kept for the next run
    assert stats["demo"].failed == 1
