"""
Orchestrator tests. Keep them hermetic: mock session ingestion + provider.
Goal: confirm the stages wire together and caching/resume works.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from agent_vibe_check.parser import Message, Session
from agent_vibe_check.pipeline import orchestrator as orch
from agent_vibe_check.pipeline.schemas import (
    CrossProject,
    ProjectSynthesis,
    SessionFacts,
    Timeline,
)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path))
    monkeypatch.setattr("agent_vibe_check.pipeline.llm_utils.time.sleep", lambda *_: None)
    yield tmp_path


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class ProviderInfo:
    name = "fake"
    default_model = "fake-1"
    display_name = "Fake"


class FakeProvider:
    """Thread-safe because call counting happens under a lock."""
    info = ProviderInfo()

    def __init__(self, responses: dict[str, str], *, delay: float = 0.0):
        self.responses = responses
        self.calls: list[str] = []
        self.delay = delay
        self._lock = threading.Lock()

    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        if self.delay:
            import time as _t
            _t.sleep(self.delay)
        for key, resp in self.responses.items():
            if key in prompt:
                with self._lock:
                    self.calls.append(key)
                return resp
        with self._lock:
            self.calls.append("default")
        return self.responses.get("_default", "{}")


def _mk_message(role: str, text: str) -> Message:
    return Message(role=role, text_blocks=[text])


def _mk_session(sid: str, project_folder: str, n_messages: int = 6) -> Session:
    msgs = []
    for i in range(n_messages):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append(_mk_message(role, f"msg {i} about stuff"))
    return Session(
        session_id=sid, project_folder=project_folder, source="jsonl",
        transcript_path=Path(f"/fake/{sid}.jsonl"), messages=msgs,
        _mtime_ns=1, _byte_size=1000,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _stage1_json() -> str:
    return json.dumps({
        "topic": "debug tests",
        "summary": "fixed a race",
        "topics_touched": ["pytest"],
        "skills": {
            "specification": {"level": "strong", "evidence": "msg_idx=1", "confidence": "high"},
        },
        "decisions": [],
        "problems": [],
        "patterns": [],
        "gotchas": [],
        "tools": [],
        "user_communication": {"rating": "good"},
    })


def _stage2_json() -> str:
    return json.dumps({
        "dna": {"what_it_is": "test project", "why_it_exists": "", "key_decisions_narrative": ""},
        "arc": {"phases": [], "pivots": []},
        "recurring_patterns": [],
        "open_threads": [],
        "ai_collaboration": {
            "delegation_style": "", "communication_tone": "",
            "strengths": [], "frustrations": [], "advice_to_user": ""
        },
        "recommended_rules": [],
    })


def _stage3_json() -> str:
    return json.dumps({
        "strongest_growth": ["specification"],
        "surprising_moves": [],
        "emerging_interests": [],
        "plateaus": [],
    })


def _stage4_json() -> str:
    return json.dumps({
        "anomalies": [
            {"observation": "all strong", "hypothesis": "only one project"}
        ]
    })


def _fake_provider() -> FakeProvider:
    return FakeProvider({
        "Stage 1": _stage1_json(),
        "Stage 2": _stage2_json(),
        "Stage 3": _stage3_json(),
        "Stage 4": _stage4_json(),
    })


def test_run_one_project_happy_path(monkeypatch):
    # Folder name must resolve to the same short name ("alpha") that run_one_project
    # writes under, so Stage 2 can find Stage 1's output.
    sessions = [_mk_session("s-1", "alpha"), _mk_session("s-2", "alpha")]
    result = orch.run_one_project("alpha", sessions, _fake_provider())
    assert result.error is None
    assert result.synthesis is not None
    assert result.synthesis.session_count == 2
    assert result.project_folder == "alpha"
    # Stage 1 cache file
    from agent_vibe_check.pipeline.session_extractor import INDEX_FILE
    from agent_vibe_check.paths import get_extractions_dir
    idx = get_extractions_dir() / "alpha" / "sessions" / INDEX_FILE
    assert idx.exists()
    # Stage 2 output
    p = get_extractions_dir() / "alpha" / "project.json"
    assert p.exists()


def test_run_full_pipeline_single_project(monkeypatch):
    # Monkeypatch portfolio selection + session bucketing
    monkeypatch.setattr(orch, "select_portfolio", lambda **kw: ["alpha"])
    sessions = [_mk_session("s-1", "alpha", n_messages=8),
                _mk_session("s-2", "alpha", n_messages=8)]
    monkeypatch.setattr(orch, "_bucket_sessions_by_short", lambda: {"alpha": sessions})

    events: list[tuple[str, dict]] = []

    def prog(ev, data):
        events.append((ev, data))

    result = orch.run_full_pipeline(_fake_provider(), progress=prog)
    assert len(result.all_syntheses) == 1
    assert result.all_syntheses[0].project_short == "alpha"
    # Stage 3 + 4 ran
    event_names = {e for e, _ in events}
    assert "stage3_done" in event_names
    assert "stage4_done" in event_names


def test_run_full_pipeline_is_idempotent(monkeypatch):
    """Second call with unchanged inputs should be fully cached."""
    monkeypatch.setattr(orch, "select_portfolio", lambda **kw: ["alpha"])
    sessions = [_mk_session("s-1", "alpha", n_messages=8)]
    monkeypatch.setattr(orch, "_bucket_sessions_by_short", lambda: {"alpha": sessions})

    first = orch.run_full_pipeline(_fake_provider())
    assert first.all_syntheses, "first run should produce output"

    # Second run: every stage should report cached=True
    second_provider = _fake_provider()
    second = orch.run_full_pipeline(second_provider)
    assert second.timeline_cached is True
    assert second.cross_cached is True
    # Provider shouldn't have been called at all (everything cached).
    assert second_provider.calls == []


def test_run_full_pipeline_empty_portfolio(monkeypatch):
    monkeypatch.setattr(orch, "select_portfolio", lambda **kw: [])
    result = orch.run_full_pipeline(_fake_provider())
    assert result.errors == ["no projects qualify for deep analysis"]


def test_write_all_reports_happy_path(monkeypatch):
    # Run pipeline, then render everything.
    monkeypatch.setattr(orch, "select_portfolio", lambda **kw: ["alpha"])
    sessions = [_mk_session("s-1", "alpha", n_messages=8)]
    monkeypatch.setattr(orch, "_bucket_sessions_by_short", lambda: {"alpha": sessions})

    result = orch.run_full_pipeline(_fake_provider())
    outputs = orch.write_all_reports(result.all_syntheses, result.all_session_facts)
    assert "project:alpha" in outputs
    assert outputs["project:alpha"].exists()
    assert "dashboard" in outputs
    assert outputs["dashboard"].exists()


def test_load_all_syntheses_reads_from_disk(monkeypatch):
    monkeypatch.setattr(orch, "select_portfolio", lambda **kw: ["alpha"])
    sessions = [_mk_session("s-1", "alpha", n_messages=8)]
    monkeypatch.setattr(orch, "_bucket_sessions_by_short", lambda: {"alpha": sessions})
    orch.run_full_pipeline(_fake_provider())

    loaded = orch.load_all_syntheses()
    assert len(loaded) == 1
    assert loaded[0].project_short == "alpha"

    loaded_facts = orch.load_all_session_facts()
    assert len(loaded_facts) == 1


# ---------------------------------------------------------------------------
# Parallelism
# ---------------------------------------------------------------------------


def _mk_portfolio(monkeypatch, shorts: list[str]):
    """Wire up the monkeypatches so the pipeline sees exactly `shorts`."""
    monkeypatch.setattr(orch, "select_portfolio", lambda **kw: shorts)
    buckets = {
        s: [_mk_session(f"{s}-s1", s, n_messages=6)]
        for s in shorts
    }
    monkeypatch.setattr(orch, "_bucket_sessions_by_short", lambda: buckets)
    return buckets


def test_parallel_produces_same_outputs_as_serial(monkeypatch):
    """Running with max_workers > 1 should yield the same set of syntheses."""
    shorts = [f"proj-{i}" for i in range(5)]
    _mk_portfolio(monkeypatch, shorts)

    result = orch.run_full_pipeline(
        _fake_provider(), max_workers=4,
    )
    produced = {s.project_short for s in result.all_syntheses}
    assert produced == set(shorts)


def test_parallel_idempotent_second_run_is_free(monkeypatch):
    shorts = [f"proj-{i}" for i in range(4)]
    _mk_portfolio(monkeypatch, shorts)

    first = _fake_provider()
    orch.run_full_pipeline(first, max_workers=3)
    assert len(first.calls) > 0

    second = _fake_provider()
    result2 = orch.run_full_pipeline(second, max_workers=3)
    assert result2.timeline_cached is True
    assert result2.cross_cached is True
    assert second.calls == []


def test_max_workers_1_falls_back_to_serial(monkeypatch):
    """max_workers=1 should use the serial loop (no ThreadPoolExecutor)."""
    shorts = ["a", "b"]
    _mk_portfolio(monkeypatch, shorts)
    result = orch.run_full_pipeline(_fake_provider(), max_workers=1)
    order = [p.project_short for p in result.per_project]
    assert order == shorts


# ---------------------------------------------------------------------------
# Cancellation
# ---------------------------------------------------------------------------


def test_cancel_before_start_skips_everything(monkeypatch):
    shorts = ["a", "b", "c"]
    _mk_portfolio(monkeypatch, shorts)
    token = orch.CancellationToken()
    token.cancel()

    provider = _fake_provider()
    result = orch.run_full_pipeline(
        provider, max_workers=2, cancel_token=token,
    )
    assert provider.calls == []
    assert all(p.error == "cancelled" for p in result.per_project)
    assert result.timeline_cached is False
    assert result.cross_cached is False


def test_cancel_mid_run_serial(monkeypatch):
    """
    Serial mode: tripping the token mid-iteration stops new projects
    from starting. Projects that already ran keep their output.
    """
    shorts = ["a", "b", "c", "d"]
    _mk_portfolio(monkeypatch, shorts)
    token = orch.CancellationToken()

    def prog(ev, data):
        if ev == "stage2_done":
            token.cancel()

    result = orch.run_full_pipeline(
        _fake_provider(),
        max_workers=1,
        progress=prog,
        cancel_token=token,
    )
    completed = [p for p in result.per_project if p.synthesis is not None]
    assert 1 <= len(completed) < len(shorts), (
        f"expected partial completion; got {len(completed)}/{len(shorts)}"
    )


def test_cancellation_token_basic():
    token = orch.CancellationToken()
    assert token.is_cancelled is False
    token.cancel()
    assert token.is_cancelled is True
