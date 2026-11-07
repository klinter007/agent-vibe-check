"""
Stage 3 (timeline) hermetic tests.

Covers:
  - monthly + quarterly bucketing with gap fill
  - dominant level picks highest weighted observation
  - deltas_vs_prev computed for level transitions
  - LLM narrative parsed and attached
  - cache roundtrip
  - empty / malformed timestamps handled gracefully
"""

from __future__ import annotations

import json

import pytest

from agent_vibe_check.pipeline import timeline as tl
from agent_vibe_check.pipeline.schemas import (
    SCHEMA_VERSION,
    SessionFacts,
    SkillSignal,
    UserCommunication,
)


class FakeInfo:
    name = "fake"
    display_name = "Fake"
    env_var = "FAKE_KEY"
    default_model = "fake-model"
    signup_url = "https://example.com"


class FakeProvider:
    info = FakeInfo()

    def __init__(self, responses=None, raises=None):
        self.responses = list(responses or [])
        self.raises = raises
        self.calls = 0

    def analyze(self, prompt, *, model=None, system=None):
        self.calls += 1
        if self.raises:
            raise self.raises
        if not self.responses:
            return "{}"
        return self.responses.pop(0)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path))
    monkeypatch.setattr("agent_vibe_check.pipeline.llm_utils.time.sleep", lambda *_: None)
    yield tmp_path


def _fact(sid, ts, project="demo", skills=None, topics=None):
    return SessionFacts(
        schema_version=SCHEMA_VERSION,
        session_id=sid,
        project_folder=f"folder-{project}",
        project_short=project,
        timestamp=ts,
        source="jsonl",
        message_count=10,
        topic=f"topic-{sid}",
        summary="sum",
        topics_touched=topics or [],
        skills_observed=[
            SkillSignal(dimension=d, level=lv, confidence=c) for (d, lv, c) in (skills or [])
        ],
        user_communication=UserCommunication(rating="good"),
    )


# --- bucketing primitives --------------------------------------------------


def test_month_and_quarter_bucket_from_iso():
    assert tl.month_bucket("2025-07-15T12:34:56") == "2025-07"
    assert tl.quarter_bucket("2025-07-15T12:34:56") == "2025-Q3"
    assert tl.month_bucket("2025-01-01") == "2025-01"
    assert tl.quarter_bucket("2025-12-31") == "2025-Q4"


def test_bucket_returns_none_for_bad_timestamp():
    assert tl.month_bucket("not a date") is None
    assert tl.quarter_bucket("") is None


def test_iter_months_fills_gaps():
    months = tl._iter_months_inclusive("2025-01-15", "2025-04-02")
    assert months == ["2025-01", "2025-02", "2025-03", "2025-04"]


def test_iter_quarters_fills_gaps():
    quarters = tl._iter_quarters_inclusive("2025-01-15", "2026-04-02")
    assert quarters == ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"]


# --- dominant level -------------------------------------------------------


def test_dominant_level_picks_highest_weighted():
    # Two "beginner" (low weight) vs one "strong" (high weight)
    obs = [("beginner", "low"), ("beginner", "low"), ("strong", "high")]
    level, conf = tl._dominant_level(obs)
    # Sum of weights: beginner=1+1=2, strong=3 -> strong wins
    assert level == "strong"
    assert conf == "high"


def test_dominant_level_tiebreak_on_rank():
    obs = [("working", "high"), ("strong", "high")]
    # Equal weight; tie-break on higher rank
    level, _ = tl._dominant_level(obs)
    assert level == "strong"


def test_dominant_level_empty_obs():
    assert tl._dominant_level([]) == ("unobserved", "low")


# --- aggregation ----------------------------------------------------------


def test_aggregate_period_counts_projects_and_topics():
    facts = [
        _fact("s-1", "2025-06-01", project="a", topics=["k8s"]),
        _fact("s-2", "2025-06-10", project="a", topics=["k8s", "context"]),
        _fact("s-3", "2025-06-20", project="b", topics=["learning-loop"]),
    ]
    period = tl._aggregate_period("2025-06", facts)
    assert period.sessions == 3
    proj_counts = {p.name: p.sessions for p in period.active_projects}
    assert proj_counts == {"a": 2, "b": 1}
    assert "k8s" in period.top_topics
    assert period.top_topics.index("k8s") == 0  # most frequent first


def test_fill_deltas_detects_level_change():
    p1 = tl._aggregate_period("2025-06", [
        _fact("s1", "2025-06-01", skills=[("specification", "beginner", "med")])
    ])
    p2 = tl._aggregate_period("2025-07", [
        _fact("s2", "2025-07-01", skills=[("specification", "strong", "med")])
    ])
    p3 = tl._aggregate_period("2025-08", [
        _fact("s3", "2025-08-01", skills=[("specification", "strong", "med")])
    ])
    periods = [p1, p2, p3]
    tl._fill_deltas(periods)
    assert periods[0].deltas_vs_prev == {}
    assert periods[1].deltas_vs_prev == {"specification": "beginner -> strong"}
    assert periods[2].deltas_vs_prev == {}


def test_bucket_facts_fills_monthly_gap_with_empty_period():
    facts = [
        _fact("s1", "2025-06-01"),
        _fact("s2", "2025-09-01"),
    ]
    monthly = tl._bucket_facts(facts, tl.month_bucket, tl._iter_months_inclusive)
    labels = [p.period for p in monthly]
    assert labels == ["2025-06", "2025-07", "2025-08", "2025-09"]
    # The two middle periods should have zero sessions
    zero_sessions = [p for p in monthly if p.sessions == 0]
    assert len(zero_sessions) == 2


def test_bucket_facts_quarterly_labels():
    facts = [
        _fact("s1", "2025-02-01"),
        _fact("s2", "2025-11-15"),
    ]
    quarterly = tl._bucket_facts(facts, tl.quarter_bucket, tl._iter_quarters_inclusive)
    labels = [p.period for p in quarterly]
    assert labels == ["2025-Q1", "2025-Q2", "2025-Q3", "2025-Q4"]


# --- full pipeline --------------------------------------------------------


_NARRATIVE = {
    "strongest_growth": ["specification", "delegation"],
    "surprising_moves": ["typescript (strong -> beginner at year end)"],
    "emerging_interests": ["verification", "learning-loop"],
}


def test_build_timeline_happy_path_with_narrative():
    facts = [
        _fact("s1", "2025-06-01", skills=[("specification", "beginner", "med")]),
        _fact("s2", "2025-07-01", skills=[("specification", "strong", "high")]),
    ]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])
    result = tl.build_timeline(facts, provider)
    assert result.error is None
    assert result.timeline is not None
    assert result.timeline.narrative.strongest_growth == ["specification", "delegation"]
    assert len(result.timeline.periods_monthly) == 2
    # June=Q2, July=Q3, so two quarterly buckets are expected.
    assert len(result.timeline.periods_quarterly) == 2
    assert provider.calls == 1


def test_build_timeline_cache_hit_skips_llm():
    facts = [_fact("s1", "2025-06-01"), _fact("s2", "2025-07-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])
    tl.build_timeline(facts, provider)
    r2 = tl.build_timeline(facts, provider)
    assert r2.cached is True
    assert provider.calls == 1


def test_build_timeline_cache_miss_after_new_fact():
    facts = [_fact("s1", "2025-06-01"), _fact("s2", "2025-07-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE), json.dumps(_NARRATIVE)])
    tl.build_timeline(facts, provider)
    facts.append(_fact("s3", "2025-08-01"))
    r = tl.build_timeline(facts, provider)
    assert r.cached is False
    assert provider.calls == 2


def test_build_timeline_no_provider_no_narrative():
    facts = [_fact("s1", "2025-06-01"), _fact("s2", "2025-07-01")]
    result = tl.build_timeline(facts, None)
    assert result.timeline is not None
    assert result.timeline.narrative.strongest_growth == []


def test_load_timeline_reads_what_was_written():
    facts = [_fact("s1", "2025-06-01"), _fact("s2", "2025-07-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])
    tl.build_timeline(facts, provider)
    loaded = tl.load_timeline()
    assert loaded is not None
    assert loaded.narrative.strongest_growth == ["specification", "delegation"]


def test_build_timeline_ignores_bad_timestamps():
    facts = [
        _fact("s1", "2025-06-01"),
        _fact("s_bad", "totally not a date"),
        _fact("s2", "2025-07-01"),
    ]
    result = tl.build_timeline(facts, None)
    # s_bad dropped, two valid sessions left -> 2 monthly periods
    assert result.timeline is not None
    total_sessions = sum(p.sessions for p in result.timeline.periods_monthly)
    assert total_sessions == 2


def test_build_timeline_empty_input():
    r = tl.build_timeline([], None)
    assert r.timeline is None
    assert r.error == "no facts provided"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
