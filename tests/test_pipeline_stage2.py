"""
Stage 2 (project synthesizer) hermetic tests.

Covers:
  - local aggregation (skill_profile peak/current/trajectory, key_decisions dedup)
  - LLM call path with FakeProvider
  - caching via input fingerprint
  - graceful handling of empty/bad LLM output
"""

from __future__ import annotations

import json

import pytest

from agent_vibe_check.pipeline import project_synthesizer as ps
from agent_vibe_check.pipeline.schemas import (
    Decision,
    Evidence,
    ProblemSolved,
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
    # The atlas workspace map reads the live Cursor install. Stub it so
    # tests stay hermetic.
    monkeypatch.setattr(
        "agent_vibe_check.pipeline.project_synthesizer._build_workspace_map",
        lambda: {},
    )
    yield tmp_path


def _fact(
    sid: str,
    ts: str,
    skills: list[tuple[str, str]] | None = None,
    decisions: list[str] | None = None,
    msg_count: int = 10,
) -> SessionFacts:
    return SessionFacts(
        schema_version=SCHEMA_VERSION,
        session_id=sid,
        project_folder="Users-me-projects-demo",
        project_short="demo",
        timestamp=ts,
        source="jsonl",
        message_count=msg_count,
        topic=f"topic-{sid}",
        summary=f"summary-{sid}",
        skills_observed=[
            SkillSignal(
                dimension=d, level=lv, confidence="med",
                evidence=[Evidence(msg_idx=1, behavior="evidence")],
            )
            for d, lv in (skills or [])
        ],
        decisions=[Decision(d, "rationale", 5) for d in (decisions or [])],
        user_communication=UserCommunication(rating="good"),
    )


# --- local aggregation -----------------------------------------------------


def test_skill_profile_peak_and_current_from_chronological_order():
    facts = [
        _fact("s-1", "2025-06-01T00:00:00", skills=[("specification", "beginner")]),
        _fact("s-2", "2025-07-01T00:00:00", skills=[("specification", "strong")]),
        _fact("s-3", "2025-08-01T00:00:00", skills=[("specification", "working")]),
    ]
    profile = ps._aggregate_skill_profile(facts)
    assert "specification" in profile
    entry = profile["specification"]
    assert entry.peak_level == "strong"
    assert entry.current_level == "working"
    # Three points with positive trend first->avg(last two): avg_first=0 (beginner),
    # avg_second=(3+2)/2=2.5, delta=2.5 -> improving
    assert entry.trajectory == "improving"
    assert entry.evidence_sessions == ["s-1", "s-2", "s-3"]


def test_skill_profile_insufficient_data_for_single_observation():
    facts = [_fact("s-1", "2025-06-01", skills=[("specification", "working")])]
    profile = ps._aggregate_skill_profile(facts)
    assert profile["specification"].trajectory == "insufficient-data"


def test_skill_profile_regressing_detected():
    facts = [
        _fact("s-1", "2025-06-01", skills=[("context", "strong")]),
        _fact("s-2", "2025-07-01", skills=[("context", "strong")]),
        _fact("s-3", "2025-08-01", skills=[("context", "beginner")]),
        _fact("s-4", "2025-09-01", skills=[("context", "beginner")]),
    ]
    profile = ps._aggregate_skill_profile(facts)
    assert profile["context"].trajectory == "regressing"


def test_skill_profile_drops_non_canonical_dimensions():
    facts = [_fact("s-1", "2025-06-01", skills=[("specification", "strong"), ("fake-skill", "expert")])]
    profile = ps._aggregate_skill_profile(facts)
    assert "specification" in profile
    assert "fake-skill" not in profile


def test_key_decisions_deduplicates_and_limits():
    facts = [
        _fact("s-1", "2025-06-01", decisions=["use redis", "use redis"]),
        _fact("s-2", "2025-07-01", decisions=["USE REDIS", "pick postgres"]),
    ]
    decisions = ps._gather_key_decisions(facts)
    texts = [d.decision for d in decisions]
    assert len(texts) == 2  # "use redis" dedup'd case-insensitively
    assert "pick postgres" in texts


def test_input_fingerprint_stable_across_reorderings():
    a = _fact("s-1", "2025-06-01")
    b = _fact("s-2", "2025-07-01")
    fp1 = ps._input_fingerprint([a, b])
    fp2 = ps._input_fingerprint([b, a])
    assert fp1 == fp2, "fingerprint must be order-insensitive"


# --- synthesize_project ----------------------------------------------------


_NARRATIVE = {
    "dna": {
        "what_it_is": "CLI tool for extracting knowledge from Cursor transcripts",
        "key_components": ["parser", "pipeline"],
        "architecture_decisions": ["stdlib-first"],
        "constraints": ["local data only"],
    },
    "arc": {
        "phases": [
            {"start": "2025-06", "end": "2025-08", "name": "v1 monolith",
             "focus": "single LLM call per project"}
        ],
        "pivots": [
            {"when": "2025-09", "from": "monolith", "to": "pipeline",
             "why": "hit context window limits"}
        ],
    },
    "recurring_strengths": [
        {"pattern": "hermetic tests with fakes",
         "evidence_sessions": ["s-1", "s-2"]}
    ],
    "recurring_pitfalls": [
        {"pattern": "occasional unrelated edits",
         "fix": "tighten scope", "evidence_sessions": ["s-1"]}
    ],
    "project_specific_dimensions": [
        {"dimension": "cursor-internals", "level": "strong", "notes": "deep on SQLite layer"}
    ],
    "key_decisions": [
        {"decision": "meta: prefers simplest approach", "rationale": "seen across sessions"}
    ],
    "ai_collaboration": {
        "what_worked": "clear scope per session",
        "what_broke_down": "",
        "typical_failure_mode": "",
    },
}


def test_synthesize_project_happy_path_writes_and_returns():
    facts = [
        _fact("s-1", "2025-06-01", skills=[("specification", "beginner")], decisions=["use redis"]),
        _fact("s-2", "2025-07-01", skills=[("specification", "strong")]),
    ]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])

    result = ps.synthesize_project("demo", "Users-me-projects-demo",
                                   facts, provider)
    assert result.error is None
    assert result.cached is False
    assert result.synthesis is not None
    s = result.synthesis
    assert s.project_short == "demo"
    assert s.session_count == 2
    assert s.dna.what_it_is.startswith("CLI tool")
    assert s.arc.phases[0].name == "v1 monolith"
    assert s.arc.pivots[0].from_ == "monolith"  # roundtripped from "from" in JSON
    assert "specification" in s.skill_profile
    # Local decision + LLM meta decision merged
    decisions = [d.decision for d in s.key_decisions]
    assert "use redis" in decisions
    assert any("meta:" in d for d in decisions)
    assert s.real_path is None  # stubbed workspace map
    assert len(s.recurring_strengths) == 1


def test_synthesize_project_cache_hit_on_matching_fingerprint():
    facts = [
        _fact("s-1", "2025-06-01"), _fact("s-2", "2025-07-01"),
    ]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])
    ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    assert provider.calls == 1

    r2 = ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    assert r2.cached is True
    assert provider.calls == 1


def test_synthesize_project_cache_miss_when_new_session_added():
    facts = [_fact("s-1", "2025-06-01"), _fact("s-2", "2025-07-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE), json.dumps(_NARRATIVE)])
    ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    assert provider.calls == 1

    facts.append(_fact("s-3", "2025-08-01"))
    r = ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    assert r.cached is False
    assert provider.calls == 2


def test_synthesize_project_force_bypasses_cache():
    facts = [_fact("s-1", "2025-06-01"), _fact("s-2", "2025-07-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE), json.dumps(_NARRATIVE)])
    ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    r = ps.synthesize_project("demo", "Users-me-demo", facts, provider, force=True)
    assert r.cached is False
    assert provider.calls == 2


def test_synthesize_project_empty_llm_output_still_produces_synthesis():
    facts = [_fact("s-1", "2025-06-01"), _fact("s-2", "2025-07-01")]
    provider = FakeProvider(responses=["not valid json at all"])
    r = ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    assert r.synthesis is not None, "we should still produce a synthesis using local data"
    # dna fields empty but skill_profile populated from locals
    assert r.synthesis.dna.what_it_is == ""
    raw_path = ps._project_root("demo") / "project.raw.txt"
    assert raw_path.exists(), "raw response saved for debugging"


def test_synthesize_project_skips_when_below_min_sessions():
    facts = [_fact("s-1", "2025-06-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])
    r = ps.synthesize_project("demo", "Users-me-demo", facts, provider, min_sessions=3)
    assert r.synthesis is None
    assert r.skipped_reason and "too_few_sessions" in r.skipped_reason
    assert provider.calls == 0


def test_load_project_synthesis_reads_what_was_written():
    facts = [_fact("s-1", "2025-06-01"), _fact("s-2", "2025-07-01")]
    provider = FakeProvider(responses=[json.dumps(_NARRATIVE)])
    ps.synthesize_project("demo", "Users-me-demo", facts, provider)
    loaded = ps.load_project_synthesis("demo")
    assert loaded is not None
    assert loaded.project_short == "demo"
    assert loaded.arc.phases[0].name == "v1 monolith"


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
