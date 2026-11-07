"""
Unit tests for the pipeline taxonomy and schema layer.

These are hermetic: no LLM, no filesystem, no Cursor install required.
"""

from __future__ import annotations

import json

import pytest

from agent_vibe_check.pipeline import schemas
from agent_vibe_check.pipeline.schemas import (
    AIFailure,
    CrossProject,
    CrossProjectAnomaly,
    CrossProjectCell,
    Decision,
    Evidence,
    ProblemSolved,
    ProjectSynthesis,
    ProjectPivot,
    SessionFacts,
    SkillSignal,
    Timeline,
    TimelinePeriod,
    TimelineSkillSnapshot,
    UserCommunication,
    VibingMoment,
)
from agent_vibe_check.pipeline.taxonomy import (
    CANONICAL_DIMENSIONS,
    CONFIDENCES,
    DIMENSION_BY_NAME,
    DIMENSION_NAMES,
    LEVELS,
    LEVEL_RANK,
    TRAJECTORIES,
    compare_levels,
    dimension_list_for_prompt,
    is_canonical,
)


# --- Taxonomy ---------------------------------------------------------------


def test_canonical_dimensions_non_empty_and_unique():
    names = [d.name for d in CANONICAL_DIMENSIONS]
    assert len(names) == 6, "v2 taxonomy is the six vibe signals"
    assert len(names) == len(set(names)), "dimension names must be unique"


def test_dimension_by_name_lookup():
    for name in DIMENSION_NAMES:
        assert DIMENSION_BY_NAME[name].name == name
    assert is_canonical("specification") is True
    assert is_canonical("totally-made-up-skill") is False


def test_prompt_injection_contains_every_dimension():
    prompt = dimension_list_for_prompt()
    for d in CANONICAL_DIMENSIONS:
        assert f"`{d.name}`" in prompt, f"{d.name} missing from prompt injection"


def test_levels_ordered():
    assert LEVELS == ("unobserved", "beginner", "working", "strong", "expert")
    assert LEVEL_RANK["unobserved"] < LEVEL_RANK["beginner"] < LEVEL_RANK["expert"]
    assert compare_levels("beginner", "strong") == -1
    assert compare_levels("expert", "expert") == 0
    assert compare_levels("strong", "working") == 1


def test_confidences_and_trajectories_are_closed_sets():
    assert set(CONFIDENCES) == {"low", "med", "high"}
    assert "improving" in TRAJECTORIES and "insufficient-data" in TRAJECTORIES


# --- SessionFacts -----------------------------------------------------------


def _valid_session_fact() -> SessionFacts:
    return SessionFacts(
        schema_version=schemas.SCHEMA_VERSION,
        session_id="s-1",
        project_folder="projects-foo",
        project_short="foo",
        timestamp="2026-01-02T03:04:05",
        source="jsonl",
        message_count=12,
        topic="debugging flaky test",
        summary="Tracked down a race in pytest fixture setup.",
        topics_touched=["pytest", "fixtures"],
        skills_observed=[
            SkillSignal(
                dimension="delegation",
                level="working",
                confidence="med",
                evidence=[Evidence(msg_idx=4, behavior="bisected by disabling fixtures")],
            ),
        ],
        decisions=[Decision("use session-scoped fixture", "avoids re-setup cost", msg_idx=9)],
        problems_solved=[
            ProblemSolved(
                symptom="intermittent failure",
                root_cause="race in fixture",
                fix="add event to gate setup",
                msg_idx_range=[2, 10],
            )
        ],
        open_questions=["is the race also present in CI runner?"],
        user_communication=UserCommunication(rating="good", notes="clear repro"),
        vibing_moments=[VibingMoment("caught race by reading logs", msg_idxs=[3, 4])],
        ai_failures=[AIFailure("suggested sleep() as a fix", msg_idx=6, user_caught_it=True)],
    )


def test_session_facts_roundtrip():
    fact = _valid_session_fact()
    d = fact.to_dict()
    # Ensure JSON-serializable.
    json.dumps(d)
    back = SessionFacts.from_dict(d)
    assert back.session_id == fact.session_id
    assert back.skills_observed[0].dimension == "delegation"
    assert back.skills_observed[0].evidence[0].msg_idx == 4
    assert back.validate() == []


def test_session_facts_validator_flags_bad_dimension():
    fact = _valid_session_fact()
    fact.skills_observed[0].dimension = "nonsense-skill"
    issues = fact.validate()
    assert any("unknown dimension" in i for i in issues)


def test_session_facts_validator_flags_bad_level_and_confidence():
    fact = _valid_session_fact()
    fact.skills_observed[0].level = "godlike"
    fact.skills_observed[0].confidence = "vibes"
    issues = fact.validate()
    assert any("invalid level" in i for i in issues)
    assert any("invalid confidence" in i for i in issues)


def test_session_facts_validator_flags_evidence_msg_idx_out_of_range():
    fact = _valid_session_fact()
    fact.message_count = 5
    fact.skills_observed[0].evidence[0].msg_idx = 99
    issues = fact.validate()
    assert any("out of range" in i for i in issues)


def test_session_facts_schema_version_mismatch():
    fact = _valid_session_fact()
    fact.schema_version = 999
    issues = fact.validate()
    assert any("schema_version" in i for i in issues)


def test_session_facts_tolerant_of_missing_optional_fields():
    minimal = {
        "session_id": "bare",
        "message_count": 1,
        "project_folder": "p",
        "project_short": "p",
        "source": "jsonl",
        "timestamp": "2026-01-01T00:00:00",
    }
    fact = SessionFacts.from_dict(minimal)
    assert fact.topic == ""
    assert fact.skills_observed == []
    assert fact.validate() == []


# --- ProjectSynthesis -------------------------------------------------------


def test_project_synthesis_pivot_from_roundtrip():
    proj = ProjectSynthesis(
        schema_version=schemas.SCHEMA_VERSION,
        project_short="foo",
        project_folder="projects-foo",
        real_path="/Users/x/projects/foo",
        first_session="2025-06-01T00:00:00",
        last_session="2026-01-01T00:00:00",
        session_count=10,
        message_count=250,
    )
    proj.arc.pivots.append(
        ProjectPivot(when="2025-10", from_="flask", to="fastapi", why="async support")
    )
    d = proj.to_dict()
    json.dumps(d)
    assert d["arc"]["pivots"][0]["from"] == "flask"
    assert "from_" not in d["arc"]["pivots"][0]
    back = ProjectSynthesis.from_dict(d)
    assert back.arc.pivots[0].from_ == "flask"
    assert back.validate() == []


def test_project_synthesis_validator_flags_unknown_skill():
    proj = ProjectSynthesis(
        schema_version=schemas.SCHEMA_VERSION,
        project_short="foo", project_folder="p", real_path=None,
        first_session="", last_session="", session_count=0, message_count=0,
    )
    from agent_vibe_check.pipeline.schemas import ProjectSkillEntry
    proj.skill_profile["not-a-dimension"] = ProjectSkillEntry(
        peak_level="strong", current_level="strong",
        trajectory="stable", confidence="high",
    )
    issues = proj.validate()
    assert any("unknown dimension" in i for i in issues)


# --- Timeline / CrossProject ------------------------------------------------


def test_timeline_roundtrip():
    tl = Timeline(
        schema_version=schemas.SCHEMA_VERSION,
        generated_at="2026-04-17T00:00:00",
        periods_monthly=[
            TimelinePeriod(
                period="2025-11",
                sessions=42,
                skill_snapshot={"specification": TimelineSkillSnapshot(level="strong", confidence="high")},
                deltas_vs_prev={"verification": "beginner -> working"},
            )
        ],
        periods_quarterly=[],
    )
    d = tl.to_dict()
    json.dumps(d)
    back = Timeline.from_dict(d)
    assert back.periods_monthly[0].skill_snapshot["specification"].level == "strong"
    assert back.periods_monthly[0].deltas_vs_prev["verification"] == "beginner -> working"


def test_cross_project_roundtrip():
    cp = CrossProject(
        schema_version=schemas.SCHEMA_VERSION,
        generated_at="2026-04-17T00:00:00",
        dimension_matrix={
            "specification": {
                "demo-app": CrossProjectCell(level="strong", sessions=42),
                "other-project": CrossProjectCell(level="working", sessions=8),
            }
        },
        anomalies=[
            CrossProjectAnomaly(
                observation="weaker debugging here",
                dimensions_involved=["delegation"],
                evidence=[{"project": "foo", "session": "s-1"}],
                hypothesis="short sessions, no time to bisect",
            )
        ],
    )
    d = cp.to_dict()
    json.dumps(d)
    back = CrossProject.from_dict(d)
    assert back.dimension_matrix["specification"]["demo-app"].level == "strong"
    assert back.anomalies[0].dimensions_involved == ["delegation"]


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
