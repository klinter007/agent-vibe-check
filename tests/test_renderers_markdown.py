"""
Markdown renderer tests. Verify output contains expected content and doesn't
crash on empty/partial inputs.

No emojis check is part of the project rule, but we don't enforce it here (no
emojis in the code == no emojis in the output).
"""

from __future__ import annotations

import pytest

from agent_vibe_check.pipeline.schemas import (
    CrossProject,
    CrossProjectAnomaly,
    CrossProjectCell,
    ProjectAICollaboration,
    ProjectArc,
    ProjectArcPhase,
    ProjectDNA,
    ProjectKeyDecision,
    ProjectPattern,
    ProjectPivot,
    ProjectSkillEntry,
    ProjectSynthesis,
    SCHEMA_VERSION,
    Timeline,
    TimelineActiveProject,
    TimelineNarrative,
    TimelinePeriod,
    TimelineSkillSnapshot,
)
from agent_vibe_check.renderers.markdown import (
    render_cross_project_md,
    render_executive_summary_md,
    render_project_md,
    render_timeline_md,
    write_global_reports,
    write_project_reports,
)


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path))
    yield tmp_path


def _synthesis() -> ProjectSynthesis:
    return ProjectSynthesis(
        schema_version=SCHEMA_VERSION,
        project_short="demo",
        project_folder="Users-me-projects-demo",
        real_path="/Users/me/projects/demo",
        first_session="2025-06-01T00:00:00",
        last_session="2026-01-15T00:00:00",
        session_count=12,
        message_count=400,
        dna=ProjectDNA(
            what_it_is="A CLI for extracting knowledge from Cursor transcripts.",
            key_components=["parser", "pipeline"],
            architecture_decisions=["stdlib-first"],
            constraints=["runs locally"],
        ),
        arc=ProjectArc(
            phases=[
                ProjectArcPhase("2025-06", "2025-08", "prototype", "single LLM call"),
                ProjectArcPhase("2025-09", "2026-01", "pipeline", "multi-stage"),
            ],
            pivots=[ProjectPivot("2025-09", "monolith", "pipeline", "context limits")],
        ),
        skill_profile={
            "specification": ProjectSkillEntry(
                peak_level="strong", current_level="strong",
                trajectory="stable", confidence="high",
                evidence_sessions=["s-1", "s-5", "s-9"],
            ),
            "steering": ProjectSkillEntry(
                peak_level="working", current_level="strong",
                trajectory="improving", confidence="med",
                evidence_sessions=["s-3", "s-7"],
            ),
        },
        project_specific_dimensions=[
            {"dimension": "cursor-internals", "level": "strong", "notes": "deep"}
        ],
        recurring_strengths=[
            ProjectPattern("hermetic tests", evidence_sessions=["s-1", "s-3"])
        ],
        recurring_pitfalls=[
            ProjectPattern(
                "occasional scope creep", fix="explicit scope",
                evidence_sessions=["s-4"],
            )
        ],
        key_decisions=[
            ProjectKeyDecision("use stdlib-only", "avoid dep hell", "s-1", 12),
        ],
        ai_collaboration=ProjectAICollaboration(
            what_worked="tight scoping",
            what_broke_down="",
            typical_failure_mode="",
        ),
    )


def _timeline() -> Timeline:
    return Timeline(
        schema_version=SCHEMA_VERSION,
        generated_at="2026-04-17T00:00:00",
        periods_monthly=[
            TimelinePeriod(
                period="2025-06", sessions=3,
                active_projects=[TimelineActiveProject("demo", 3)],
                top_topics=["specification"],
                skill_snapshot={
                    "specification": TimelineSkillSnapshot(level="working", confidence="med"),
                },
            ),
            TimelinePeriod(
                period="2025-07", sessions=5,
                active_projects=[TimelineActiveProject("demo", 5)],
                top_topics=["specification", "tests"],
                skill_snapshot={
                    "specification": TimelineSkillSnapshot(level="strong", confidence="high"),
                },
                deltas_vs_prev={"specification": "working -> strong"},
            ),
        ],
        periods_quarterly=[
            TimelinePeriod(
                period="2025-Q2", sessions=8,
                active_projects=[TimelineActiveProject("demo", 8)],
                top_topics=["specification"],
            )
        ],
        narrative=TimelineNarrative(
            strongest_growth=["specification"],
            surprising_moves=["typescript went quiet"],
            emerging_interests=["learning-loop"],
        ),
    )


def _cross() -> CrossProject:
    return CrossProject(
        schema_version=SCHEMA_VERSION,
        generated_at="2026-04-17T00:00:00",
        dimension_matrix={
            "specification": {
                "demo": CrossProjectCell(level="strong", sessions=5),
                "legacy": CrossProjectCell(level="beginner", sessions=2),
            },
            "steering": {
                "demo": CrossProjectCell(level="working", sessions=3),
            },
        },
        anomalies=[
            CrossProjectAnomaly(
                observation="Strong python except in legacy",
                dimensions_involved=["specification"],
                evidence=[
                    {"project": "demo", "level": "strong"},
                    {"project": "legacy", "level": "beginner"},
                ],
                hypothesis="legacy predates current idioms",
            )
        ],
    )


# --- project renderer -----------------------------------------------------


def test_render_project_md_has_expected_sections():
    syn = _synthesis()
    syn.headline = "A local CLI that turns AI chat logs into reports."
    syn.outcomes = {"shipped": 7, "unknown": 5}
    syn.tools = {"Claude Code": 9, "Cursor": 3}
    syn.friction = [{"type": "loop", "cause": "missing-context", "count": 2, "cost_messages": 30,
                     "examples": ["retried the same fix"]}]
    syn.friction_cost = 30
    syn.project_rules = [{"rule": "Use the project venv.", "count": 3}]
    md = render_project_md(syn)
    assert "# Project: demo" in md
    assert "> A local CLI" in md
    assert "Sessions: 12" in md
    assert "7 shipped" in md and "Claude Code (9)" in md
    assert "## What it is" in md and "A CLI for extracting knowledge" in md
    assert "## Vibes on this project" in md and "`specification`" in md
    assert "## Friction" in md and "missing-context" in md
    assert "## What works" in md and "## What keeps going wrong" in md
    assert "## Rules for this project" in md and "Use the project venv." in md
    assert "## Arc" in md and "monolith -> pipeline" in md
    assert "## Key decisions" in md and "use stdlib-only" in md


def test_render_project_md_handles_empty_synthesis():
    syn = ProjectSynthesis(
        schema_version=SCHEMA_VERSION, project_short="empty",
        project_folder="p", real_path=None,
        first_session="", last_session="",
        session_count=0, message_count=0,
    )
    md = render_project_md(syn)
    assert "# Project: empty" in md
    assert "Friction: 0 messages" in md


# --- timeline renderer ---------------------------------------------------


def test_render_timeline_md_has_sections_and_deltas():
    md = render_timeline_md(_timeline())
    assert "# Timeline" in md
    assert "## Strongest growth" in md
    assert "- specification" in md
    assert "## Quarterly" in md
    assert "### 2025-Q2" in md
    assert "## Monthly" in md
    assert "### 2025-06" in md
    assert "### 2025-07" in md
    assert "working -> strong" in md


def test_render_timeline_md_handles_empty():
    empty = Timeline(schema_version=SCHEMA_VERSION, generated_at="t")
    md = render_timeline_md(empty)
    assert "# Timeline" in md


# --- cross-project renderer ----------------------------------------------


def test_render_cross_project_md_has_matrix_and_anomalies():
    md = render_cross_project_md(_cross())
    assert "# Cross-project comparison" in md
    assert "Anomalies worth noticing" in md
    assert "Strong python except in legacy" in md
    assert "## Dimension matrix" in md
    # One of the taxonomy groups should appear as a subsection header
    assert "### Vibes" in md
    assert "strong (5)" in md
    assert "beginner (2)" in md
    # Missing cells should render as em-dash
    assert "—" in md


def test_render_cross_project_md_empty():
    md = render_cross_project_md(
        CrossProject(schema_version=SCHEMA_VERSION, generated_at="t")
    )
    assert "no cross-project data yet" in md


# --- executive summary ----------------------------------------------------


def _profile() -> dict:
    return {
        "stats": {
            "sessions": 40, "projects": 3, "tools_used": ["Claude Code", "Cursor"],
            "first": "2026-01-01", "last": "2026-06-01",
            "vibes": {"specification": {"score": 61, "n": 9}, "context": {"score": None, "n": 0}},
            "friction_messages": 120, "friction_share": 0.08,
            "friction_by_type": {"loop": 80, "correction": 40},
            "tools": [{"tool": "Cursor", "sessions": 30, "ship_rate": 0.5, "friction_per_session": 3.0}],
        },
        "archetype": {"name": "The Detective", "tagline": "Nothing is done until you have seen it run."},
        "narrative": {
            "headline": "Checks everything twice.",
            "strengths": [{"title": "Tight checks", "detail": "Asks for test output."}],
            "weakness": {"title": "Late context", "detail": "Gives files late.", "fix": "Point to files first."},
            "top_rules": [{"rule": "Run the tests before you say done.", "why": "x", "sessions": 6}],
        },
    }


def test_render_profile_md_has_sections():
    from agent_vibe_check.renderers.markdown import render_profile_md
    md = render_profile_md(_profile())
    assert "# Vibe Check profile" in md
    assert "## The Detective" in md
    assert "`specification` | 61" in md
    assert "## Friction" in md and "120 messages lost" in md
    assert "## Agents compared" in md and "50%" in md
    assert "Weakness: Late context" in md
    assert "Run the tests before you say done. (6 sessions)" in md


def test_render_profile_md_empty_profile():
    md = render_executive_summary_md(_timeline(), _cross(), [_synthesis()])
    assert "# Vibe Check profile" in md


# --- writers -------------------------------------------------------------


def test_write_project_reports_writes_file():
    syn = _synthesis()
    paths = write_project_reports("demo", syn)
    assert paths["project"].exists()
    content = paths["project"].read_text()
    assert "# Project: demo" in content


def test_write_global_reports_writes_all_files():
    paths = write_global_reports(_timeline(), _cross(), [_synthesis()], profile=_profile())
    assert paths["timeline"].exists()
    assert paths["cross_project"].exists()
    assert paths["profile"].exists()
    assert "Run the tests before you say done." in paths["rules"].read_text()
    assert "Timeline" in paths["timeline"].read_text()
    assert "Cross-project" in paths["cross_project"].read_text()


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
