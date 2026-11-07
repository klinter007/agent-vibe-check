"""
Stage 4 (cross-project) hermetic tests.

Covers:
  - local matrix construction (sparse, unobserved dropped)
  - non-canonical dimensions dropped
  - LLM-produced anomalies parsed and attached
  - cache roundtrip on matching input fingerprint
  - graceful handling when provider is None or empty matrix
"""

from __future__ import annotations

import json

import pytest

from agent_vibe_check.pipeline import cross_project as cp
from agent_vibe_check.pipeline.schemas import (
    SCHEMA_VERSION,
    ProjectAICollaboration,
    ProjectArc,
    ProjectDNA,
    ProjectSkillEntry,
    ProjectSynthesis,
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


def _syn(
    short: str,
    profile: dict[str, tuple[str, int]] | None = None,
    sessions: int = 5,
    messages: int = 100,
) -> ProjectSynthesis:
    """Build a minimal ProjectSynthesis with the given skill_profile shortcuts."""
    skill_profile: dict[str, ProjectSkillEntry] = {}
    for dim, (level, n_sessions) in (profile or {}).items():
        skill_profile[dim] = ProjectSkillEntry(
            peak_level=level,
            current_level=level,
            trajectory="stable",
            confidence="med",
            evidence_sessions=[f"{short}-s{i}" for i in range(n_sessions)],
        )
    return ProjectSynthesis(
        schema_version=SCHEMA_VERSION,
        project_short=short,
        project_folder=f"folder-{short}",
        real_path=None,
        first_session="2025-06-01T00:00:00",
        last_session="2026-01-01T00:00:00",
        session_count=sessions,
        message_count=messages,
        dna=ProjectDNA(what_it_is=f"description of {short}"),
        arc=ProjectArc(),
        skill_profile=skill_profile,
        ai_collaboration=ProjectAICollaboration(),
    )


# --- matrix ----------------------------------------------------------------


def test_build_matrix_basic():
    syns = [
        _syn("a", profile={"specification": ("strong", 3), "verification": ("working", 2)}),
        _syn("b", profile={"specification": ("beginner", 2)}),
    ]
    m = cp.build_matrix(syns)
    assert set(m.keys()) == {"specification", "verification"}
    assert m["specification"]["a"].level == "strong"
    assert m["specification"]["a"].sessions == 3
    assert m["specification"]["b"].level == "beginner"
    assert "verification" in m
    assert list(m["verification"].keys()) == ["a"]  # b doesn't have k8s observed


def test_build_matrix_drops_unobserved_entries():
    syns = [_syn("a", profile={"specification": ("unobserved", 1)})]
    m = cp.build_matrix(syns)
    assert m == {}


def test_build_matrix_drops_non_canonical_dimensions():
    syns = [_syn("a", profile={"fake-skill": ("strong", 3), "specification": ("working", 2)})]
    m = cp.build_matrix(syns)
    assert "fake-skill" not in m
    assert "specification" in m


def test_build_matrix_sparse_across_projects():
    syns = [
        _syn("a", profile={"specification": ("strong", 3)}),
        _syn("b", profile={"context": ("working", 2)}),
        _syn("c", profile={"specification": ("expert", 5), "context": ("strong", 3)}),
    ]
    m = cp.build_matrix(syns)
    assert set(m["specification"].keys()) == {"a", "c"}
    assert set(m["context"].keys()) == {"b", "c"}


# --- full pipeline --------------------------------------------------------


_ANOMALIES = {
    "anomalies": [
        {
            "observation": "Strong python everywhere except 'legacy' project",
            "dimensions_involved": ["specification"],
            "evidence": [
                {"project": "a", "level": "strong"},
                {"project": "legacy", "level": "beginner"},
            ],
            "hypothesis": "legacy predates current idioms",
        },
        {
            "observation": "Testing discipline uniformly low",
            "dimensions_involved": ["steering"],
            "evidence": [
                {"project": "a", "level": "beginner"},
                {"project": "b", "level": "beginner"},
            ],
            "hypothesis": "not yet a habit across the portfolio",
        },
    ]
}


def test_build_cross_project_happy_path():
    syns = [
        _syn("a", profile={"specification": ("strong", 3), "steering": ("beginner", 2)}),
        _syn("b", profile={"steering": ("beginner", 2)}),
        _syn("legacy", profile={"specification": ("beginner", 3)}),
    ]
    provider = FakeProvider(responses=[json.dumps(_ANOMALIES)])
    r = cp.build_cross_project(syns, provider)
    assert r.error is None
    assert r.cross is not None
    assert len(r.cross.anomalies) == 2
    assert "specification" in r.cross.dimension_matrix
    assert provider.calls == 1


def test_build_cross_project_cache_hit_on_match():
    syns = [
        _syn("a", profile={"specification": ("strong", 3)}),
        _syn("b", profile={"specification": ("beginner", 2)}),
    ]
    provider = FakeProvider(responses=[json.dumps(_ANOMALIES)])
    cp.build_cross_project(syns, provider)
    r2 = cp.build_cross_project(syns, provider)
    assert r2.cached is True
    assert provider.calls == 1


def test_build_cross_project_invalidates_cache_on_skill_change():
    syns = [
        _syn("a", profile={"specification": ("strong", 3)}),
        _syn("b", profile={"specification": ("beginner", 2)}),
    ]
    provider = FakeProvider(responses=[json.dumps(_ANOMALIES), json.dumps(_ANOMALIES)])
    cp.build_cross_project(syns, provider)
    assert provider.calls == 1

    # Change b's python level
    syns[1].skill_profile["specification"] = ProjectSkillEntry(
        peak_level="working", current_level="working",
        trajectory="improving", confidence="med",
        evidence_sessions=["b-s0", "b-s1", "b-s2"],
    )
    r = cp.build_cross_project(syns, provider)
    assert r.cached is False
    assert provider.calls == 2


def test_build_cross_project_without_provider_still_builds_matrix():
    syns = [
        _syn("a", profile={"specification": ("strong", 3)}),
        _syn("b", profile={"specification": ("beginner", 2)}),
    ]
    r = cp.build_cross_project(syns, None)
    assert r.cross is not None
    assert r.cross.anomalies == []
    assert "specification" in r.cross.dimension_matrix


def test_build_cross_project_empty_input_returns_error():
    r = cp.build_cross_project([], None)
    assert r.cross is None
    assert r.error == "no project syntheses provided"


def test_build_cross_project_bad_llm_json_still_produces_matrix():
    syns = [
        _syn("a", profile={"specification": ("strong", 3)}),
        _syn("b", profile={"specification": ("beginner", 2)}),
    ]
    provider = FakeProvider(responses=["totally not json"])
    r = cp.build_cross_project(syns, provider)
    assert r.cross is not None
    assert r.cross.anomalies == []
    assert "specification" in r.cross.dimension_matrix


def test_load_cross_project_reads_disk():
    syns = [_syn("a", profile={"specification": ("strong", 3)})]
    cp.build_cross_project(syns, None)
    loaded = cp.load_cross_project()
    assert loaded is not None
    assert "specification" in loaded.dimension_matrix


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
