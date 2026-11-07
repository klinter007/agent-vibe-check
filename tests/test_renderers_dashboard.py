"""Tests for the v2 HTML report, the Wrapped card, the share export and Stage 5 profile."""

from __future__ import annotations

import pytest

from agent_vibe_check.pipeline.profile import (
    ARCHETYPES,
    build_profile,
    compute_features,
    compute_stats,
    score_archetypes,
    vibe_scores,
)
from agent_vibe_check.pipeline.schemas import (
    SCHEMA_VERSION,
    Friction,
    Outcome,
    ProjectSynthesis,
    RuleCandidate,
    SessionFacts,
    SkillSignal,
)
from agent_vibe_check.renderers.card import render_card_svg, wrap
from agent_vibe_check.renderers.dashboard import render_html, rules_block, write_dashboard
from agent_vibe_check.renderers.share import ShareLeakError, redacted_profile, write_share


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path))
    yield tmp_path


def _fact(i: int, *, project="infra-lab", tool="Cursor", driver="human", level="strong",
          status="shipped", friction=None, rules=None) -> SessionFacts:
    return SessionFacts(
        schema_version=SCHEMA_VERSION, session_id=f"s{i}", project_folder=project,
        project_short=project, timestamp=f"2026-0{1 + i % 6}-1{i % 9}T1{i % 9}:00:00",
        source="jsonl", message_count=40, tool=tool, driver=driver, task_type="fix-bug",
        skills_observed=[SkillSignal("verification", level, "high"), SkillSignal("specification", "working", "med")],
        outcome=Outcome(status=status),
        friction=friction or [],
        rule_candidates=rules or [],
    )


def _facts() -> list[SessionFacts]:
    out = []
    for i in range(12):
        out.append(_fact(
            i, tool="Claude Code" if i % 2 else "Cursor",
            status=["shipped", "partial", "abandoned", "unknown"][i % 4],
            friction=[Friction(type="loop", cause="missing-context", cost_messages=5, what_happened="retry")] if i % 3 == 0 else [],
            rules=[RuleCandidate(rule="Run the tests before you say a fix works.", scope="general")] if i % 2 == 0 else [],
        ))
    out.append(_fact(99, project="side-app", driver="agent", level="expert"))
    return out


def _syn(name="infra-lab") -> ProjectSynthesis:
    return ProjectSynthesis(
        schema_version=SCHEMA_VERSION, project_short=name, project_folder=name, real_path=None,
        first_session="2026-01-01", last_session="2026-06-01", session_count=12, message_count=480,
        headline="A GPU cluster for training jobs.", outcomes={"shipped": 3}, tools={"Cursor": 6},
    )


# -- profile ------------------------------------------------------------------


def test_vibe_scores_scale_and_counts():
    scores = vibe_scores(_facts())
    assert scores["verification"]["n"] == 13  # vibe_scores itself does not filter
    assert 0 <= scores["verification"]["score"] <= 100
    assert scores["steering"]["score"] is None


def test_compute_stats_scores_only_own_sessions():
    st = compute_stats(_facts(), [_syn()])
    # Only sessions the person typed are scored; the agent-driven one is counted.
    assert st["sessions"] == 12
    assert st["reviewed_sessions"] == 13
    assert st["agent_driven_sessions"] == 1
    assert "orchestration_vibes" not in st
    assert st["friction_by_type"] == {"loop": 20}
    assert {t["tool"] for t in st["tools"]} == {"Cursor", "Claude Code"}
    assert 0 < st["ship_rate"] < 1


def test_archetype_scoring_is_complete_and_sorted():
    st = compute_stats(_facts(), [_syn()])
    ranked = score_archetypes(compute_features(st, _facts()))
    assert {k for k, _ in ranked} == {a.key for a in ARCHETYPES}
    assert ranked == sorted(ranked, key=lambda kv: -kv[1])


class _FakeProvider:
    class info:
        name = "fake"
        default_model = "m"

    def __init__(self):
        self.calls = 0

    def analyze(self, prompt, *, model=None, system=None):
        self.calls += 1
        return ('{"headline": "Checks things.", "top_rules": [{"rule": "Run the tests before you say a fix works.",'
                ' "why": "agents claim success", "sessions": 6}], "strengths": [], "fun_facts": ["x"]}')


def test_build_profile_writes_and_caches():
    prov = _FakeProvider()
    r1 = build_profile(_facts(), [_syn()], prov)
    assert r1.profile["narrative"]["headline"] == "Checks things."
    assert r1.profile["rule_candidates"][0]["count"] == 6
    r2 = build_profile(_facts(), [_syn()], prov)
    assert r2.cached and prov.calls == 1


# -- card / html ----------------------------------------------------------------


def _profile() -> dict:
    return build_profile(_facts(), [_syn()], _FakeProvider()).profile


def test_wrap_respects_limits():
    lines = wrap("one two three four five six seven", 9, 2)
    assert len(lines) == 2 and lines[-1].endswith("…")


def test_card_svg_is_valid_and_has_content():
    import xml.etree.ElementTree as ET
    svg = render_card_svg(_profile())
    ET.fromstring(svg)  # well-formed
    assert 'width="1080"' in svg and 'height="1350"' in svg
    assert "Verification" not in svg  # scores live in the report, not on the card
    assert "Run the tests before you say a fix works." in svg


def test_html_report_sections_and_no_external_requests():
    html = render_html([_syn()], None, None, _facts(), _profile())
    for needle in ("Your card", "Rules for your agent instructions file", "where time went",
                   "how you direct agents", "Agents compared", "Projects", "infra-lab"):
        assert needle in html
    assert "http://" not in html.replace("http://www.w3.org/2000/svg", "") and "https://fonts" not in html
    assert "<script src" not in html


def test_rules_block_is_paste_ready():
    block = rules_block(_profile())
    assert block.startswith("## ")
    assert "- Run the tests before you say a fix works." in block


def test_write_dashboard(tmp_path):
    p = write_dashboard([_syn()], None, None, _facts(), profile=_profile())
    assert p.name == "dashboard.html" and p.exists()


# -- share export ----------------------------------------------------------------


def test_share_export_removes_project_names():
    prof = _profile()
    prof["narrative"]["headline"] = "Most time went into infra-lab at /Users/alice/work/kube."
    paths = write_share(prof, ["infra-lab", "side-app"])
    page = paths["page"].read_text()
    card = paths["card"].read_text()
    assert "infra-lab" not in page and "infra-lab" not in card
    assert "/Users/alice" not in page
    assert "Project A" in page
    assert "Shareable" in page


def test_share_export_drops_project_list():
    safe = redacted_profile(_profile(), ["infra-lab", "side-app"])
    assert "top_projects" not in safe["stats"]
    assert "rule_candidates" not in safe


def test_share_export_blocks_leaks(monkeypatch):
    from agent_vibe_check.renderers import share
    monkeypatch.setattr(share, "find_leaks", lambda obj, names=(): ["fake leak"])
    with pytest.raises(ShareLeakError):
        write_share(_profile(), ["infra-lab"])


def test_glossary_headers_render_as_html_not_text():
    html = render_html([_syn()], None, None, _facts(), _profile())
    assert "&lt;abbr" not in html
    assert '<abbr title="Sessions where another agent or a script typed the prompts. Counted, never scored.">Agent-driven</abbr>' in html


def test_terms_split_history_and_card_shows_them():
    from datetime import datetime, timedelta
    from agent_vibe_check.pipeline.profile import compute_terms
    base = datetime(2026, 1, 1)
    own = []
    for i in range(60):
        f = _fact(i, friction=[Friction(type="loop", cost_messages=20 if i < 30 else 2)])
        f.timestamp = (base + timedelta(days=3 * i)).isoformat()
        own.append(f)
    terms = compute_terms(own, [], n_terms=4)
    assert len(terms) == 4
    assert sum(t["sessions"] for t in terms) == 60
    assert terms[0]["friction_share"] > terms[-1]["friction_share"]
    prof = _profile()
    prof["stats"]["terms"] = terms
    svg = render_card_svg(prof)
    assert "FRICTION PER SESSION" in svg and "less time lost" in svg


@pytest.mark.parametrize("n,step_hours,unit,min_terms", [
    (60, 72, "month", 4),      # ~6 months
    (30, 40, "day", 2),        # ~7 weeks
    (12, 6, "sessions", 2),    # 3 days, 12 sessions
])
def test_terms_adapt_to_short_histories(n, step_hours, unit, min_terms):
    from datetime import datetime, timedelta
    from agent_vibe_check.pipeline.profile import compute_terms
    own = []
    for i in range(n):
        f = _fact(i)
        f.timestamp = (datetime(2026, 1, 1) + timedelta(hours=step_hours * i)).isoformat()
        own.append(f)
    terms = compute_terms(own, [])
    assert len(terms) >= min_terms and {t["unit"] for t in terms} == {unit}
    assert sum(t["sessions"] for t in terms) == n


def test_tiny_history_has_no_terms_and_card_still_renders():
    import xml.etree.ElementTree as ET
    from agent_vibe_check.pipeline.profile import compute_terms
    own = [_fact(i) for i in range(6)]
    assert compute_terms(own, []) == []
    prof = _profile()
    prof["stats"]["terms"] = []
    ET.fromstring(render_card_svg(prof))


def test_duration_cell_units():
    from agent_vibe_check.renderers.card import _duration
    assert _duration("2025-09-01", "2026-11-07") == "15 mo"
    assert _duration("2026-10-01", "2026-10-25") == "4 wk"
    assert _duration("2026-10-01", "2026-10-09") == "9 days"



def test_type_needs_evidence():
    from agent_vibe_check.pipeline.profile import has_enough_evidence
    few = [_fact(i) for i in range(4)]
    assert not has_enough_evidence(compute_stats(few, []))
    r = build_profile(few, [], _FakeProvider())
    assert r.profile["archetype"]["key"] == "warmup"
    agents_only = [_fact(i, driver="agent") for i in range(20)]
    prov = _FakeProvider()
    r = build_profile(agents_only, [], prov)
    assert r.profile["archetype"]["key"] == "warmup" and prov.calls == 0  # no narrative about 0 own sessions


def test_loop_breaker_needs_loops():
    st = compute_stats(_facts(), [_syn()])
    f = compute_features(st, _facts())
    f["loops_seen"] = False
    scores = dict(score_archetypes(f))
    f["loops_seen"] = True
    assert dict(score_archetypes(f))["loopbreaker"] > scores["loopbreaker"]
