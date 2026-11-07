"""
Markdown renderers for the four pipeline stages.

Function-per-output so the TUI / CLI can pick and choose:

  render_project_md(synthesis)       -> str
  render_timeline_md(timeline)       -> str
  render_cross_project_md(cross)     -> str
  render_executive_summary_md(...)   -> str  (takes all three)

File writers are in the submodule's only side-effect helpers:

  write_project_reports(short, syn)          -> {name: Path}
  write_global_reports(timeline, cross, ...) -> {name: Path}

No emojis (project-wide rule).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

from ..paths import get_atlas_dir, get_extractions_dir
from ..pipeline.schemas import (
    CrossProject,
    ProjectSynthesis,
    SessionFacts,
    Timeline,
    TimelinePeriod,
)
from ..pipeline.taxonomy import CANONICAL_DIMENSIONS, DIMENSION_BY_NAME, LEVEL_RANK


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _h(text: str, level: int = 1) -> str:
    return "#" * level + " " + text


def _bullet(text: str) -> str:
    return f"- {text}"


def _code(text: str) -> str:
    return f"`{text}`"


def _nonempty(lines: list[str]) -> list[str]:
    return [l for l in lines if l.strip()]


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    sep = "|".join(["---"] * len(headers))
    out = ["| " + " | ".join(headers) + " |", f"|{sep}|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Project report
# ---------------------------------------------------------------------------


def render_project_md(syn: ProjectSynthesis) -> str:
    lines: list[str] = [_h(f"Project: {syn.project_short}")]
    if syn.headline:
        lines.append(f"> {syn.headline}\n")
    meta = [
        f"- Sessions: {syn.session_count} ({syn.message_count:,} messages)",
        f"- Active: {syn.first_session[:10]} to {syn.last_session[:10]}",
    ]
    if syn.real_path:
        meta.append(f"- Path: `{syn.real_path}`")
    if syn.tools:
        meta.append("- Tools: " + ", ".join(f"{k} ({v})" for k, v in syn.tools.items()))
    if syn.outcomes:
        meta.append("- Outcomes: " + ", ".join(f"{v} {k}" for k, v in syn.outcomes.items()))
    if syn.drivers:
        meta.append("- Driven by: " + ", ".join(f"{k} ({v})" for k, v in syn.drivers.items()))
    meta.append(f"- Friction: {syn.friction_cost:,} messages")
    lines.append("\n".join(meta) + "\n")

    if syn.dna.what_it_is:
        lines += [_h("What it is", 2), syn.dna.what_it_is + "\n"]
        if syn.dna.key_components:
            lines.append("\n".join(_bullet(x) for x in syn.dna.key_components) + "\n")

    if syn.skill_profile:
        rows = []
        for d in CANONICAL_DIMENSIONS:
            e = syn.skill_profile.get(d.name)
            if e:
                rows.append([f"`{d.name}`", e.current_level, e.peak_level, e.trajectory,
                             e.confidence, str(len(e.evidence_sessions))])
        if rows:
            lines += [_h("Vibes on this project", 2),
                      _table(["signal", "current", "peak", "trajectory", "confidence", "sessions"], rows)]

    if syn.friction:
        rows = [[f["type"], f["cause"], str(f["count"]), str(f["cost_messages"]),
                 (f.get("examples") or [""])[0][:120]] for f in syn.friction[:8]]
        lines += [_h("Friction", 2), _table(["type", "cause", "events", "messages lost", "example"], rows)]

    if syn.recurring_strengths:
        lines += [_h("What works", 2)] + [_bullet(p.pattern) for p in syn.recurring_strengths] + [""]
    if syn.recurring_pitfalls:
        lines.append(_h("What keeps going wrong", 2))
        for p in syn.recurring_pitfalls:
            lines.append(_bullet(p.pattern + (f" Fix: {p.fix}" if p.fix else "")))
        lines.append("")
    if syn.project_rules:
        lines += [_h("Rules for this project", 2)]
        lines += [_bullet(f"{r.get('rule')} ({r.get('count', 1)} sessions)") for r in syn.project_rules] + [""]

    ac = syn.ai_collaboration
    if ac.what_worked or ac.what_broke_down:
        lines.append(_h("Working with the agent", 2))
        if ac.what_worked:
            lines.append(_bullet(f"Worked: {ac.what_worked}"))
        if ac.what_broke_down:
            lines.append(_bullet(f"Broke down: {ac.what_broke_down}"))
        if ac.typical_failure_mode:
            lines.append(_bullet(f"Typical failure: {ac.typical_failure_mode}"))
        lines.append("")

    if syn.arc.phases:
        lines.append(_h("Arc", 2))
        for ph in syn.arc.phases:
            lines.append(_bullet(f"{ph.start} to {ph.end}: **{ph.name}**. {ph.focus}"))
        for pv in syn.arc.pivots:
            lines.append(_bullet(f"Pivot {pv.when}: {pv.from_} -> {pv.to}. {pv.why}"))
        lines.append("")

    if syn.key_decisions:
        lines.append(_h("Key decisions", 2))
        for d in syn.key_decisions[:12]:
            lines.append(_bullet(d.decision + (f" ({d.rationale})" if d.rationale else "")))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _format_period(p: TimelinePeriod, *, compact: bool = False) -> list[str]:
    out: list[str] = []
    out.append(_h(p.period, 3))
    out.append(f"_{p.sessions} sessions_  ")
    if p.active_projects:
        top = ", ".join(f"{a.name} ({a.sessions})" for a in p.active_projects[:6])
        out.append(f"**Active projects:** {top}")
    if p.top_topics:
        out.append(f"**Top topics:** {', '.join(p.top_topics)}")
    if p.deltas_vs_prev:
        out.append("**Changes vs previous period:**")
        for dim, delta in sorted(p.deltas_vs_prev.items()):
            out.append(f"- `{dim}`: {delta}")
    if not compact and p.skill_snapshot:
        out.append("**Skill snapshot:**")
        rows = []
        for dim, snap in sorted(p.skill_snapshot.items()):
            rows.append([_code(dim), snap.level, snap.confidence])
        out.append(_table(["dimension", "level", "confidence"], rows))
    out.append("")
    return out


def render_timeline_md(tl: Timeline) -> str:
    lines: list[str] = []
    lines.append(_h("Timeline"))
    lines.append("")
    lines.append(f"> Generated {tl.generated_at}")
    lines.append("")

    if tl.narrative.strongest_growth:
        lines.append(_h("Strongest growth", 2))
        lines.extend(_bullet(x) for x in tl.narrative.strongest_growth)
        lines.append("")
    if tl.narrative.surprising_moves:
        lines.append(_h("Surprising moves", 2))
        lines.extend(_bullet(x) for x in tl.narrative.surprising_moves)
        lines.append("")
    if tl.narrative.emerging_interests:
        lines.append(_h("Emerging interests", 2))
        lines.extend(_bullet(x) for x in tl.narrative.emerging_interests)
        lines.append("")

    if tl.periods_quarterly:
        lines.append(_h("Quarterly", 2))
        for p in tl.periods_quarterly:
            lines.extend(_format_period(p))
    if tl.periods_monthly:
        lines.append(_h("Monthly", 2))
        for p in tl.periods_monthly:
            lines.extend(_format_period(p, compact=True))

    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Cross-project report
# ---------------------------------------------------------------------------


def render_cross_project_md(cross: CrossProject) -> str:
    lines: list[str] = []
    lines.append(_h("Cross-project comparison"))
    lines.append("")
    lines.append(f"> Generated {cross.generated_at}")
    lines.append("")

    if cross.anomalies:
        lines.append(_h("Anomalies worth noticing", 2))
        for a in cross.anomalies:
            lines.append(f"- **{a.observation}**")
            if a.dimensions_involved:
                lines.append(
                    f"  _dimensions:_ {', '.join(_code(d) for d in a.dimensions_involved)}"
                )
            if a.evidence:
                ev = ", ".join(
                    f"{e.get('project', '?')}={e.get('level', '?')}" for e in a.evidence
                )
                lines.append(f"  _evidence:_ {ev}")
            if a.hypothesis:
                lines.append(f"  _hypothesis:_ {a.hypothesis}")
        lines.append("")

    # Collect project shorts across all rows (for consistent column ordering)
    projects: list[str] = sorted({p for row in cross.dimension_matrix.values() for p in row})

    # Matrix
    lines.append(_h("Dimension matrix", 2))
    if not projects or not cross.dimension_matrix:
        lines.append("_no cross-project data yet_")
        lines.append("")
    else:
        # Group by taxonomy group so the matrix is readable
        by_group: dict[str, list[str]] = {}
        for d in CANONICAL_DIMENSIONS:
            if d.name in cross.dimension_matrix:
                by_group.setdefault(d.group, []).append(d.name)

        for group, dims in by_group.items():
            lines.append(_h(group, 3))
            headers = ["dimension"] + projects
            rows: list[list[str]] = []
            for dim in dims:
                row = [_code(dim)]
                for proj in projects:
                    cell = cross.dimension_matrix.get(dim, {}).get(proj)
                    if cell is None:
                        row.append("—")
                    else:
                        row.append(f"{cell.level} ({cell.sessions})")
                rows.append(row)
            lines.append(_table(headers, rows))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Executive summary
# ---------------------------------------------------------------------------


def render_profile_md(profile: dict | None, syntheses: Iterable[ProjectSynthesis] = ()) -> str:
    profile = profile or {}
    st = profile.get("stats", {})
    narr = profile.get("narrative") or {}
    arch = profile.get("archetype", {})
    lines = [_h("Vibe Check profile")]
    lines.append(
        f"> {st.get('sessions', 0):,} sessions | {st.get('projects', 0)} projects | "
        f"{', '.join(st.get('tools_used', []))} | {st.get('first', '')} to {st.get('last', '')}\n"
    )
    if arch:
        lines += [_h(arch.get("name", ""), 2), arch.get("tagline", "") + "\n"]
        if narr.get("archetype_reason"):
            lines.append(narr["archetype_reason"] + "\n")
    if narr.get("headline"):
        lines.append(f"**{narr['headline']}**\n")

    vibes = st.get("vibes") or {}
    if vibes:
        rows = []
        for d in CANONICAL_DIMENSIONS:
            v = vibes.get(d.name, {})
            sc = v.get("score") if (v.get("n") or 0) >= 3 else None
            rows.append([f"`{d.name}`", "-" if sc is None else str(sc), str(v.get("n", 0))])
        lines += [_h("How you direct agents (hands-on sessions)", 2), _table(["signal", "score", "evidence"], rows)]

    if st.get("friction_by_type"):
        lines.append(_h("Friction", 2))
        lines.append(f"{st.get('friction_messages', 0):,} messages lost "
                     f"({round(st.get('friction_share', 0) * 100)}% of all).\n")
        if narr.get("biggest_time_sink"):
            lines.append(narr["biggest_time_sink"] + "\n")
        rows = [[k, str(v)] for k, v in st["friction_by_type"].items()]
        lines.append(_table(["type", "messages lost"], rows))

    if st.get("tools"):
        rows = [[t["tool"], str(t["sessions"]),
                 "n/a" if t.get("ship_rate") is None else f"{round(t['ship_rate'] * 100)}%",
                 str(t.get("friction_per_session", ""))] for t in st["tools"]]
        lines += [_h("Agents compared", 2), _table(["tool", "sessions", "goals reached", "friction / session"], rows)]

    if narr.get("strengths") or narr.get("weakness"):
        lines.append(_h("Strengths and weakness", 2))
        for s in narr.get("strengths") or []:
            if isinstance(s, dict):
                lines.append(_bullet(f"**{s.get('title')}.** {s.get('detail')}"))
        w = narr.get("weakness") or {}
        if isinstance(w, dict) and w.get("title"):
            lines.append(_bullet(f"**Weakness: {w.get('title')}.** {w.get('detail')} Fix: {w.get('fix')}"))
        lines.append("")

    rules = narr.get("top_rules") or []
    if rules:
        lines.append(_h("Rules for your agent instructions file", 2))
        lines += [_bullet(f"{r.get('rule')} ({r.get('sessions', '?')} sessions)") for r in rules if isinstance(r, dict)]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_executive_summary_md(timeline, cross, syntheses, sessions=None, profile: dict | None = None) -> str:
    """Kept for callers of v1; v2 content is the profile."""
    return render_profile_md(profile, syntheses)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def write_project_reports(project_short: str, syn: ProjectSynthesis) -> dict[str, Path]:
    root = get_extractions_dir() / project_short
    root.mkdir(parents=True, exist_ok=True)
    out_path = root / "PROJECT.md"
    out_path.write_text(render_project_md(syn), encoding="utf-8")
    return {"project": out_path}


def write_global_reports(
    timeline: Timeline | None,
    cross: CrossProject | None,
    syntheses: Iterable[ProjectSynthesis],
    sessions: Iterable[SessionFacts] | None = None,
    profile: dict | None = None,
) -> dict[str, Path]:
    out: dict[str, Path] = {}
    atlas = get_atlas_dir()
    if timeline is not None:
        p = atlas / "timeline.md"
        p.write_text(render_timeline_md(timeline), encoding="utf-8")
        out["timeline"] = p
    if cross is not None:
        p = atlas / "cross_project.md"
        p.write_text(render_cross_project_md(cross), encoding="utf-8")
        out["cross_project"] = p
    profile_path = atlas / "PROFILE.md"
    profile_path.write_text(render_profile_md(profile, syntheses), encoding="utf-8")
    out["profile"] = profile_path
    if profile and (profile.get("narrative") or {}).get("top_rules"):
        from .dashboard import rules_block
        rp = atlas / "RULES.md"
        rp.write_text(rules_block(profile) + "\n", encoding="utf-8")
        out["rules"] = rp
    return out


__all__ = [
    "render_project_md",
    "render_timeline_md",
    "render_cross_project_md",
    "render_executive_summary_md",
    "render_profile_md",
    "write_project_reports",
    "write_global_reports",
]
