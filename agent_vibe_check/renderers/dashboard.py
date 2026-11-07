"""
Private HTML report: one self-contained page, read top to bottom.

Order follows what the reader needs first:
  1. The card (and buttons to save it as PNG / SVG)
  2. Rules to paste into the agent instructions file
  3. Friction: where time went
  4. Vibes: how you direct agents (sessions you typed only)
  5. Agents compared
  6. Strengths and weakness
  7. Projects (expandable rows)
  8. Rhythm

No external requests: system fonts, inline CSS/JS, data inlined.
This page shows real project names, so it is PRIVATE. The share export
(`renderers/share.py`) is the redacted version.
"""

from __future__ import annotations

import json
from datetime import datetime
from html import escape
from pathlib import Path

from ..paths import get_atlas_dir
from ..pipeline.schemas import CrossProject, ProjectSynthesis, SessionFacts, Timeline
from ..pipeline.taxonomy import CANONICAL_DIMENSIONS
from .card import SIGNAL_LABELS, level_word, render_card_svg

GLOSSARY = {
    "Friction": "A stretch of a session where progress stalled: rephrased asks, corrections, loops, reverts, tool failures.",
    "Vibes": "Six signals of how you direct an AI agent: specification, context, verification, steering, delegation, learning loop.",
    "Agent-driven": "Sessions where another agent or a script typed the prompts. Counted, never scored.",
    "Goals reached": "Shipped counts 1, partial counts 0.5, abandoned 0. Sessions with unknown outcome are left out.",
}

CSS = """
:root{--bg:#f6f7f8;--surface:#fff;--ink:#14171b;--muted:#5a636e;--rule:#d8dce1;--accent:#1f56d1;
--ok:#17803d;--warn:#a15c00;--bad:#c0262d;--gutter:clamp(16px,2.5vw,40px);
--font:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
--mono:ui-monospace,"SF Mono",Menlo,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#101317;--surface:#171b20;--ink:#e6e9ed;
--muted:#9aa3ad;--rule:#2b3139;--accent:#7ea3ff;--ok:#4cc27a;--warn:#e3a340;--bad:#ff6b70;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#101317;--surface:#171b20;--ink:#e6e9ed;--muted:#9aa3ad;--rule:#2b3139;
--accent:#7ea3ff;--ok:#4cc27a;--warn:#e3a340;--bad:#ff6b70;color-scheme:dark}
*{box-sizing:border-box}html,body{overflow-x:clip}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.55 var(--font);padding:28px var(--gutter) 64px}
a{color:var(--accent)}
.page-head{display:flex;flex-wrap:wrap;align-items:baseline;justify-content:space-between;gap:8px 24px;
border-bottom:1px solid var(--rule);padding-bottom:14px;margin-bottom:28px}
.page-head h1{margin:0;font-size:1.75rem}
.facts{display:flex;flex-wrap:wrap;gap:4px 18px;color:var(--muted);font-size:.875rem}
main{display:flex;flex-direction:column;align-items:flex-start;gap:48px}
section{display:flex;flex-direction:column;gap:14px;max-width:100%;min-width:0}
h2{margin:0;font-size:1.25rem}h3{margin:0;font-size:1rem}
p,ul,ol{margin:0;max-width:72ch}
.muted{color:var(--muted)}code{font-family:var(--mono);font-size:.875em}
.card-wrap{display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start}
.card-wrap svg{width:min(540px,100%);height:auto;border-radius:18px;display:block}
.btns{display:flex;flex-wrap:wrap;gap:8px}
button{font:inherit;font-size:.875rem;padding:7px 12px;border:1px solid var(--rule);border-radius:6px;
background:var(--surface);color:var(--ink);cursor:pointer}
button:hover{border-color:var(--accent)}
.table-wrap{width:fit-content;max-width:100%;overflow-x:auto;border:1px solid var(--rule);border-radius:6px;background:var(--surface)}
table{border-collapse:collapse;font-size:.9375rem;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:9px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
td{max-width:60ch}
th{font-size:.75rem;font-weight:600;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);background:var(--surface)}
tr:last-child td{border-bottom:0}
.num{text-align:right}
.bar{display:inline-block;height:10px;border-radius:5px;background:var(--accent);vertical-align:middle}
.bar-bg{display:inline-block;width:120px;height:10px;border-radius:5px;background:color-mix(in srgb,var(--ink) 10%,transparent);margin-right:8px;vertical-align:middle}
.st{display:inline-flex;align-items:center;gap:6px;white-space:nowrap}
.st::before{content:"";width:8px;height:8px;border-radius:50%;background:var(--muted)}
.st.ok::before{background:var(--ok)}.st.warn::before{background:var(--warn)}.st.bad::before{background:var(--bad)}
details summary{cursor:pointer}
details .detail{padding:8px 0 4px;display:flex;flex-direction:column;gap:6px;max-width:72ch}
abbr[title]{text-decoration:underline dotted;cursor:help}
pre.rules{white-space:pre-wrap;background:var(--surface);border:1px solid var(--rule);border-radius:6px;
padding:12px 14px;font-family:var(--mono);font-size:.85rem;max-width:100ch;margin:0}
.hours{display:flex;align-items:flex-end;gap:3px;height:80px}
.hours span{width:14px;background:var(--accent);border-radius:2px 2px 0 0;min-height:1px}
.hours-x{display:flex;gap:3px;font-size:.7rem;color:var(--muted)}.hours-x span{width:14px;text-align:center}
"""

JS = """
function svgBlob(){const s=document.querySelector('#card svg');return new Blob([new XMLSerializer().serializeToString(s)],{type:'image/svg+xml'})}
function dl(blob,name){const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(a.href),2000)}
function saveSvg(){dl(svgBlob(),'vibe-check-card.svg')}
function savePng(){const img=new Image();const url=URL.createObjectURL(svgBlob());img.onload=()=>{const c=document.createElement('canvas');c.width=1080;c.height=1350;
c.getContext('2d').drawImage(img,0,0,1080,1350);URL.revokeObjectURL(url);c.toBlob(b=>dl(b,'vibe-check-card.png'),'image/png')};img.src=url}
function copyRules(btn){const t=document.getElementById('rules-text').innerText;navigator.clipboard.writeText(t).then(()=>{btn.textContent='Copied';setTimeout(()=>btn.textContent='Copy rules',1500)})}
"""


def _e(s) -> str:
    return escape(str(s if s is not None else ""))


def _table(headers: list[str], rows: list[list[str]], num_cols: tuple[int, ...] = ()) -> str:
    # Headers built with _abbr() are already HTML; plain headers get escaped.
    th = "".join(
        f'<th class="{"num" if i in num_cols else ""}">{h if h.startswith("<abbr") else _e(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


def _abbr(term: str) -> str:
    return f'<abbr title="{_e(GLOSSARY[term])}">{_e(term)}</abbr>' if term in GLOSSARY else _e(term)


def _bar(score) -> str:
    if score is None:
        return '<span class="muted">not enough evidence</span>'
    return f'<span class="bar-bg"><span class="bar" style="width:{score * 1.2:.0f}px"></span></span>{score}'


def _pct(v) -> str:
    return "n/a" if v is None else f"{round(v * 100)}%"


def rules_block(profile: dict) -> str:
    rules = (profile.get("narrative") or {}).get("top_rules") or []
    lines = ["## Working with me (generated by Vibe Check)", ""]
    for r in rules:
        if isinstance(r, dict) and r.get("rule"):
            lines.append(f"- {r['rule']}")
    return "\n".join(lines)


def _outcome_cell(oc: dict) -> str:
    parts = []
    for k, cls in (("shipped", "ok"), ("partial", "warn"), ("abandoned", "bad"), ("unknown", "")):
        if oc.get(k):
            parts.append(f'<span class="st {cls}">{oc[k]} {k}</span>')
    return " ".join(parts) or '<span class="muted">-</span>'


def render_html(
    syntheses: list[ProjectSynthesis],
    timeline: Timeline | None,
    cross: CrossProject | None,
    sessions: list[SessionFacts],
    profile: dict | None,
    *,
    public: bool = False,
) -> str:
    profile = profile or {}
    stats = profile.get("stats", {})
    narr = profile.get("narrative") or {}
    arch = profile.get("archetype", {})
    gen = datetime.now().strftime("%Y-%m-%d %H:%M")
    out: list[str] = []

    out.append(
        '<header class="page-head"><h1>Vibe Check</h1><div class="facts">'
        f"<span>{stats.get('sessions', len(sessions)):,} sessions</span>"
        f"<span>{stats.get('projects', len(syntheses))} projects</span>"
        f"<span>{_e(stats.get('first', ''))} to {_e(stats.get('last', ''))}</span>"
        f"<span>{_e(', '.join(stats.get('tools_used', [])))}</span>"
        f"<span>Generated {gen}</span>"
        + ("<span>Shareable: names and paths removed</span>" if public else "<span>Private: shows project names</span>")
        + "</div></header><main>"
    )

    # 1. Card
    if profile:
        out.append('<section id="card-section"><h2>Your card</h2>')
        if narr.get("headline"):
            out.append(f"<p>{_e(narr['headline'])}</p>")
        if narr.get("archetype_reason"):
            out.append(f'<p class="muted">{_e(arch.get("name", ""))}: {_e(narr["archetype_reason"])}</p>')
        out.append(f'<div class="card-wrap"><div id="card">{render_card_svg(profile)}</div></div>')
        if public:
            out.append('<div class="btns"><button onclick="savePng()">Save PNG</button>'
                       '<button onclick="saveSvg()">Save SVG</button></div>')
        else:
            # Saving happens only from the redacted share page, never from here.
            out.append('<p class="muted">To post your card, open the <a href="share/index.html">share page</a>: '
                       'it has project names, paths and secrets removed, and buttons to save the card.</p>')
        out.append("</section>")

    # 2. Rules
    rules = narr.get("top_rules") or []
    if rules:
        out.append('<section id="rules"><h2>Rules for your agent instructions file</h2>')
        out.append("<p>Paste these into CLAUDE.md, AGENTS.md, .cursorrules or GEMINI.md. "
                   "Each one comes from friction that repeated across sessions.</p>")
        rows = [[_e(r.get("rule", "")), _e(r.get("why", "")), str(r.get("sessions", ""))]
                for r in rules if isinstance(r, dict)]
        out.append(_table(["Rule", "Why", "Sessions"], rows, num_cols=(2,)))
        out.append(f'<pre class="rules" id="rules-text">{_e(rules_block(profile))}</pre>')
        out.append('<div class="btns"><button onclick="copyRules(this)">Copy rules</button></div></section>')

    # 3. Friction
    if stats.get("friction_by_type"):
        out.append(f'<section id="friction"><h2>{_abbr("Friction")}: where time went</h2>')
        ov = stats.get("overall") or {}
        if ov.get("friction_share") is not None:
            out.append(f"<p>In a typical session you lose {round(ov['friction_share'] * 100)}% of the messages "
                       f"to friction. {round((ov.get('loop_rate') or 0) * 100)}% of sessions hit a loop.</p>")
        terms = stats.get("terms") or []
        if terms:
            rows = [[_e(t.get("label", "")), f"{t.get('sessions', 0)}",
                     "-" if t.get("friction_share") is None else f"{round(t['friction_share'] * 100)}%",
                     f"{round((t.get('loop_rate') or 0) * 100)}%", _pct(t.get("goals_reached")),
                     _e(", ".join(t.get("tools", [])))] for t in terms]
            out.append("<h3>By term</h3>" + _table(
                ["Term", "Sessions", "Lost to friction", "Hit a loop", _abbr("Goals reached"), "Main tools"],
                rows, num_cols=(1, 2, 3, 4)))
        if narr.get("biggest_time_sink"):
            out.append(f"<p>{_e(narr['biggest_time_sink'])}</p>")
        ev = stats.get("friction_events_by_type", {})
        rows = [[_e(t), str(ev.get(t, 0)), f"{c:,}"] for t, c in stats["friction_by_type"].items()]
        out.append("<h3>By type</h3>" + _table(["Type", "Events", "Messages lost"], rows, num_cols=(1, 2)))
        rows = [[_e(c), f"{n:,}"] for c, n in stats.get("friction_by_cause", {}).items()]
        out.append("<h3>By cause</h3>" + _table(["Cause", "Messages lost"], rows, num_cols=(1,)) + "</section>")

    # 4. Vibes
    if stats.get("vibes"):
        out.append(f'<section id="vibes"><h2>{_abbr("Vibes")}: how you direct agents</h2>')
        trend = stats.get("trend", {})
        rows = []
        for d in CANONICAL_DIMENSIONS:
            v = stats["vibes"].get(d.name, {})
            sc = v.get("score") if (v.get("n") or 0) >= 3 else None
            t = trend.get(d.name)
            rows.append([
                f"<strong>{_e(SIGNAL_LABELS[d.name])}</strong><br><span class='muted'>{_e(d.description)}</span>",
                _bar(sc), _e(level_word(sc)), str(v.get("n", 0)),
                "" if t is None else (f"+{t}" if t > 0 else str(t)),
            ])
        out.append(_table(["Signal", "Score", "Level", "Evidence", "Change"], rows, num_cols=(3, 4)))
        out.append(f'<p class="muted">Built from {stats.get("sessions", 0):,} sessions you typed. '
                   f'{stats.get("agent_driven_sessions", 0):,} {_abbr("Agent-driven")} sessions are left out. '
                   'Levels: beginner 20, working 50, strong 80, expert 100; a score is the average. '
                   '"Change" compares the second half of the period with the first half.</p></section>')

    # 5. Agents compared
    if stats.get("tools"):
        out.append('<section id="tools"><h2>Agents compared</h2>')
        rows = [[_e(t["tool"]), f"{t['sessions']:,}", _pct(t.get("ship_rate")),
                 str(t.get("friction_per_session", "")), "-" if t.get("vibe_score") is None else str(t["vibe_score"]),
                 _e(t.get("top_task", ""))] for t in stats["tools"]]
        out.append(_table(["Tool", "Sessions", _abbr("Goals reached"), "Friction msgs / session",
                           "Vibe score", "Most common task"], rows, num_cols=(1, 2, 3, 4)) + "</section>")

    # 6. Strengths / weakness
    if narr.get("strengths") or narr.get("weakness"):
        out.append('<section id="strengths"><h2>Strengths and weakness</h2><ul>')
        for s in narr.get("strengths") or []:
            if isinstance(s, dict):
                out.append(f"<li><strong>{_e(s.get('title'))}.</strong> {_e(s.get('detail'))}</li>")
        w = narr.get("weakness") or {}
        if isinstance(w, dict) and w.get("title"):
            out.append(f"<li><strong>Weakness: {_e(w.get('title'))}.</strong> {_e(w.get('detail'))} "
                       f"<em>Fix:</em> {_e(w.get('fix'))}</li>")
        out.append("</ul>")
        if narr.get("fun_facts"):
            out.append("<h3>Fun facts</h3><ul>" + "".join(f"<li>{_e(x)}</li>" for x in narr["fun_facts"]) + "</ul>")
        out.append("</section>")

    # 7. Projects
    if syntheses:
        out.append('<section id="projects"><h2>Projects</h2>')
        rows = []
        for s in sorted(syntheses, key=lambda x: -x.session_count):
            detail = []
            if s.dna.what_it_is:
                detail.append(f"<div>{_e(s.dna.what_it_is)}</div>")
            if s.ai_collaboration.what_worked:
                detail.append(f"<div><strong>Worked:</strong> {_e(s.ai_collaboration.what_worked)}</div>")
            if s.ai_collaboration.what_broke_down:
                detail.append(f"<div><strong>Broke down:</strong> {_e(s.ai_collaboration.what_broke_down)}</div>")
            for p in s.recurring_pitfalls[:3]:
                detail.append(f"<div><strong>Pitfall:</strong> {_e(p.pattern)}"
                              + (f" <em>Fix:</em> {_e(p.fix)}" if p.fix else "") + "</div>")
            for r in s.project_rules[:4]:
                detail.append(f"<div><strong>Project rule:</strong> {_e(r.get('rule'))}</div>")
            name = (f"<details><summary><strong>{_e(s.project_short)}</strong></summary>"
                    f"<div class='detail'>{''.join(detail)}</div></details>") if detail else f"<strong>{_e(s.project_short)}</strong>"
            rows.append([
                name, _e(s.headline), f"{s.session_count}", _e(", ".join(s.tools)),
                _outcome_cell(s.outcomes), f"{s.friction_cost:,}", _e(s.last_session[:10]),
            ])
        out.append(_table(["Project", "What it is", "Sessions", "Tools", "Outcomes", "Friction msgs", "Last"],
                          rows, num_cols=(2, 5)) + "</section>")

    # 8. Rhythm
    hours = stats.get("hours") or {}
    if hours and any(hours.values()):
        mx = max(hours.values()) or 1
        bars = "".join(f'<span style="height:{80 * hours.get(str(h), 0) / mx:.0f}px" title="{h}:00 - {hours.get(str(h), 0)} sessions"></span>'
                       for h in range(24))
        xs = "".join(f"<span>{h if h % 3 == 0 else ''}</span>" for h in range(24))
        out.append('<section id="rhythm"><h2>Rhythm</h2>'
                   f"<p>Peak hour {stats.get('peak_hour')}:00. Busiest month {_e(stats.get('busiest_month'))}. "
                   f"Most active day {_e(stats.get('top_weekday'))}. Longest streak "
                   f"{stats.get('longest_streak_days', 0)} days.</p>"
                   f'<div class="hours">{bars}</div><div class="hours-x">{xs}</div></section>')

    out.append("</main>")
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
        f"<title>Vibe Check</title><style>{CSS}</style></head><body>"
        + "".join(out)
        + f"<script>{JS}</script></body></html>"
    )


def build_payload(syntheses, timeline, cross, sessions, profile=None) -> dict:
    """Machine-readable summary embedded for tooling and tests."""
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "projects": len(syntheses),
        "sessions": len(sessions),
        "has_profile": bool(profile),
        "archetype": (profile or {}).get("archetype", {}).get("name"),
    }


def write_dashboard(
    syntheses: list[ProjectSynthesis],
    timeline: Timeline | None,
    cross: CrossProject | None,
    sessions: list[SessionFacts],
    profile: dict | None = None,
    out_dir: Path | None = None,
) -> Path:
    out_dir = out_dir or get_atlas_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "dashboard.html"
    path.write_text(render_html(syntheses, timeline, cross, sessions, profile), encoding="utf-8")
    return path


__all__ = ["render_html", "write_dashboard", "build_payload", "rules_block"]
