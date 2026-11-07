"""
Build the README images from a made-up developer. No real history is read and
no LLM is called: sessions, projects and the written parts are synthetic.

    python scripts/make_demo_assets.py            # writes docs/images/demo-card.svg
    python scripts/make_demo_assets.py --png      # also PNG screenshots (needs Chrome)

Everything is written under docs/images/ plus a temp VIBE_CHECK_HOME,
so the real ~/.vibe-check is never touched.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
os.environ["VIBE_CHECK_HOME"] = tempfile.mkdtemp(prefix="vibe-demo-")
sys.path.insert(0, str(ROOT))

from agent_vibe_check.pipeline.profile import (  # noqa: E402
    ARCHETYPE_BY_KEY, compute_features, compute_stats, own_sessions, score_archetypes,
)
from agent_vibe_check.pipeline.schemas import (  # noqa: E402
    SCHEMA_VERSION, AIFailure, Friction, Outcome, ProjectAICollaboration, ProjectDNA,
    ProjectPattern, ProjectSynthesis, SessionFacts, SkillSignal,
)
from agent_vibe_check.renderers.card import render_card_svg  # noqa: E402
from agent_vibe_check.renderers.dashboard import render_html  # noqa: E402
from agent_vibe_check.renderers.share import redacted_profile  # noqa: E402

PROJECTS = {
    "weather-api": ("Claude Code", "A small weather API with caching and rate limits."),
    "recipe-box": ("Cursor", "A recipe app with search and shopping lists."),
    "pixel-quest": ("Cursor", "A browser pixel-art adventure game."),
    "invoice-bot": ("Codex", "A bot that turns emails into invoices."),
    "dotfiles": ("Gemini CLI", "Personal shell and editor setup."),
}
SIGNALS = ["specification", "context", "verification", "steering", "delegation", "learning-loop"]


def _facts(rng: random.Random) -> list[SessionFacts]:
    start = datetime(2026, 2, 1, 9)
    out: list[SessionFacts] = []
    for i in range(312):
        progress = i / 311                      # gets better over time
        project = rng.choice(list(PROJECTS))
        tool = PROJECTS[project][0] if rng.random() < 0.8 else rng.choice(["Cursor", "Claude Code"])
        msgs = rng.randint(12, 140)
        lost = int(msgs * max(0.03, rng.gauss(0.26 - 0.19 * progress, 0.05)))
        friction = []
        if lost > 1:
            ftype = rng.choices(["correction", "loop", "env-failure", "rephrase"], [5, 3, 2, 2])[0]
            friction.append(Friction(type=ftype, cause="missing-context", cost_messages=lost))
        levels = []
        for sig in rng.sample(SIGNALS, 3):
            p_strong = 0.15 + (0.35 if sig == "verification" else 0.0) + 0.2 * progress
            lvl = "strong" if rng.random() < p_strong else ("beginner" if rng.random() < 0.15 else "working")
            levels.append(SkillSignal(sig, lvl, "med"))
        status = rng.choices(["shipped", "partial", "abandoned", "unknown"],
                             [0.35 + 0.25 * progress, 0.4, 0.08, 0.1])[0]
        out.append(SessionFacts(
            schema_version=SCHEMA_VERSION, session_id=f"demo-{i}", project_folder=project,
            project_short=project,
            timestamp=(start + timedelta(hours=i * 21 + rng.randint(0, 9))).isoformat(),
            source="demo", message_count=msgs, tool=tool, driver="human",
            task_type=rng.choice(["build-feature", "fix-bug", "refactor", "research-explore"]),
            skills_observed=levels, outcome=Outcome(status=status), friction=friction,
            ai_failures=[AIFailure("claimed tests passed", 3, user_caught_it=rng.random() < 0.7)],
        ))
    for i in range(40):  # a few sessions run by agents: counted, never scored
        out.append(SessionFacts(
            schema_version=SCHEMA_VERSION, session_id=f"crew-{i}", project_folder="weather-api",
            project_short="weather-api", timestamp=(start + timedelta(days=150 + i)).isoformat(),
            source="demo", message_count=60, tool="Claude Code", driver="agent",
            outcome=Outcome(status="shipped" if i % 3 else "partial"),
        ))
    return out


def _syntheses(facts: list[SessionFacts]) -> list[ProjectSynthesis]:
    out = []
    for name, (tool, what) in PROJECTS.items():
        fs = [f for f in facts if f.project_short == name]
        out.append(ProjectSynthesis(
            schema_version=SCHEMA_VERSION, project_short=name, project_folder=name,
            real_path=f"/home/demo/code/{name}", first_session=min(f.timestamp for f in fs),
            last_session=max(f.timestamp for f in fs), session_count=len(fs),
            message_count=sum(f.message_count for f in fs), headline=what, dna=ProjectDNA(what_it_is=what),
            tools={tool: len(fs)},
            outcomes={s: sum(1 for f in fs if f.outcome.status == s) for s in ("shipped", "partial", "abandoned")},
            friction_cost=sum(f.friction_cost for f in fs),
            recurring_pitfalls=[ProjectPattern("Starts coding before the test exists.",
                                               fix="Write the failing test first, then the fix.")],
            ai_collaboration=ProjectAICollaboration(what_worked="Small, checkable steps.",
                                                    what_broke_down="Long sessions without a reset."),
        ))
    return out


def build() -> dict:
    rng = random.Random(7)
    facts = _facts(rng)
    syns = _syntheses(facts)
    stats = compute_stats(facts, syns)
    ranked = score_archetypes(compute_features(stats, own_sessions(facts)))
    a = ARCHETYPE_BY_KEY[ranked[0][0]]  # the formula decides, as for real users
    print("type for the demo data:", a.name)
    profile = {
        "stats": stats,
        "archetype": {"key": a.key, "name": a.name, "tagline": a.tagline, "color": a.color},
        "narrative": {
            "headline": "You read what the agent wrote and catch its mistakes early.",
            "archetype_reason": "You catch most agent mistakes yourself, and verification is your strongest signal.",
            "biggest_time_sink": "Corrections after vague first asks cost the most messages.",
            "card_rules": [
                "Show the test output before you say it works",
                "Ask before you add a new dependency",
                "Keep each change small enough to review",
            ],
            "top_rules": [
                {"rule": "Run the tests and show the output before you say a fix works.", "why": "Claims of success without proof.", "sessions": 14},
                {"rule": "Ask before you add a new dependency.", "why": "Surprise packages caused reverts.", "sessions": 6},
                {"rule": "Keep each change small enough to review in one sitting.", "why": "Big diffs hid bugs.", "sessions": 5},
            ],
            "strengths": [
                {"title": "Proof first", "detail": "You ask for test output before you accept a fix."},
                {"title": "Fast course corrections", "detail": "Wrong turns rarely last more than two messages."},
                {"title": "Small steps", "detail": "Most sessions end with one reviewable change."},
            ],
            "weakness": {"title": "Vague first asks", "detail": "Many sessions start without a clear done-check.",
                         "fix": "Start with the goal and how you will check it."},
            "fun_facts": ["Your busiest day is Tuesday.", "Your longest streak is 11 days.", "You used 4 AI tools."],
        },
    }
    return {"profile": profile, "facts": facts, "syntheses": syns}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = build()
    names = list(PROJECTS)
    (OUT / "demo-card.svg").write_text(render_card_svg(redacted_profile(data["profile"], names)), encoding="utf-8")
    report = Path(os.environ["VIBE_CHECK_HOME"]) / "demo-report.html"   # temp; only its PNG ships
    report.write_text(render_html(data["syntheses"], None, None, data["facts"], data["profile"]), encoding="utf-8")
    print("wrote", OUT / "demo-card.svg")

    if "--png" in sys.argv:
        chrome = shutil.which("google-chrome") or shutil.which("chromium") or \
            "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        shots = [(OUT / "demo-card.svg", "demo-card.png", "1080,1350"), (report, "demo-report.png", "1300,2400")]
        for src, dst, size in shots:
            subprocess.run([chrome, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            f"--window-size={size}", f"--screenshot={OUT / dst}", src.as_uri()],
                           check=True, capture_output=True)
            print("wrote", OUT / dst)


if __name__ == "__main__":
    main()
