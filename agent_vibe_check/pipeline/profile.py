"""
Stage 5: portfolio profile ("Wrapped").

Input:  every SessionFacts + every ProjectSynthesis
Output: atlas/profile.json

Mostly local and deterministic:
  - Vibe scores per signal. Only sessions where the person typed the
    prompts count (driver == "human"). Sessions another agent drove are
    the agents' work, not the person's: they are counted, never scored.
  - Outcomes, friction totals by type and cause, agent-tool comparison,
    work rhythm, most improved signal.
  - Archetype: a fixed set scored from those features, so two users are
    comparable and the pick is explainable.

One LLM call writes the human part: summary, 3 strengths, 1 weakness, and
the merged list of general rules to paste into an agent instructions file.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..paths import get_atlas_dir
from ..providers import LLMProvider
from .llm_utils import atomic_write_json, call_with_retries, parse_json_lenient
from .project_synthesizer import gather_rules
from .schemas import ProjectSynthesis, SessionFacts
from .taxonomy import CANONICAL_DIMENSIONS

PROFILE_VERSION = 5  # v5: 12 types, relative vibe features
_CONF_WEIGHT = {"low": 0.5, "med": 1.0, "high": 1.5}
# Points per level. "working" is competent and normal, so it sits at 50,
# not at the bottom third of the scale.
LEVEL_POINTS = {"beginner": 20, "working": 50, "strong": 80, "expert": 100}
MIN_OBS_FOR_SCORE = 3


# ---------------------------------------------------------------------------
# Archetypes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Archetype:
    key: str
    name: str
    tagline: str
    color: str  # accent used by the card


# Names are meant to be shared: short, a little playful, still accurate.
# Each type has a fixed formula over measured numbers (see score_archetypes),
# so a card can only show a type the data supports.
# Taglines must fit ONE line on the card.
ARCHETYPES: tuple[Archetype, ...] = (
    Archetype("blueprint", "Blueprint Brain", "Defines done before the first line of code.", "#1c4fb8"),
    Archetype("receipts", "Receipt Keeper", "Wants proof before anything is merged.", "#0b7a5c"),
    Archetype("loopbreaker", "Loop Breaker", "Sees the agent circling and stops it.", "#b3261e"),
    Archetype("speedrunner", "Speedrunner", "Many short sessions, shipped fast.", "#e07a00"),
    Archetype("nightshift", "Night Shift Pilot", "Lets long runs go on while away.", "#3b3f8f"),
    Archetype("hoarder", "Context Hoarder", "Gives the agent every file it needs.", "#007c8f"),
    Archetype("diffhawk", "Diff Hawk", "Reads every line the agent writes.", "#5a3e8a"),
    Archetype("surfer", "Vibe Surfer", "Follows the flow, fixes it later.", "#00838f"),
    Archetype("guardrail", "Guardrail Builder", "Writes a rule after each mistake.", "#5f6b1e"),
    Archetype("sculptor", "Prompt Sculptor", "Shapes the ask until the first try works.", "#8a5a00"),
    Archetype("detective", "Bug Detective", "Starts each session from a stack trace.", "#6d4c41"),
    Archetype("sandbox", "Sandbox Scientist", "Tests on a copy before production.", "#2e7d32"),
)
# Shown when there is not enough evidence for any type. Never scored.
WARMUP = Archetype("warmup", "Warming Up", "Not enough sessions yet for a type.", "#1b4b8f")
MIN_OWN_SESSIONS_FOR_TYPE = 10
MIN_SIGNALS_FOR_TYPE = 2

ARCHETYPE_BY_KEY = {a.key: a for a in ARCHETYPES + (WARMUP,)}


def has_enough_evidence(stats: dict) -> bool:
    signals = sum(1 for v in stats.get("vibes", {}).values() if (v.get("n") or 0) >= MIN_OBS_FOR_SCORE)
    return stats.get("sessions", 0) >= MIN_OWN_SESSIONS_FOR_TYPE and signals >= MIN_SIGNALS_FOR_TYPE


def score_archetypes(feat: dict) -> list[tuple[str, float]]:
    """
    Score every type from measured features. Highest first.

    Vibe signals enter RELATIVE to the person's own average (rel), because
    most people sit near "working" on every signal: the type reflects what
    stands out for this person, not an absolute grade.
    """
    rel = feat["vibes_rel"]
    s = {
        "blueprint": rel["specification"] + 0.5 * rel["delegation"] + 0.2,
        "sculptor": rel["specification"] + 0.5 * rel["context"] - 0.3 * feat["correction_share"],
        "receipts": rel["verification"] + 0.4 * feat["ship_rate"],
        "diffhawk": 0.8 * feat["caught_rate"] + 0.5 * rel["verification"],
        # Breaking loops only counts when loops were actually seen.
        "loopbreaker": rel["steering"] + (0.6 * (1 - feat["loop_rate"]) - 0.2 if feat["loops_seen"] else -0.3),
        "hoarder": rel["context"] + 0.3 * rel["specification"] + 0.1,
        "guardrail": 1.3 * rel["learning-loop"] + 0.1,
        "speedrunner": feat["short_session_share"] + 0.5 * feat["ship_rate"] - 0.3 * rel["specification"] - 0.6,
        "nightshift": 1.4 * feat["agent_share"] + 0.6 * feat["night_share"],
        "surfer": -rel["specification"] + 0.5 * feat["correction_share"] + 0.2 * feat["ship_rate"] - 0.2,
        "detective": 1.3 * feat["bugfix_share"] + 0.3 * rel["steering"],
        "sandbox": 1.1 * feat["infra_data_share"] + 0.5 * rel["verification"],
    }
    return sorted(s.items(), key=lambda kv: -kv[1])


# ---------------------------------------------------------------------------
# Local aggregation
# ---------------------------------------------------------------------------


def _ts(f: SessionFacts) -> datetime | None:
    try:
        return datetime.fromisoformat(f.timestamp)
    except (TypeError, ValueError):
        return None


def vibe_scores(facts: list[SessionFacts]) -> dict[str, dict]:
    """Per signal: score 0-100 (confidence-weighted mean level) + observation count."""
    acc: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for f in facts:
        for s in f.skills_observed:
            pts = LEVEL_POINTS.get(s.level)
            if pts is None:
                continue
            acc[s.dimension].append((pts, _CONF_WEIGHT.get(s.confidence, 0.5)))
    out: dict[str, dict] = {}
    for d in CANONICAL_DIMENSIONS:
        obs = acc.get(d.name, [])
        if not obs:
            out[d.name] = {"score": None, "n": 0}
            continue
        mean = sum(p * w for p, w in obs) / sum(w for _, w in obs)
        out[d.name] = {"score": round(mean), "n": len(obs)}
    return out


def _trend(facts: list[SessionFacts]) -> dict[str, int | None]:
    """Score change per signal between the first and second half of the period."""
    dated = sorted((f for f in facts if _ts(f)), key=_ts)
    if len(dated) < 10:
        return {}
    mid = len(dated) // 2
    a, b = vibe_scores(dated[:mid]), vibe_scores(dated[mid:])
    out = {}
    for k in a:
        if a[k]["score"] is not None and b[k]["score"] is not None and min(a[k]["n"], b[k]["n"]) >= MIN_OBS_FOR_SCORE:
            out[k] = b[k]["score"] - a[k]["score"]
    return out


def _longest_streak(days: set) -> int:
    best = cur = 0
    prev = None
    for d in sorted(days):
        cur = cur + 1 if prev is not None and d - prev == timedelta(days=1) else 1
        best = max(best, cur)
        prev = d
    return best


def _resolved_rate(outcomes: Counter) -> float | None:
    known = outcomes["shipped"] + outcomes["partial"] + outcomes["abandoned"]
    if not known:
        return None
    return (outcomes["shipped"] + 0.5 * outcomes["partial"]) / known


def tool_comparison(facts: list[SessionFacts]) -> list[dict]:
    by_tool: dict[str, list[SessionFacts]] = defaultdict(list)
    for f in facts:
        by_tool[f.tool or f.source].append(f)
    rows = []
    for tool, fs in by_tool.items():
        oc = Counter(f.outcome.status for f in fs)
        scores = [v["score"] for v in vibe_scores(fs).values() if v["score"] is not None]
        task = Counter(f.task_type for f in fs).most_common(1)
        rows.append({
            "tool": tool,
            "sessions": len(fs),
            "ship_rate": _resolved_rate(oc),
            "friction_per_session": round(sum(f.friction_cost for f in fs) / len(fs), 1),
            "vibe_score": round(sum(scores) / len(scores)) if scores else None,
            "top_task": task[0][0] if task else "",
            "outcomes": dict(oc),
        })
    return sorted(rows, key=lambda r: -r["sessions"])


def _session_friction_share(f: SessionFacts) -> float:
    """Share of one session's messages lost to friction (0..1)."""
    return min(1.0, f.friction_cost / max(1, f.message_count))


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    m = len(ys) // 2
    return ys[m] if len(ys) % 2 else (ys[m - 1] + ys[m]) / 2


def _period_numbers(own: list[SessionFacts], crew: list[SessionFacts]) -> dict:
    oc = Counter(f.outcome.status for f in own)
    crew_oc = Counter(f.outcome.status for f in crew)
    tools = Counter(f.tool or f.source for f in own)
    n = max(1, len(own))
    return {
        "sessions": len(own),
        "goals_reached": _resolved_rate(oc),
        # Medians of per-session shares: long agentic sessions (many tool
        # messages) do not drown out short chats.
        "friction_share": _median([_session_friction_share(f) for f in own]),
        "loop_rate": sum(1 for f in own if any(x.type == "loop" for x in f.friction)) / n,
        "tools": [t for t, c in tools.most_common() if c / n >= 0.2][:2],
        "crew_sessions": len(crew),
        "crew_goals_reached": _resolved_rate(crew_oc),
    }


def compute_terms(own: list[SessionFacts], crew: list[SessionFacts], n_terms: int = 4) -> list[dict]:
    """
    Split the person's history into "terms" so the card shows change, not one
    flat average. The split adapts to how much history there is:

      >= 90 days and >= 40 sessions  -> up to 4 terms of equal time (month labels)
      >= 28 days and >= 20 sessions  -> 2-4 terms of equal time (day labels)
      >= 10 sessions                 -> 2-3 groups by session order ("FIRST 8")
      fewer                          -> no terms (the card shows one number)

    Each term keeps >= 5 sessions, so one odd session cannot make a bar.
    """
    dated = sorted(((d, f) for f in own if (d := _ts(f))), key=lambda t: t[0])
    crew_dated = [(d, f) for f in crew if (d := _ts(f))]
    n = len(dated)
    if n < 10:
        return []
    t0, t1 = dated[0][0], dated[-1][0]
    days = (t1 - t0).days

    def _row(own_i, crew_i, label, a, b, unit):
        row = _period_numbers(own_i, crew_i)
        row.update({"start": a.date().isoformat(), "end": b.date().isoformat(), "label": label, "unit": unit})
        return row

    if days >= 28 and n >= 20:
        unit = "month" if days >= 90 and n >= 40 else "day"
        k = max(2, min(n_terms, n // 5, days // (30 if unit == "month" else 7)))
        span = (t1 - t0) / k
        terms = []
        for i in range(k):
            a, b = t0 + span * i, (t0 + span * (i + 1)) if i < k - 1 else t1
            own_i = [f for d, f in dated if (a <= d <= b) and (i == 0 or d > a)]
            crew_i = [f for d, f in crew_dated if (a <= d <= b) and (i == 0 or d > a)]
            if unit == "month":
                label = f"{a.strftime('%b')}\u2013{b.strftime('%b')}"
            elif a.month == b.month:
                label = f"{a.strftime('%b')} {a.day}\u2013{b.day}"
            else:
                label = f"{a.strftime('%b')} {a.day}\u2013{b.strftime('%b')} {b.day}"
            terms.append(_row(own_i, crew_i, label, a, b, unit))
        # Merge thin terms into their neighbour rather than show a 2-session bar.
        if all(t["sessions"] >= 5 for t in terms):
            return terms

    # Short history: split by order of sessions.
    k = 3 if n >= 15 else 2
    size = n // k
    terms = []
    for i in range(k):
        chunk = dated[i * size:(i + 1) * size] if i < k - 1 else dated[i * size:]
        a, b = chunk[0][0], chunk[-1][0]
        crew_i = [f for d, f in crew_dated if a <= d <= b]
        label = ("FIRST" if i == 0 else "LAST" if i == k - 1 else "NEXT") + f" {len(chunk)}"
        terms.append(_row([f for _, f in chunk], crew_i, label, a, b, "sessions"))
    return terms


def own_sessions(facts: list[SessionFacts]) -> list[SessionFacts]:
    """Sessions the person typed. Everything in the profile is built from these."""
    return [f for f in facts if f.driver == "human"]


def compute_stats(all_facts: list[SessionFacts], syntheses: list[ProjectSynthesis]) -> dict:
    facts = own_sessions(all_facts)
    outcomes = Counter(f.outcome.status for f in facts)
    friction_by_type: Counter = Counter()
    friction_cost_by_type: Counter = Counter()
    friction_by_cause: Counter = Counter()
    for f in facts:
        for x in f.friction:
            friction_by_type[x.type] += 1
            friction_cost_by_type[x.type] += x.cost_messages
            friction_by_cause[x.cause] += x.cost_messages
    total_friction = sum(friction_cost_by_type.values())
    total_messages = sum(f.message_count for f in facts)
    tasks = Counter(f.task_type for f in facts)
    dts = [d for d in (_ts(f) for f in facts) if d]
    hours = Counter(d.hour for d in dts)
    weekdays = Counter(d.strftime("%A") for d in dts)
    months = Counter(d.strftime("%Y-%m") for d in dts)
    projects = Counter(f.project_short for f in facts)

    return {
        "sessions": len(facts),
        "messages": total_messages,
        "projects": len(projects),
        "tools_used": sorted({f.tool or f.source for f in facts}),
        "first": min(dts).date().isoformat() if dts else "",
        "last": max(dts).date().isoformat() if dts else "",
        "active_days": len({d.date() for d in dts}),
        "longest_streak_days": _longest_streak({d.date() for d in dts}),
        "busiest_month": months.most_common(1)[0][0] if months else "",
        "peak_hour": hours.most_common(1)[0][0] if hours else None,
        "hours": {str(h): hours.get(h, 0) for h in range(24)},
        "top_weekday": weekdays.most_common(1)[0][0] if weekdays else "",
        "outcomes": dict(outcomes),
        "ship_rate": _resolved_rate(outcomes),
        "reviewed_sessions": len(all_facts),
        "agent_driven_sessions": len(all_facts) - len(facts),
        "overall": _period_numbers(facts, [f for f in all_facts if f.driver != "human"]),
        "terms": compute_terms(facts, [f for f in all_facts if f.driver != "human"]),
        "vibes": vibe_scores(facts),
        "trend": _trend(facts),
        "friction_messages": total_friction,
        "friction_share": round(total_friction / total_messages, 3) if total_messages else 0,
        "friction_by_type": dict(friction_cost_by_type.most_common()),
        "friction_events_by_type": dict(friction_by_type.most_common()),
        "friction_by_cause": dict(friction_by_cause.most_common()),
        "task_types": dict(tasks.most_common()),
        "tools": tool_comparison(facts),
        "sessions_with_commits_after": sum(1 for f in facts if f.outcome.git_commits_after > 0),
        "top_projects": [{"project": p, "sessions": n} for p, n in projects.most_common(8)],
    }


def _known_outcomes(stats: dict) -> int:
    oc = stats.get("outcomes", {})
    return oc.get("shipped", 0) + oc.get("partial", 0) + oc.get("abandoned", 0)


def compute_features(stats: dict, facts: list[SessionFacts]) -> dict:
    n = max(1, stats["sessions"])
    scored = {k: v["score"] for k, v in stats["vibes"].items()
              if v["score"] is not None and v["n"] >= MIN_OBS_FOR_SCORE}
    mean = sum(scored.values()) / len(scored) if scored else 50
    # Distance from the person's own average, in "levels" (30 points = one level).
    vibes_rel = {k: ((scored[k] - mean) / 30 if k in scored else 0.0) for k in stats["vibes"]}
    tasks = stats["task_types"]
    fr = stats["friction_events_by_type"]
    fr_total = max(1, sum(fr.values()))
    failures = [a for f in facts for a in f.ai_failures]
    night = sum(1 for f in facts if (d := _ts(f)) and (d.hour >= 22 or d.hour < 6))
    return {
        "vibes_rel": vibes_rel,
        "agent_share": stats["agent_driven_sessions"] / max(1, stats["reviewed_sessions"]),
        # Outcomes count only with enough known results (unknown is common).
        "ship_rate": (stats["ship_rate"] or 0) if _known_outcomes(stats) >= 5 else 0,
        "loops_seen": stats.get("friction_events_by_type", {}).get("loop", 0) > 0,
        "short_session_share": sum(1 for f in facts if f.message_count < 30) / n,
        "loop_rate": (stats.get("overall") or {}).get("loop_rate") or 0,
        "correction_share": (fr.get("correction", 0) + fr.get("rephrase", 0)) / fr_total,
        "caught_rate": (sum(1 for a in failures if a.user_caught_it) / len(failures)) if len(failures) >= 5 else 0,
        "night_share": night / n,
        "bugfix_share": tasks.get("fix-bug", 0) / n,
        "infra_data_share": (tasks.get("setup-infra", 0) + tasks.get("data-ml", 0) + tasks.get("ops-automation", 0)) / n,
    }


# ---------------------------------------------------------------------------
# LLM narrative
# ---------------------------------------------------------------------------


STAGE5_PROMPT = """You are writing the summary for a person's "AI coding Wrapped": a
year-in-review of how they work with AI coding agents (Cursor, Claude Code,
Codex, Gemini CLI). The numbers below come from {sessions} sessions where the
person typed the prompts. Sessions that other agents drove were left out:
never describe agent-to-agent work as the person's. Write for the person:
direct, specific, warm but honest. No hype.

The archetype was picked by formula: {archetype_name} ("{archetype_tagline}").
Runner-up: {runner_up}.

# Rules

1. Base every claim on the data below. No invented numbers.
2. Never name projects, companies, people, hosts or URLs. Say "a Kubernetes
   project" or "an image-generation app" instead.
3. `strengths`: exactly 3. `weakness`: exactly 1, with a concrete fix.
4. `top_rules`: merge the candidate rules into at most 8 distinct rules for
   the person's agent instructions file (CLAUDE.md / AGENTS.md). One
   imperative sentence each, addressed to the agent. Prefer rules backed by
   more sessions. Keep `sessions` as the sum of the merged counts.
5. `fun_facts`: 3 short, true, shareable lines from the stats.
6. `archetype_reason`: one sentence on why this archetype fits, using the data.

# Output

Return ONE JSON object:
{{
  "headline": "one sentence that captures how this person works with AI",
  "archetype_reason": "...",
  "strengths": [{{"title": "2-5 words", "detail": "one sentence with evidence"}}],
  "weakness": {{"title": "2-5 words", "detail": "one sentence with evidence", "fix": "one sentence"}},
  "biggest_time_sink": "one sentence on the friction that cost the most",
  "top_rules": [{{"rule": "...", "why": "short reason", "sessions": 4}}],
  "card_rules": ["the 3 most valuable rules, each at most 55 characters, imperative, understandable by any developer (no tool flags or internal names)"],
  "fun_facts": ["...", "...", "..."]
}}

# Stats

{stats}

# Recurring patterns from project reviews

{patterns}

# Candidate rules (general scope, with how many sessions suggested each)

{rules}
"""


def _fingerprint(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


def _patterns_payload(syntheses: list[ProjectSynthesis], limit: int = 30) -> list[dict]:
    rows = []
    for s in sorted(syntheses, key=lambda x: -x.session_count):
        for p in s.recurring_strengths[:2]:
            rows.append({"kind": "strength", "pattern": p.pattern})
        for p in s.recurring_pitfalls[:2]:
            rows.append({"kind": "pitfall", "pattern": p.pattern, "fix": p.fix})
    return rows[:limit]


@dataclass
class ProfileResult:
    profile: dict = field(default_factory=dict)
    cached: bool = False
    error: str | None = None


def build_profile(
    facts: list[SessionFacts],
    syntheses: list[ProjectSynthesis],
    provider: LLMProvider | None,
    *,
    model: str | None = None,
    force: bool = False,
) -> ProfileResult:
    out_path = get_atlas_dir() / "profile.json"
    stats = compute_stats(facts, syntheses)
    own = own_sessions(facts)
    features = compute_features(stats, own)
    ranked = score_archetypes(features)
    if has_enough_evidence(stats):
        arch = ARCHETYPE_BY_KEY[ranked[0][0]]
        runner = ARCHETYPE_BY_KEY[ranked[1][0]]
    else:
        arch = runner = WARMUP
    rules = gather_rules(own, "general", limit=40)
    patterns = _patterns_payload(syntheses)
    fp = _fingerprint(PROFILE_VERSION, stats, [r["rule"] for r in rules], patterns)

    profile = {
        "profile_version": PROFILE_VERSION,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "stats": stats,
        "features": features,
        "archetype": {"key": arch.key, "name": arch.name, "tagline": arch.tagline, "color": arch.color},
        "runner_up": {"key": runner.key, "name": runner.name},
        "archetype_scores": {k: round(v, 3) for k, v in ranked},
        "rule_candidates": rules,
        "narrative": {},
        "_input_fingerprint": fp,
    }

    if not force and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("_input_fingerprint") == fp and existing.get("_llm_done"):
                return ProfileResult(profile=existing, cached=True)
        except (OSError, json.JSONDecodeError):
            pass

    err = None
    if provider is not None and own:
        prompt = STAGE5_PROMPT.format(
            sessions=stats["sessions"],
            archetype_name=arch.name,
            archetype_tagline=arch.tagline,
            runner_up=runner.name,
            # git_commits_after overlaps between sessions in one repo, so it is
            # evidence per session, not a total worth quoting.
            stats=json.dumps({k: v for k, v in stats.items()
                              if k not in ("hours", "top_projects", "git_commits_after_sessions")}, indent=1, default=str),
            patterns=json.dumps(patterns, indent=1, ensure_ascii=False),
            rules=json.dumps([{"rule": r["rule"], "sessions": r["count"]} for r in rules], indent=1, ensure_ascii=False),
        )
        raw, e = call_with_retries(lambda: provider.analyze(prompt, model=model), max_attempts=3)
        if raw is None:
            err = f"{type(e).__name__}: {e}" if e else "unknown error"
        else:
            parsed = parse_json_lenient(raw)
            if parsed is None:
                err = "bad_json"
            else:
                profile["narrative"] = parsed
                profile["_llm_done"] = True

    atomic_write_json(out_path, profile)
    return ProfileResult(profile=profile, cached=False, error=err)


def load_profile() -> dict | None:
    path = get_atlas_dir() / "profile.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "ARCHETYPES", "Archetype", "ProfileResult", "build_profile", "compute_stats",
    "compute_features", "load_profile", "score_archetypes", "vibe_scores", "tool_comparison",
]
