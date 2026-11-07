"""
Stage 3: timeline aggregation.

Input:  list[SessionFacts] across all projects
Output: Timeline (monthly + quarterly buckets + one narrative)
On disk: atlas/timeline.json

We bucket sessions into periods and compute:
  - session count per period
  - active projects (name + session count) per period
  - top topics per period (from topics_touched, frequency-ranked)
  - skill_snapshot per period: for each dimension, the dominant level observed
    that period. "Dominant" = most frequent level weighted by confidence.
  - deltas_vs_prev: for each dimension that changed level across adjacent
    periods, a human-readable "beginner -> working" string.

The LLM call is ONE shot across the whole timeline to produce a narrative:
  - strongest_growth (dimensions with biggest positive trajectory)
  - surprising_moves (unexpected transitions: e.g. sudden regression, pivots)
  - emerging_interests (topics appearing late in the timeline)

Cache: timeline.json embeds `_input_fingerprint = sha256(multiset of all
session facts)`. Re-runs that match skip the LLM call.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..paths import get_atlas_dir
from ..providers import LLMProvider
from .llm_utils import atomic_write_json, call_with_retries, parse_json_lenient
from .schemas import (
    SCHEMA_VERSION,
    SessionFacts,
    Timeline,
    TimelineActiveProject,
    TimelineNarrative,
    TimelinePeriod,
    TimelineSkillSnapshot,
)
from .taxonomy import DIMENSION_BY_NAME, LEVEL_RANK, LEVELS


CONF_RANK = {"low": 1, "med": 2, "high": 3}


# ---------------------------------------------------------------------------
# Bucketing
# ---------------------------------------------------------------------------


def _parse_ts(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        # Sometimes we get bare YYYY-MM-DD
        try:
            return datetime.strptime(ts[:10], "%Y-%m-%d")
        except ValueError:
            return None


def month_bucket(ts: str) -> str | None:
    dt = _parse_ts(ts)
    if dt is None:
        return None
    return f"{dt.year:04d}-{dt.month:02d}"


def quarter_bucket(ts: str) -> str | None:
    dt = _parse_ts(ts)
    if dt is None:
        return None
    q = (dt.month - 1) // 3 + 1
    return f"{dt.year:04d}-Q{q}"


def _iter_months_inclusive(first: str, last: str) -> list[str]:
    """Yield every YYYY-MM between first and last inclusive (zero-fill gaps)."""
    dt1 = _parse_ts(first)
    dt2 = _parse_ts(last)
    if dt1 is None or dt2 is None:
        return []
    out: list[str] = []
    y, m = dt1.year, dt1.month
    while (y, m) <= (dt2.year, dt2.month):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def _iter_quarters_inclusive(first: str, last: str) -> list[str]:
    dt1 = _parse_ts(first)
    dt2 = _parse_ts(last)
    if dt1 is None or dt2 is None:
        return []
    out: list[str] = []
    y = dt1.year
    q = (dt1.month - 1) // 3 + 1
    end_y = dt2.year
    end_q = (dt2.month - 1) // 3 + 1
    while (y, q) <= (end_y, end_q):
        out.append(f"{y:04d}-Q{q}")
        q += 1
        if q > 4:
            y += 1
            q = 1
    return out


# ---------------------------------------------------------------------------
# Per-period aggregation
# ---------------------------------------------------------------------------


def _dominant_level(observations: list[tuple[str, str]]) -> tuple[str, str]:
    """
    Given [(level, confidence), ...] pick the dominant (level, confidence).
    Weight each observation by its confidence rank and pick the level with the
    highest total weight. Confidence of the output is the max seen for that
    level.
    """
    if not observations:
        return ("unobserved", "low")
    weights: dict[str, int] = defaultdict(int)
    confs_seen: dict[str, list[str]] = defaultdict(list)
    for level, conf in observations:
        if level not in LEVELS:
            continue
        w = CONF_RANK.get(conf, 1)
        weights[level] += w
        confs_seen[level].append(conf)
    if not weights:
        return ("unobserved", "low")
    level = max(weights, key=lambda lv: (weights[lv], LEVEL_RANK[lv]))
    inv_rank = {v: k for k, v in CONF_RANK.items()}
    best_conf = inv_rank[max(CONF_RANK.get(c, 1) for c in confs_seen[level])]
    return (level, best_conf)


def _aggregate_period(
    period_label: str,
    facts: list[SessionFacts],
) -> TimelinePeriod:
    active_counts: Counter[str] = Counter()
    topic_counts: Counter[str] = Counter()
    by_dim: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for f in facts:
        if f.project_short:
            active_counts[f.project_short] += 1
        for t in f.topics_touched:
            t_norm = t.strip().lower()
            if t_norm:
                topic_counts[t_norm] += 1
        for s in f.skills_observed:
            if s.dimension not in DIMENSION_BY_NAME:
                continue
            by_dim[s.dimension].append((s.level, s.confidence))

    skill_snapshot = {
        dim: TimelineSkillSnapshot(level=lv, confidence=conf)
        for dim, obs in by_dim.items()
        for (lv, conf) in [_dominant_level(obs)]
    }

    active_projects = [
        TimelineActiveProject(name=name, sessions=count)
        for name, count in active_counts.most_common()
    ]
    top_topics = [t for t, _ in topic_counts.most_common(8)]

    return TimelinePeriod(
        period=period_label,
        sessions=len(facts),
        active_projects=active_projects,
        top_topics=top_topics,
        skill_snapshot=skill_snapshot,
    )


def _fill_deltas(periods: list[TimelinePeriod]) -> None:
    """
    For each period after the first, write deltas_vs_prev for dimensions whose
    dominant level changed. In-place mutation.
    """
    prev: dict[str, TimelineSkillSnapshot] | None = None
    for p in periods:
        if prev is not None:
            deltas: dict[str, str] = {}
            for dim, snap in p.skill_snapshot.items():
                before = prev.get(dim)
                if before is None and snap.level != "unobserved":
                    deltas[dim] = f"unobserved -> {snap.level}"
                elif before is not None and before.level != snap.level:
                    deltas[dim] = f"{before.level} -> {snap.level}"
            p.deltas_vs_prev = deltas
        prev = p.skill_snapshot


def _bucket_facts(
    facts: list[SessionFacts],
    bucket_fn,
    iter_range_fn,
) -> list[TimelinePeriod]:
    by_period: dict[str, list[SessionFacts]] = defaultdict(list)
    for f in facts:
        b = bucket_fn(f.timestamp)
        if b is None:
            continue
        by_period[b].append(f)

    if not by_period:
        return []

    labels = sorted(by_period.keys())
    # Fill gaps with empty periods so the timeline reads contiguously.
    labels = iter_range_fn(labels[0] + "-01", labels[-1] + "-01") if bucket_fn is month_bucket \
        else iter_range_fn(_quarter_to_date(labels[0]), _quarter_to_date(labels[-1]))

    periods: list[TimelinePeriod] = []
    for label in labels:
        periods.append(_aggregate_period(label, by_period.get(label, [])))
    _fill_deltas(periods)
    return periods


def _quarter_to_date(q_label: str) -> str:
    """Convert '2025-Q3' to a timestamp string for _iter_quarters_inclusive."""
    y, q = q_label.split("-Q")
    month = (int(q) - 1) * 3 + 1
    return f"{int(y):04d}-{month:02d}-01"


# ---------------------------------------------------------------------------
# LLM narrative
# ---------------------------------------------------------------------------


STAGE3_PROMPT = """You are writing a brief narrative about how a developer's skills
evolved over a time range. You see the full per-period skill snapshots; pick
out what's most interesting.

# Rules

1. "strongest_growth": list <=5 dimensions that made the clearest positive
   transition (e.g. beginner -> strong) over the range. Use canonical
   dimension names only.
2. "surprising_moves": list <=5 dimensions that did something unexpected
   (regression, abrupt jumps, reappearance after dormancy).
3. "emerging_interests": list <=5 topics (free-form, from top_topics) that
   show up in the late periods but not earlier.
4. Keep each list item a short phrase, not a paragraph.
5. Be honest: if the timeline is short or signals are thin, return short
   lists. Do not invent growth that isn't there.
6. Respond with a SINGLE JSON object matching the schema:
   {{"strongest_growth": [...], "surprising_moves": [...], "emerging_interests": [...]}}

# Timeline data

Monthly periods:
{monthly}

Quarterly periods:
{quarterly}
"""


def _period_payload(p: TimelinePeriod) -> dict:
    return {
        "period": p.period,
        "sessions": p.sessions,
        "active_projects": [{"name": a.name, "sessions": a.sessions} for a in p.active_projects],
        "top_topics": p.top_topics,
        "skill_snapshot": {
            dim: {"level": s.level, "confidence": s.confidence}
            for dim, s in p.skill_snapshot.items()
        },
        "deltas_vs_prev": p.deltas_vs_prev,
    }


def _build_prompt(monthly: list[TimelinePeriod], quarterly: list[TimelinePeriod]) -> str:
    return STAGE3_PROMPT.format(
        monthly=json.dumps([_period_payload(p) for p in monthly], indent=2, ensure_ascii=False),
        quarterly=json.dumps([_period_payload(p) for p in quarterly], indent=2, ensure_ascii=False),
    )


def _input_fingerprint(facts: list[SessionFacts]) -> str:
    h = hashlib.sha256()
    for f in sorted(facts, key=lambda x: x.session_id):
        h.update(f.session_id.encode())
        h.update(b"|")
        h.update(str(f.message_count).encode())
        h.update(b"|")
        h.update(f.timestamp.encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class TimelineResult:
    timeline: Timeline | None
    cached: bool = False
    error: str | None = None


def build_timeline(
    facts: list[SessionFacts],
    provider: LLMProvider | None,
    *,
    model: str | None = None,
    force: bool = False,
) -> TimelineResult:
    """
    Build the timeline from all session facts across every project.

    If `provider` is None, we skip the LLM call and return a Timeline with an
    empty narrative. Useful for local-only / no-key mode.
    """
    if not facts:
        return TimelineResult(timeline=None, error="no facts provided")

    out_path = get_atlas_dir() / "timeline.json"
    input_fp = _input_fingerprint(facts)

    if not force and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("_input_fingerprint") == input_fp:
                clean = {k: v for k, v in existing.items() if not k.startswith("_")}
                return TimelineResult(timeline=Timeline.from_dict(clean), cached=True)
        except (OSError, json.JSONDecodeError):
            pass

    monthly = _bucket_facts(facts, month_bucket, _iter_months_inclusive)
    quarterly = _bucket_facts(facts, quarter_bucket, _iter_quarters_inclusive)

    narrative = TimelineNarrative()
    if provider is not None:
        prompt = _build_prompt(monthly, quarterly)

        def _do_call() -> str:
            return provider.analyze(prompt, model=model)

        raw, err = call_with_retries(_do_call, max_attempts=3)
        if raw is not None:
            parsed = parse_json_lenient(raw) or {}
            narrative = TimelineNarrative.from_dict(parsed)
        else:
            # Narrative stays empty; structural data still written.
            pass

    timeline = Timeline(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        periods_monthly=monthly,
        periods_quarterly=quarterly,
        narrative=narrative,
    )

    payload = timeline.to_dict()
    payload["_input_fingerprint"] = input_fp
    payload["_provider"] = provider.info.name if provider else ""
    payload["_model"] = (model or (provider.info.default_model if provider else ""))
    atomic_write_json(out_path, payload)
    return TimelineResult(timeline=timeline, cached=False)


def load_timeline() -> Timeline | None:
    path = get_atlas_dir() / "timeline.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        return Timeline.from_dict(clean)
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "TimelineResult",
    "build_timeline",
    "load_timeline",
    "month_bucket",
    "quarter_bucket",
]
