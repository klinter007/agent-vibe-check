"""
Stage 2: per-project synthesis.

Input:  list[SessionFacts] for one project (written by Stage 1)
Output: ProjectSynthesis, stored at extractions/<short>/project.json

We split the work into:
  1. Local aggregation (pure python): counts, skill_profile, trajectory.
     This is deterministic and doesn't need an LLM.
  2. LLM narrative call: dna, arc, recurring patterns, ai_collaboration, top
     decisions. The LLM sees per-session topics + summaries + skill signals,
     NOT the raw transcripts. Stage 1 already distilled them; repeating that
     work would burn tokens.

Cache:
  Stage 2 output is cheap relative to Stage 1 (~1 call per project), but LLM
  output varies slightly run-to-run so we still want to skip when inputs
  haven't changed. We hash the multiset of (session_id, message_count,
  mtime_ns) across all Stage 1 facts and store it in the project.json under
  `_input_fingerprint`. On re-run, matching fingerprint means skip.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..atlas import _build_workspace_map
from ..providers import LLMProvider
from ..paths import get_extractions_dir
from .llm_utils import atomic_write_json, call_with_retries, parse_json_lenient
from .schemas import (
    SCHEMA_VERSION,
    ProjectAICollaboration,
    ProjectArc,
    ProjectArcPhase,
    ProjectDNA,
    ProjectKeyDecision,
    ProjectPattern,
    ProjectPivot,
    ProjectSkillEntry,
    ProjectSynthesis,
    SessionFacts,
)
from .taxonomy import (
    DIMENSION_BY_NAME,
    LEVEL_RANK,
    LEVELS,
    TRAJECTORIES,
    dimension_list_for_prompt,
)


# ---------------------------------------------------------------------------
# Local aggregation (no LLM)
# ---------------------------------------------------------------------------


def _project_root(project_short: str) -> Path:
    p = get_extractions_dir() / project_short
    p.mkdir(parents=True, exist_ok=True)
    return p


def _input_fingerprint(facts: list[SessionFacts]) -> str:
    """
    Hash the set of inputs so we can tell when Stage 1 has updated.
    Sort by session_id so the order Stage 1 produced in doesn't matter.
    """
    h = hashlib.sha256()
    for f in sorted(facts, key=lambda x: x.session_id):
        h.update(f.session_id.encode())
        h.update(b"|")
        h.update(str(f.message_count).encode())
        h.update(b"|")
        h.update(f.timestamp.encode())
        h.update(b"\n")
    return h.hexdigest()


def _sort_facts_by_timestamp(facts: list[SessionFacts]) -> list[SessionFacts]:
    def key(f: SessionFacts) -> str:
        return f.timestamp or "0000-00-00"
    return sorted(facts, key=key)


def _aggregate_skill_profile(
    facts: list[SessionFacts],
) -> dict[str, ProjectSkillEntry]:
    """
    For every dimension observed across the project, produce:
      - peak_level (max rank ever observed)
      - current_level (most recent observation)
      - trajectory (improving/stable/regressing/insufficient-data)
      - confidence (worst-of strategy: conservative)
      - evidence_sessions (session IDs that contributed signals)

    Trajectory rule: split observations in half by time. If second-half
    average rank > first-half + 0.5 -> improving, < -0.5 -> regressing,
    else stable. Fewer than 2 observations -> insufficient-data.
    """
    sorted_facts = _sort_facts_by_timestamp(facts)

    # dimension -> list of (timestamp, rank, confidence, session_id)
    by_dim: dict[str, list[tuple[str, int, str, str]]] = defaultdict(list)
    for f in sorted_facts:
        for s in f.skills_observed:
            if s.dimension not in DIMENSION_BY_NAME:
                continue
            rank = LEVEL_RANK.get(s.level, 0)
            by_dim[s.dimension].append((f.timestamp, rank, s.confidence, f.session_id))

    out: dict[str, ProjectSkillEntry] = {}
    for dim, obs in by_dim.items():
        peak_rank = max(o[1] for o in obs)
        current_rank = obs[-1][1]
        # Worst-of for confidence (conservative aggregation)
        conf_rank = {"low": 0, "med": 1, "high": 2}
        rev = {0: "low", 1: "med", 2: "high"}
        worst_conf = rev[min(conf_rank.get(o[2], 0) for o in obs)]

        if len(obs) < 2:
            trajectory = "insufficient-data"
        else:
            mid = len(obs) // 2
            first_half = obs[:mid] if mid > 0 else obs[:1]
            second_half = obs[mid:] if mid > 0 else obs
            avg_first = sum(o[1] for o in first_half) / len(first_half)
            avg_second = sum(o[1] for o in second_half) / len(second_half)
            delta = avg_second - avg_first
            if delta > 0.5:
                trajectory = "improving"
            elif delta < -0.5:
                trajectory = "regressing"
            else:
                trajectory = "stable"

        # Dedup session IDs, preserve chronological order
        seen = set()
        evidence: list[str] = []
        for o in obs:
            sid = o[3]
            if sid not in seen:
                evidence.append(sid)
                seen.add(sid)

        out[dim] = ProjectSkillEntry(
            peak_level=LEVELS[peak_rank],
            current_level=LEVELS[current_rank],
            trajectory=trajectory,
            confidence=worst_conf,
            evidence_sessions=evidence,
        )
    return out


def _count(values) -> dict[str, int]:
    out: dict[str, int] = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _aggregate_friction(facts: list[SessionFacts]) -> list[dict]:
    agg: dict[tuple[str, str], dict] = {}
    for f in facts:
        for x in f.friction:
            row = agg.setdefault((x.type, x.cause), {
                "type": x.type, "cause": x.cause, "count": 0, "cost_messages": 0, "examples": [],
            })
            row["count"] += 1
            row["cost_messages"] += x.cost_messages
            if x.what_happened and len(row["examples"]) < 3:
                row["examples"].append(x.what_happened)
    return sorted(agg.values(), key=lambda r: (-r["cost_messages"], -r["count"]))


def _rule_key(rule: str) -> str:
    import re
    words = re.findall(r"[a-z0-9]+", rule.lower())
    return " ".join(words[:8])


def gather_rules(facts: list[SessionFacts], scope: str, limit: int = 12) -> list[dict]:
    """Dedupe rule candidates of one scope by their first words; most-cited first."""
    agg: dict[str, dict] = {}
    for f in facts:
        for r in f.rule_candidates:
            if r.scope != scope or not r.rule.strip():
                continue
            k = _rule_key(r.rule)
            row = agg.setdefault(k, {"rule": r.rule.strip(), "prevents": r.prevents, "count": 0, "sessions": []})
            row["count"] += 1
            if f.session_id not in row["sessions"]:
                row["sessions"].append(f.session_id)
    rows = sorted(agg.values(), key=lambda r: -r["count"])
    return rows[:limit]


def _gather_key_decisions(facts: list[SessionFacts], limit: int = 15) -> list[ProjectKeyDecision]:
    """
    Collect all decisions, deduplicate near-identical ones by lowercased text,
    return the top `limit` in chronological order.
    """
    out: list[ProjectKeyDecision] = []
    seen: set[str] = set()
    for f in _sort_facts_by_timestamp(facts):
        for d in f.decisions:
            key = d.decision.strip().lower()[:120]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(ProjectKeyDecision(
                decision=d.decision,
                rationale=d.rationale,
                session_id=f.session_id,
                msg_idx=d.msg_idx,
            ))
            if len(out) >= limit:
                return out
    return out


# ---------------------------------------------------------------------------
# LLM narrative call
# ---------------------------------------------------------------------------


STAGE2_PROMPT = """You are summarizing one project from per-session facts. Each session was
already reviewed: what the person wanted, whether it shipped, how well they
directed the AI agent ("vibes"), and where time was lost ("friction").
Your job is the project-level story. You do NOT see raw transcripts.
Do not invent anything you cannot see in the facts below.

# Vibe signals (how the person directs the agent)

{taxonomy}

# Rules

1. `headline`: one sentence a stranger could read on a public profile: what
   was built or explored, in plain words. No internal names, hosts, people or
   secrets. Example: "A local CLI that turns AI chat logs into skill reports."
2. `dna.what_it_is`: one concrete sentence. If unclear, say "unclear from sessions".
3. `arc.phases`: at most 4 phases (YYYY-MM). Skip if too few sessions.
4. `recurring_strengths` / `recurring_pitfalls`: patterns in HOW the person
   works with the agent on this project, seen in 3+ sessions. Cite
   `evidence_sessions`. A pitfall gets a `fix`: one sentence the person could
   put in the agent instructions file.
5. `ai_collaboration`: what worked, what broke down, the typical failure mode.
6. Never copy secrets, emails, internal URLs or people's names.

# Output

Return a SINGLE JSON object:

{{
  "headline": "...",
  "dna": {{"what_it_is": "...", "key_components": ["..."], "architecture_decisions": ["..."], "constraints": ["..."]}},
  "arc": {{
    "phases": [{{"start": "2026-03", "end": "2026-05", "name": "prototype", "focus": "..."}}],
    "pivots": [{{"when": "2026-06", "from": "...", "to": "...", "why": "..."}}]
  }},
  "recurring_strengths": [{{"pattern": "...", "evidence_sessions": ["s-1", "s-5", "s-9"]}}],
  "recurring_pitfalls": [{{"pattern": "...", "fix": "...", "evidence_sessions": ["s-2", "s-6", "s-9"]}}],
  "ai_collaboration": {{"what_worked": "...", "what_broke_down": "...", "typical_failure_mode": "..."}}
}}

# Project metadata

short name: {project_short}
total sessions: {session_count}
first session: {first_session}
last session: {last_session}
total messages: {message_count}

# Per-session facts (chronological)

{sessions}
"""


def _session_payload(f: SessionFacts) -> dict:
    """Compact payload the LLM sees. Keep small; this multiplies by N sessions."""
    return {
        "session_id": f.session_id,
        "timestamp": f.timestamp,
        "source": f.source,
        "message_count": f.message_count,
        "topic": f.topic,
        "summary": f.summary,
        "topics_touched": f.topics_touched,
        "skills_observed": [
            {"dimension": s.dimension, "level": s.level, "confidence": s.confidence}
            for s in f.skills_observed
        ],
        "decisions": [{"decision": d.decision, "rationale": d.rationale} for d in f.decisions],
        "problems_solved": [
            {"symptom": p.symptom, "root_cause": p.root_cause, "fix": p.fix}
            for p in f.problems_solved
        ],
        "tool": f.tool,
        "driver": f.driver,
        "goal": f.goal,
        "task_type": f.task_type,
        "outcome": f.outcome.status,
        "friction": [
            {"type": x.type, "cause": x.cause, "cost": x.cost_messages, "what": x.what_happened}
            for x in f.friction
        ],
        "wins": [v.description for v in f.vibing_moments],
        "ai_failures": [
            {"what_ai_did_wrong": a.what_ai_did_wrong, "user_caught_it": a.user_caught_it}
            for a in f.ai_failures
        ],
    }


def _build_prompt(project_short: str, facts: list[SessionFacts]) -> str:
    ordered = _sort_facts_by_timestamp(facts)
    return STAGE2_PROMPT.format(
        taxonomy=dimension_list_for_prompt(),
        project_short=project_short,
        session_count=len(ordered),
        first_session=ordered[0].timestamp if ordered else "",
        last_session=ordered[-1].timestamp if ordered else "",
        message_count=sum(f.message_count for f in ordered),
        sessions=json.dumps([_session_payload(f) for f in ordered], ensure_ascii=False, indent=2),
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    synthesis: ProjectSynthesis | None
    cached: bool = False
    skipped_reason: str | None = None
    error: str | None = None


def _build_synthesis(
    project_short: str,
    project_folder: str,
    facts: list[SessionFacts],
    narrative: dict,
    input_fp: str,
    real_path_hint: str | None = None,
) -> ProjectSynthesis:
    sorted_facts = _sort_facts_by_timestamp(facts)
    first_ts = sorted_facts[0].timestamp if sorted_facts else ""
    last_ts = sorted_facts[-1].timestamp if sorted_facts else ""

    # Lazy-build the workspace map. Cheap enough that per-project lookup is
    # fine for small runs; the caller can pre-resolve by passing a synthesis
    # directly if this ever becomes a bottleneck.
    from ..sources import enabled_keys
    # Cursor's workspace files are read only when the user allowed Cursor.
    ws_map = _build_workspace_map() if "cursor" in enabled_keys() else {}
    real_path = ws_map.get(project_folder) or real_path_hint

    dna = ProjectDNA.from_dict(narrative.get("dna") or {})
    arc = ProjectArc.from_dict(narrative.get("arc") or {})
    strengths = [ProjectPattern.from_dict(p) for p in (narrative.get("recurring_strengths") or [])]
    pitfalls = [ProjectPattern.from_dict(p) for p in (narrative.get("recurring_pitfalls") or [])]

    # LLM may add meta-decisions; our local aggregator supplies the concrete ones.
    local_decisions = _gather_key_decisions(facts)
    llm_decisions_raw = narrative.get("key_decisions") or []
    llm_decisions = [ProjectKeyDecision.from_dict(d) for d in llm_decisions_raw if isinstance(d, dict)]

    ai_collab = ProjectAICollaboration.from_dict(narrative.get("ai_collaboration") or {})

    # project_specific_dimensions is a list of dicts with keys {dimension, level, confidence, notes}
    psd_raw = narrative.get("project_specific_dimensions") or []
    psd = [d for d in psd_raw if isinstance(d, dict) and d.get("dimension")]

    synthesis = ProjectSynthesis(
        schema_version=SCHEMA_VERSION,
        project_short=project_short,
        project_folder=project_folder,
        real_path=real_path,
        first_session=first_ts,
        last_session=last_ts,
        session_count=len(facts),
        message_count=sum(f.message_count for f in facts),
        dna=dna,
        arc=arc,
        skill_profile=_aggregate_skill_profile(facts),
        project_specific_dimensions=psd,
        recurring_strengths=strengths,
        recurring_pitfalls=pitfalls,
        key_decisions=local_decisions + llm_decisions,
        ai_collaboration=ai_collab,
        headline=str(narrative.get("headline", "")),
        outcomes=_count(f.outcome.status for f in facts),
        tools=_count(f.tool or f.source for f in facts),
        drivers=_count(f.driver for f in facts),
        task_types=_count(f.task_type for f in facts),
        friction=_aggregate_friction(facts),
        friction_cost=sum(f.friction_cost for f in facts),
        project_rules=gather_rules(facts, "project"),
    )
    return synthesis


def _read_existing(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def synthesize_project(
    project_short: str,
    project_folder: str,
    facts: list[SessionFacts],
    provider: LLMProvider,
    *,
    model: str | None = None,
    force: bool = False,
    min_sessions: int = 1,
    real_path_hint: str | None = None,
) -> SynthesisResult:
    """
    Run Stage 2 for one project. Returns and also writes the synthesis to disk.
    """
    if len(facts) < min_sessions:
        return SynthesisResult(
            synthesis=None,
            skipped_reason=f"too_few_sessions ({len(facts)} < {min_sessions})",
        )

    out_path = _project_root(project_short) / "project.json"
    input_fp = _input_fingerprint(facts)

    if not force:
        existing = _read_existing(out_path)
        if (existing and existing.get("_input_fingerprint") == input_fp
                and int(existing.get("schema_version", 0)) == SCHEMA_VERSION):
            try:
                existing_clean = {k: v for k, v in existing.items() if not k.startswith("_")}
                return SynthesisResult(
                    synthesis=ProjectSynthesis.from_dict(existing_clean),
                    cached=True,
                )
            except Exception:  # noqa: BLE001 - fall through to re-synthesize
                pass

    prompt = _build_prompt(project_short, facts)

    def _do_call() -> str:
        return provider.analyze(prompt, model=model)

    raw, err = call_with_retries(_do_call, max_attempts=3)
    if raw is None:
        return SynthesisResult(
            synthesis=None, error=f"{type(err).__name__}: {err}" if err else "unknown error"
        )

    narrative = parse_json_lenient(raw) or {}
    if not narrative:
        (out_path.parent / "project.raw.txt").write_text(raw, encoding="utf-8")
        # Still build a synthesis from local aggregation + empty narrative so
        # downstream stages have something to chew on.
        narrative = {}

    synthesis = _build_synthesis(project_short, project_folder, facts, narrative, input_fp,
                                 real_path_hint=real_path_hint)

    payload = synthesis.to_dict()
    payload["_input_fingerprint"] = input_fp
    payload["_generated_at"] = datetime.now().isoformat(timespec="seconds")
    payload["_provider"] = provider.info.name
    payload["_model"] = model or provider.info.default_model
    atomic_write_json(out_path, payload)

    return SynthesisResult(synthesis=synthesis, cached=False)


def load_project_synthesis(project_short: str) -> ProjectSynthesis | None:
    """Read a project's Stage 2 output from disk; None if missing/corrupt."""
    path = _project_root(project_short) / "project.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        return ProjectSynthesis.from_dict(clean)
    except (OSError, json.JSONDecodeError):
        return None


def iter_all_project_syntheses(project_shorts: Iterable[str]) -> list[ProjectSynthesis]:
    """Helper for Stages 3-4: load every project's Stage 2 output."""
    out: list[ProjectSynthesis] = []
    for short in project_shorts:
        syn = load_project_synthesis(short)
        if syn is not None:
            out.append(syn)
    return out


__all__ = [
    "SynthesisResult",
    "synthesize_project",
    "load_project_synthesis",
    "iter_all_project_syntheses",
    "gather_rules",
]
