"""
Pipeline data schemas.

Dataclasses + validators for the four stage outputs. All share:

  - `schema_version`: so we can detect drift when schemas evolve.
  - `from_dict` / `to_dict`: tolerant of missing optional fields so partial LLM
    outputs are salvageable.
  - `validate`: returns a list of issues; empty list means well-formed.

We deliberately don't use pydantic to keep dependencies minimal (see
pyproject.toml philosophy: stdlib-first).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from .taxonomy import (
    CONFIDENCES,
    DRIVERS,
    FRICTION_CAUSES,
    FRICTION_TYPES,
    OUTCOMES,
    RULE_SCOPES,
    TASK_TYPES,
    DIMENSION_BY_NAME,
    LEVELS,
    TRAJECTORIES,
)


SCHEMA_VERSION = 2  # v2: vibes / friction / outcome / rules


# ---------------------------------------------------------------------------
# Stage 1: session facts
# ---------------------------------------------------------------------------


@dataclass
class Evidence:
    """Pin a claim to a concrete message in the session."""
    msg_idx: int
    behavior: str

    @classmethod
    def from_dict(cls, d: dict) -> "Evidence":
        return cls(
            msg_idx=int(d.get("msg_idx", -1)),
            behavior=str(d.get("behavior", "")),
        )


@dataclass
class SkillSignal:
    """
    One skill observation for a single session.

    `dimension` should be a canonical name from `taxonomy.DIMENSION_NAMES`.
    Stage 1 prompt enforces that; validator will flag non-canonical values.
    """
    dimension: str
    level: str  # one of LEVELS
    confidence: str  # one of CONFIDENCES
    evidence: list[Evidence] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "SkillSignal":
        return cls(
            dimension=str(d.get("dimension", "")),
            level=str(d.get("level", "unobserved")),
            confidence=str(d.get("confidence", "low")),
            evidence=[Evidence.from_dict(e) for e in d.get("evidence", [])],
        )


@dataclass
class Decision:
    decision: str
    rationale: str = ""
    msg_idx: int = -1

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        return cls(
            decision=str(d.get("decision", "")),
            rationale=str(d.get("rationale", "")),
            msg_idx=int(d.get("msg_idx", -1)),
        )


@dataclass
class ProblemSolved:
    symptom: str
    root_cause: str = ""
    fix: str = ""
    msg_idx_range: list[int] = field(default_factory=list)  # [start, end] inclusive

    @classmethod
    def from_dict(cls, d: dict) -> "ProblemSolved":
        rng_raw = d.get("msg_idx_range") or []
        rng = [int(x) for x in rng_raw] if isinstance(rng_raw, list) else []
        return cls(
            symptom=str(d.get("symptom", "")),
            root_cause=str(d.get("root_cause", "")),
            fix=str(d.get("fix", "")),
            msg_idx_range=rng,
        )


@dataclass
class VibingMoment:
    description: str
    msg_idxs: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "VibingMoment":
        idxs_raw = d.get("msg_idxs") or []
        idxs = [int(x) for x in idxs_raw] if isinstance(idxs_raw, list) else []
        return cls(
            description=str(d.get("description", "")),
            msg_idxs=idxs,
        )


@dataclass
class AIFailure:
    what_ai_did_wrong: str
    msg_idx: int = -1
    user_caught_it: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "AIFailure":
        return cls(
            what_ai_did_wrong=str(d.get("what_ai_did_wrong", "")),
            msg_idx=int(d.get("msg_idx", -1)),
            user_caught_it=bool(d.get("user_caught_it", False)),
        )


@dataclass
class UserCommunication:
    rating: str = "mixed"  # good | mixed | poor
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "UserCommunication":
        return cls(
            rating=str(d.get("rating", "mixed")),
            notes=str(d.get("notes", "")),
        )


@dataclass
class OutcomeEvidence:
    msg_idx: int
    signal: str

    @classmethod
    def from_dict(cls, d: dict) -> "OutcomeEvidence":
        return cls(msg_idx=_as_int(d.get("msg_idx"), -1), signal=str(d.get("signal", "")))


@dataclass
class Outcome:
    status: str = "unknown"  # one of OUTCOMES
    confidence: str = "low"
    evidence: list[OutcomeEvidence] = field(default_factory=list)
    # Filled locally (not by the LLM): commits in the project's git repo
    # between session start and 48h after it ended. -1 = not checked.
    git_commits_after: int = -1

    @classmethod
    def from_dict(cls, d: dict) -> "Outcome":
        d = d if isinstance(d, dict) else {}
        status = str(d.get("status", "unknown"))
        conf = str(d.get("confidence", "low"))
        return cls(
            status=status if status in OUTCOMES else "unknown",
            confidence=conf if conf in CONFIDENCES else "low",
            evidence=[OutcomeEvidence.from_dict(x) for x in (d.get("evidence") or []) if isinstance(x, dict)],
            git_commits_after=_as_int(d.get("git_commits_after"), -1),
        )


@dataclass
class Friction:
    type: str
    cause: str = "unknown"
    msg_idx_range: list[int] = field(default_factory=list)
    cost_messages: int = 0
    what_happened: str = ""
    prevention: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Friction":
        t = str(d.get("type", ""))
        c = str(d.get("cause", "unknown"))
        raw_rng = d.get("msg_idx_range")
        if not isinstance(raw_rng, (list, tuple)):
            raw_rng = [raw_rng] if raw_rng is not None else []
        rng = [_as_int(x, 0) for x in raw_rng][:2]
        return cls(
            type=t if t in FRICTION_TYPES else "correction",
            cause=c if c in FRICTION_CAUSES else "unknown",
            msg_idx_range=rng,
            cost_messages=max(0, _as_int(d.get("cost_messages"), 0)),
            what_happened=str(d.get("what_happened", "")),
            prevention=str(d.get("prevention", "")),
        )


@dataclass
class RuleCandidate:
    rule: str
    prevents: str = ""
    scope: str = "general"  # general | project
    msg_idx: int = -1

    @classmethod
    def from_dict(cls, d: dict) -> "RuleCandidate":
        scope = str(d.get("scope", "general"))
        return cls(
            rule=str(d.get("rule", "")),
            prevents=str(d.get("prevents", "")),
            scope=scope if scope in RULE_SCOPES else "general",
            msg_idx=_as_int(d.get("msg_idx"), -1),
        )


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


@dataclass
class SessionFacts:
    schema_version: int
    session_id: str
    project_folder: str
    project_short: str
    timestamp: str  # ISO 8601
    source: str  # "jsonl" | "composer_db"
    message_count: int
    topic: str = ""
    summary: str = ""
    topics_touched: list[str] = field(default_factory=list)
    skills_observed: list[SkillSignal] = field(default_factory=list)
    decisions: list[Decision] = field(default_factory=list)
    problems_solved: list[ProblemSolved] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    user_communication: UserCommunication = field(default_factory=UserCommunication)
    vibing_moments: list[VibingMoment] = field(default_factory=list)  # v2: "wins"
    ai_failures: list[AIFailure] = field(default_factory=list)
    # v2 -----------------------------------------------------------------
    tool: str = ""            # human label: Cursor, Claude Code, Codex, Gemini CLI
    driver: str = "human"     # who typed the prompts: human | agent | mixed
    goal: str = ""
    task_type: str = "other"
    outcome: Outcome = field(default_factory=Outcome)
    friction: list[Friction] = field(default_factory=list)
    rule_candidates: list[RuleCandidate] = field(default_factory=list)
    ended_at: str = ""        # ISO 8601, last activity

    @classmethod
    def from_dict(cls, d: dict) -> "SessionFacts":
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            session_id=str(d.get("session_id", "")),
            project_folder=str(d.get("project_folder", "")),
            project_short=str(d.get("project_short", "")),
            timestamp=str(d.get("timestamp", "")),
            source=str(d.get("source", "")),
            message_count=int(d.get("message_count", 0)),
            topic=str(d.get("topic", "")),
            summary=str(d.get("summary", "")),
            topics_touched=[str(x) for x in (d.get("topics_touched") or [])],
            skills_observed=[SkillSignal.from_dict(x) for x in (d.get("skills_observed") or [])],
            decisions=[Decision.from_dict(x) for x in (d.get("decisions") or [])],
            problems_solved=[ProblemSolved.from_dict(x) for x in (d.get("problems_solved") or [])],
            open_questions=[str(x) for x in (d.get("open_questions") or [])],
            user_communication=UserCommunication.from_dict(d.get("user_communication") or {}),
            vibing_moments=[VibingMoment.from_dict(x) for x in (d.get("vibing_moments") or [])],
            ai_failures=[AIFailure.from_dict(x) for x in (d.get("ai_failures") or [])],
            tool=str(d.get("tool", "")),
            driver=str(d.get("driver", "human")) if d.get("driver") in DRIVERS else "human",
            goal=str(d.get("goal", "")),
            task_type=str(d.get("task_type", "other")) if d.get("task_type") in TASK_TYPES else "other",
            outcome=Outcome.from_dict(d.get("outcome") or {}),
            friction=[Friction.from_dict(x) for x in (d.get("friction") or []) if isinstance(x, dict)],
            rule_candidates=[
                RuleCandidate.from_dict(x) for x in (d.get("rule_candidates") or [])
                if isinstance(x, dict) and x.get("rule")
            ],
            ended_at=str(d.get("ended_at", "")),
        )

    @property
    def friction_cost(self) -> int:
        return sum(f.cost_messages for f in self.friction)

    def to_dict(self) -> dict:
        return asdict(self)

    def validate(self) -> list[str]:
        """Return a list of validation issues. Empty list = clean."""
        issues: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            issues.append(f"schema_version {self.schema_version} != expected {SCHEMA_VERSION}")
        if not self.session_id:
            issues.append("missing session_id")
        if self.message_count < 0:
            issues.append(f"negative message_count: {self.message_count}")

        for i, s in enumerate(self.skills_observed):
            if s.dimension not in DIMENSION_BY_NAME:
                issues.append(f"skills_observed[{i}]: unknown dimension '{s.dimension}'")
            if s.level not in LEVELS:
                issues.append(f"skills_observed[{i}]: invalid level '{s.level}'")
            if s.confidence not in CONFIDENCES:
                issues.append(f"skills_observed[{i}]: invalid confidence '{s.confidence}'")
            for j, e in enumerate(s.evidence):
                if e.msg_idx < 0 or e.msg_idx >= self.message_count:
                    issues.append(
                        f"skills_observed[{i}].evidence[{j}]: msg_idx {e.msg_idx} "
                        f"out of range [0, {self.message_count})"
                    )
        return issues


# ---------------------------------------------------------------------------
# Stage 2: project synthesis
# ---------------------------------------------------------------------------


@dataclass
class ProjectSkillEntry:
    peak_level: str = "unobserved"
    current_level: str = "unobserved"
    trajectory: str = "insufficient-data"
    confidence: str = "low"
    evidence_sessions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectSkillEntry":
        return cls(
            peak_level=str(d.get("peak_level", "unobserved")),
            current_level=str(d.get("current_level", "unobserved")),
            trajectory=str(d.get("trajectory", "insufficient-data")),
            confidence=str(d.get("confidence", "low")),
            evidence_sessions=[str(s) for s in (d.get("evidence_sessions") or [])],
        )


@dataclass
class ProjectArcPhase:
    start: str  # YYYY-MM
    end: str
    name: str
    focus: str = ""


@dataclass
class ProjectPivot:
    when: str  # YYYY-MM
    from_: str
    to: str
    why: str = ""

    def to_dict(self) -> dict:
        return {"when": self.when, "from": self.from_, "to": self.to, "why": self.why}

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectPivot":
        return cls(
            when=str(d.get("when", "")),
            from_=str(d.get("from", d.get("from_", ""))),
            to=str(d.get("to", "")),
            why=str(d.get("why", "")),
        )


@dataclass
class ProjectPattern:
    pattern: str
    fix: str = ""  # only meaningful for pitfalls
    evidence_sessions: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectPattern":
        return cls(
            pattern=str(d.get("pattern", "")),
            fix=str(d.get("fix", "")),
            evidence_sessions=[str(s) for s in (d.get("evidence_sessions") or [])],
        )


@dataclass
class ProjectKeyDecision:
    decision: str
    rationale: str = ""
    session_id: str = ""
    msg_idx: int = -1

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectKeyDecision":
        return cls(
            decision=str(d.get("decision", "")),
            rationale=str(d.get("rationale", "")),
            session_id=str(d.get("session_id", "")),
            msg_idx=int(d.get("msg_idx", -1)),
        )


@dataclass
class ProjectDNA:
    what_it_is: str = ""
    key_components: list[str] = field(default_factory=list)
    architecture_decisions: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectDNA":
        return cls(
            what_it_is=str(d.get("what_it_is", "")),
            key_components=[str(x) for x in (d.get("key_components") or [])],
            architecture_decisions=[str(x) for x in (d.get("architecture_decisions") or [])],
            constraints=[str(x) for x in (d.get("constraints") or [])],
        )


@dataclass
class ProjectArc:
    phases: list[ProjectArcPhase] = field(default_factory=list)
    pivots: list[ProjectPivot] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectArc":
        phases_raw = d.get("phases") or []
        return cls(
            phases=[
                ProjectArcPhase(
                    start=str(p.get("start", "")),
                    end=str(p.get("end", "")),
                    name=str(p.get("name", "")),
                    focus=str(p.get("focus", "")),
                )
                for p in phases_raw
            ],
            pivots=[ProjectPivot.from_dict(p) for p in (d.get("pivots") or [])],
        )


@dataclass
class ProjectAICollaboration:
    what_worked: str = ""
    what_broke_down: str = ""
    typical_failure_mode: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectAICollaboration":
        return cls(
            what_worked=str(d.get("what_worked", "")),
            what_broke_down=str(d.get("what_broke_down", "")),
            typical_failure_mode=str(d.get("typical_failure_mode", "")),
        )


@dataclass
class ProjectSynthesis:
    schema_version: int
    project_short: str
    project_folder: str
    real_path: str | None
    first_session: str
    last_session: str
    session_count: int
    message_count: int
    dna: ProjectDNA = field(default_factory=ProjectDNA)
    arc: ProjectArc = field(default_factory=ProjectArc)
    skill_profile: dict[str, ProjectSkillEntry] = field(default_factory=dict)
    project_specific_dimensions: list[dict] = field(default_factory=list)
    recurring_strengths: list[ProjectPattern] = field(default_factory=list)
    recurring_pitfalls: list[ProjectPattern] = field(default_factory=list)
    key_decisions: list[ProjectKeyDecision] = field(default_factory=list)
    ai_collaboration: ProjectAICollaboration = field(default_factory=ProjectAICollaboration)
    # v2 (local aggregation unless noted) ---------------------------------
    headline: str = ""                                        # LLM: one public-safe line
    outcomes: dict[str, int] = field(default_factory=dict)    # status -> sessions
    tools: dict[str, int] = field(default_factory=dict)       # tool label -> sessions
    drivers: dict[str, int] = field(default_factory=dict)     # human/agent/mixed -> sessions
    task_types: dict[str, int] = field(default_factory=dict)
    friction: list[dict] = field(default_factory=list)        # [{type, cause, count, cost_messages}]
    friction_cost: int = 0
    project_rules: list[dict] = field(default_factory=list)   # [{rule, count, prevents}]

    @classmethod
    def from_dict(cls, d: dict) -> "ProjectSynthesis":
        prof_raw = d.get("skill_profile") or {}
        prof = {k: ProjectSkillEntry.from_dict(v or {}) for k, v in prof_raw.items()}
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            project_short=str(d.get("project_short", "")),
            project_folder=str(d.get("project_folder", "")),
            real_path=d.get("real_path"),
            first_session=str(d.get("first_session", "")),
            last_session=str(d.get("last_session", "")),
            session_count=int(d.get("session_count", 0)),
            message_count=int(d.get("message_count", 0)),
            dna=ProjectDNA.from_dict(d.get("dna") or {}),
            arc=ProjectArc.from_dict(d.get("arc") or {}),
            skill_profile=prof,
            project_specific_dimensions=list(d.get("project_specific_dimensions") or []),
            recurring_strengths=[ProjectPattern.from_dict(x) for x in (d.get("recurring_strengths") or [])],
            recurring_pitfalls=[ProjectPattern.from_dict(x) for x in (d.get("recurring_pitfalls") or [])],
            key_decisions=[ProjectKeyDecision.from_dict(x) for x in (d.get("key_decisions") or [])],
            ai_collaboration=ProjectAICollaboration.from_dict(d.get("ai_collaboration") or {}),
            headline=str(d.get("headline", "")),
            outcomes={str(k): int(v) for k, v in (d.get("outcomes") or {}).items()},
            tools={str(k): int(v) for k, v in (d.get("tools") or {}).items()},
            drivers={str(k): int(v) for k, v in (d.get("drivers") or {}).items()},
            task_types={str(k): int(v) for k, v in (d.get("task_types") or {}).items()},
            friction=[x for x in (d.get("friction") or []) if isinstance(x, dict)],
            friction_cost=int(d.get("friction_cost", 0) or 0),
            project_rules=[x for x in (d.get("project_rules") or []) if isinstance(x, dict)],
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        # Re-key pivot's 'from_' -> 'from' for JSON friendliness.
        for p in d.get("arc", {}).get("pivots", []):
            if "from_" in p:
                p["from"] = p.pop("from_")
        return d

    def validate(self) -> list[str]:
        issues: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            issues.append(f"schema_version {self.schema_version} != expected {SCHEMA_VERSION}")
        for name, entry in self.skill_profile.items():
            if name not in DIMENSION_BY_NAME:
                issues.append(f"skill_profile: unknown dimension '{name}'")
            if entry.peak_level not in LEVELS:
                issues.append(f"skill_profile['{name}'].peak_level invalid: {entry.peak_level}")
            if entry.current_level not in LEVELS:
                issues.append(f"skill_profile['{name}'].current_level invalid: {entry.current_level}")
            if entry.trajectory not in TRAJECTORIES:
                issues.append(f"skill_profile['{name}'].trajectory invalid: {entry.trajectory}")
        return issues


# ---------------------------------------------------------------------------
# Stage 3: timeline
# ---------------------------------------------------------------------------


@dataclass
class TimelineSkillSnapshot:
    level: str = "unobserved"
    confidence: str = "low"

    @classmethod
    def from_dict(cls, d: dict) -> "TimelineSkillSnapshot":
        return cls(
            level=str(d.get("level", "unobserved")),
            confidence=str(d.get("confidence", "low")),
        )


@dataclass
class TimelineActiveProject:
    name: str
    sessions: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "TimelineActiveProject":
        return cls(
            name=str(d.get("name", "")),
            sessions=int(d.get("sessions", 0)),
        )


@dataclass
class TimelinePeriod:
    period: str  # "2025-Q3" or "2025-08"
    sessions: int = 0
    active_projects: list[TimelineActiveProject] = field(default_factory=list)
    top_topics: list[str] = field(default_factory=list)
    skill_snapshot: dict[str, TimelineSkillSnapshot] = field(default_factory=dict)
    deltas_vs_prev: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "TimelinePeriod":
        snap_raw = d.get("skill_snapshot") or {}
        snap = {k: TimelineSkillSnapshot.from_dict(v or {}) for k, v in snap_raw.items()}
        return cls(
            period=str(d.get("period", "")),
            sessions=int(d.get("sessions", 0)),
            active_projects=[TimelineActiveProject.from_dict(p) for p in (d.get("active_projects") or [])],
            top_topics=[str(x) for x in (d.get("top_topics") or [])],
            skill_snapshot=snap,
            deltas_vs_prev={str(k): str(v) for k, v in (d.get("deltas_vs_prev") or {}).items()},
        )


@dataclass
class TimelineNarrative:
    strongest_growth: list[str] = field(default_factory=list)
    surprising_moves: list[str] = field(default_factory=list)
    emerging_interests: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "TimelineNarrative":
        return cls(
            strongest_growth=[str(x) for x in (d.get("strongest_growth") or [])],
            surprising_moves=[str(x) for x in (d.get("surprising_moves") or [])],
            emerging_interests=[str(x) for x in (d.get("emerging_interests") or [])],
        )


@dataclass
class Timeline:
    schema_version: int
    generated_at: str
    periods_monthly: list[TimelinePeriod] = field(default_factory=list)
    periods_quarterly: list[TimelinePeriod] = field(default_factory=list)
    narrative: TimelineNarrative = field(default_factory=TimelineNarrative)

    @classmethod
    def from_dict(cls, d: dict) -> "Timeline":
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            generated_at=str(d.get("generated_at", "")),
            periods_monthly=[TimelinePeriod.from_dict(p) for p in (d.get("periods_monthly") or [])],
            periods_quarterly=[TimelinePeriod.from_dict(p) for p in (d.get("periods_quarterly") or [])],
            narrative=TimelineNarrative.from_dict(d.get("narrative") or {}),
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Stage 4: cross-project
# ---------------------------------------------------------------------------


@dataclass
class CrossProjectCell:
    level: str = "unobserved"
    sessions: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "CrossProjectCell":
        return cls(
            level=str(d.get("level", "unobserved")),
            sessions=int(d.get("sessions", 0)),
        )


@dataclass
class CrossProjectAnomaly:
    observation: str
    dimensions_involved: list[str] = field(default_factory=list)
    evidence: list[dict] = field(default_factory=list)  # [{"project": "...", "session": "..."}]
    hypothesis: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "CrossProjectAnomaly":
        return cls(
            observation=str(d.get("observation", "")),
            dimensions_involved=[str(x) for x in (d.get("dimensions_involved") or [])],
            evidence=list(d.get("evidence") or []),
            hypothesis=str(d.get("hypothesis", "")),
        )


@dataclass
class CrossProject:
    schema_version: int
    generated_at: str
    # {dimension: {project_short: CrossProjectCell}}
    dimension_matrix: dict[str, dict[str, CrossProjectCell]] = field(default_factory=dict)
    anomalies: list[CrossProjectAnomaly] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "CrossProject":
        matrix_raw = d.get("dimension_matrix") or {}
        matrix: dict[str, dict[str, CrossProjectCell]] = {}
        for dim, row in matrix_raw.items():
            if not isinstance(row, dict):
                continue
            matrix[str(dim)] = {
                str(proj): CrossProjectCell.from_dict(cell or {})
                for proj, cell in row.items()
            }
        return cls(
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            generated_at=str(d.get("generated_at", "")),
            dimension_matrix=matrix,
            anomalies=[CrossProjectAnomaly.from_dict(a) for a in (d.get("anomalies") or [])],
        )

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "dimension_matrix": {
                dim: {proj: asdict(cell) for proj, cell in row.items()}
                for dim, row in self.dimension_matrix.items()
            },
            "anomalies": [asdict(a) for a in self.anomalies],
        }


__all__ = [
    "SCHEMA_VERSION",
    # Stage 1
    "Evidence", "SkillSignal", "Decision", "ProblemSolved", "VibingMoment",
    "Outcome", "OutcomeEvidence", "Friction", "RuleCandidate",
    "AIFailure", "UserCommunication", "SessionFacts",
    # Stage 2
    "ProjectSkillEntry", "ProjectArcPhase", "ProjectPivot", "ProjectPattern",
    "ProjectKeyDecision", "ProjectDNA", "ProjectArc", "ProjectAICollaboration",
    "ProjectSynthesis",
    # Stage 3
    "TimelineSkillSnapshot", "TimelineActiveProject", "TimelinePeriod",
    "TimelineNarrative", "Timeline",
    # Stage 4
    "CrossProjectCell", "CrossProjectAnomaly", "CrossProject",
]
