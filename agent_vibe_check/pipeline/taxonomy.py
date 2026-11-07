"""
Shared skill taxonomy.

Purpose: give the pipeline a *controlled vocabulary* of skill dimensions so
Stage 1 outputs from different sessions can be aggregated at the project,
timeline, and cross-project levels without fuzzy string matching.

Rules:
  - Stage 1 MUST cite dimension names from `CANONICAL_DIMENSIONS`.
    If the LLM wants to flag something outside this set, it uses
    `project_specific_dimensions` at Stage 2, which never rolls up to
    cross-project views.
  - Adding a dimension is cheap but *breaks historical comparability*. Prefer
    to express variants via evidence notes rather than new dimensions.
  - `beginner_signal` / `strong_signal` descriptions are injected into the
    Stage 1 prompt so the model scores consistently across sessions.

Levels:
    unobserved < beginner < working < strong < expert

A session's `skill_signals` may include "unobserved" explicitly when the
topic came up but the session doesn't produce enough evidence to score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


LEVELS: tuple[str, ...] = ("unobserved", "beginner", "working", "strong", "expert")
LEVEL_RANK: dict[str, int] = {lvl: i for i, lvl in enumerate(LEVELS)}

TRAJECTORIES: tuple[str, ...] = (
    "improving",
    "stable",
    "regressing",
    "insufficient-data",
)

CONFIDENCES: tuple[str, ...] = ("low", "med", "high")

# v2 session-level vocabularies -------------------------------------------------

TASK_TYPES: tuple[str, ...] = (
    "build-feature", "fix-bug", "refactor", "research-explore", "setup-infra",
    "data-ml", "creative-media", "writing-docs", "ops-automation", "other",
)
OUTCOMES: tuple[str, ...] = ("shipped", "partial", "abandoned", "unknown")
DRIVERS: tuple[str, ...] = ("human", "agent", "mixed")
FRICTION_TYPES: tuple[str, ...] = (
    "rephrase",      # person asks the same thing again in other words
    "correction",    # "no, I meant...", agent did the wrong thing
    "loop",          # same failing fix tried 3+ times
    "revert",        # work undone, rolled back
    "env-failure",   # tool, auth, install, permission, quota problems
    "frustration",   # caps, swearing, "why do you keep..."
    "scope-drift",   # task grew or changed mid-session
    "abandon",       # person gives up on the goal
)
FRICTION_CAUSES: tuple[str, ...] = (
    "unclear-spec", "missing-context", "agent-error", "tool-or-env",
    "scope-change", "user-changed-mind", "unknown",
)
RULE_SCOPES: tuple[str, ...] = ("general", "project")


@dataclass(frozen=True)
class Dimension:
    """One skill axis we score across sessions/projects/time."""
    name: str
    group: str
    description: str
    beginner_signal: str
    strong_signal: str

    def prompt_line(self) -> str:
        return (
            f"- `{self.name}` ({self.group}): {self.description}\n"
            f"    beginner looks like: {self.beginner_signal}\n"
            f"    strong looks like:   {self.strong_signal}"
        )


# -- Canonical dimensions -----------------------------------------------------
# v2: we no longer grade code skills (python, react, ...). In agent work the AI
# writes the code, so those scores measured the AI. We score how the person
# DIRECTS the agent ("Vibes"). Six signals, one group.

VIBES_GROUP = "Vibes"

_VIBES = [
    Dimension(
        "specification", VIBES_GROUP,
        "Does the person say what done looks like before the work starts?",
        "one-line asks with no goal, limits or success check; the agent has to guess what good means",
        "states goal, constraints and how success will be checked up front; gives an example of the expected output",
    ),
    Dimension(
        "context", VIBES_GROUP,
        "Does the agent get what it needs, or does it guess?",
        "assumes the agent knows the history; no files, logs or exact error text; agent explores blindly",
        "points to exact files, pastes full errors, links docs, keeps a rules file (CLAUDE.md / AGENTS.md / .cursorrules) that carries context",
    ),
    Dimension(
        "verification", VIBES_GROUP,
        "Does the person check the result, or trust 'done'?",
        "accepts 'done' without running or looking; problems surface much later",
        "asks for tests or evidence, runs the result, reads diffs or outputs, catches agent mistakes quickly",
    ),
    Dimension(
        "steering", VIBES_GROUP,
        "When the agent goes wrong, how fast and how clearly does the person correct it?",
        "lets the agent loop on a failing approach for many turns; corrects late, vaguely, or only with frustration",
        "stops a wrong direction within 1-2 turns with a precise correction and the reason; changes approach when stuck",
    ),
    Dimension(
        "delegation", VIBES_GROUP,
        "Is the work cut into pieces the agent can do and the person can check?",
        "asks for everything at once, or micro-manages every line; no plan before big changes",
        "splits work into checkable steps, asks for a plan first on big changes, uses subagents or parallel sessions on purpose, takes over by hand when that is faster",
    ),
    Dimension(
        "learning-loop", VIBES_GROUP,
        "Do lessons stick?",
        "repeats the same correction across turns or sessions; nothing gets written down",
        "turns a repeated correction into a rule, skill or doc; refers back to earlier decisions; the same mistake does not come back",
    ),
]


CANONICAL_DIMENSIONS: tuple[Dimension, ...] = tuple(_VIBES)

DIMENSION_NAMES: tuple[str, ...] = tuple(d.name for d in CANONICAL_DIMENSIONS)
DIMENSION_BY_NAME: dict[str, Dimension] = {d.name: d for d in CANONICAL_DIMENSIONS}


def dimension_list_for_prompt() -> str:
    """
    Return a bulleted dimension list suitable for injecting into an LLM prompt.
    Keeping this a pure function of taxonomy contents so prompts stay in sync
    when we add/remove dimensions.
    """
    return "\n".join(d.prompt_line() for d in CANONICAL_DIMENSIONS)


def is_canonical(name: str) -> bool:
    return name in DIMENSION_BY_NAME


def compare_levels(a: str, b: str) -> int:
    """Return -1 / 0 / 1 comparing two level names by rank."""
    ra = LEVEL_RANK.get(a, -1)
    rb = LEVEL_RANK.get(b, -1)
    return (ra > rb) - (ra < rb)


__all__ = [
    "LEVELS",
    "LEVEL_RANK",
    "TRAJECTORIES",
    "CONFIDENCES",
    "RULE_SCOPES",
    "FRICTION_CAUSES",
    "FRICTION_TYPES",
    "DRIVERS",
    "OUTCOMES",
    "TASK_TYPES",
    "Dimension",
    "VIBES_GROUP",
    "CANONICAL_DIMENSIONS",
    "DIMENSION_NAMES",
    "DIMENSION_BY_NAME",
    "dimension_list_for_prompt",
    "is_canonical",
    "compare_levels",
]
