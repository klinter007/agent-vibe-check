"""
Stage 4: cross-project comparison.

Input:  list[ProjectSynthesis] (one per deep-analyzed project)
Output: CrossProject (dimension matrix + anomaly list)
On disk: atlas/cross_project.json

The matrix layer is fully local:
  {dimension: {project_short: {level, sessions}}}

The anomaly layer uses one LLM call. Its job is to look at the matrix and name
surprising patterns, e.g. "user is expert-level Python everywhere except in
project X, where they reset to beginner". The LLM is given the matrix plus a
thin slice of project metadata -- not the raw synthesis blobs -- so cost is
bounded independent of session count.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from ..paths import get_atlas_dir
from ..providers import LLMProvider
from .llm_utils import atomic_write_json, call_with_retries, parse_json_lenient
from .schemas import (
    SCHEMA_VERSION,
    CrossProject,
    CrossProjectAnomaly,
    CrossProjectCell,
    ProjectSynthesis,
)
from .taxonomy import DIMENSION_BY_NAME


# ---------------------------------------------------------------------------
# Matrix (local)
# ---------------------------------------------------------------------------


def build_matrix(
    syntheses: list[ProjectSynthesis],
) -> dict[str, dict[str, CrossProjectCell]]:
    """
    Build {dimension: {project_short: cell}}.

    cell.level is the project's `current_level` (what the user looks like
    NOW on that dimension in that context). cell.sessions counts how many
    sessions contributed evidence.

    Projects with `unobserved` or missing entries for a dimension are
    omitted from that row so the matrix stays sparse and readable.
    """
    matrix: dict[str, dict[str, CrossProjectCell]] = {}
    for syn in syntheses:
        for dim, entry in syn.skill_profile.items():
            if dim not in DIMENSION_BY_NAME:
                continue
            if entry.current_level == "unobserved":
                continue
            row = matrix.setdefault(dim, {})
            row[syn.project_short] = CrossProjectCell(
                level=entry.current_level,
                sessions=len(entry.evidence_sessions),
            )
    return matrix


# ---------------------------------------------------------------------------
# LLM anomaly detection
# ---------------------------------------------------------------------------


STAGE4_PROMPT = """You are looking for anomalies in a dimension-by-project matrix of
skill levels. You see one row per canonical skill dimension and one column per
project. Cells show {{level, sessions}} for projects where the skill was observed.

# Your job

Identify up to 7 anomalies that would interest a senior mentor reviewing this
person's work. An anomaly is one of:

  - A dimension where the person is much stronger in one project than the rest
    (e.g. strong Python in project A, beginner Python in project B).
  - A dimension where skill level is surprisingly uneven given sample size.
  - Two or more dimensions that tend to co-drop (e.g. "testing-discipline"
    and "debugging-systematic" are weakest in the same projects).
  - A dimension that's consistently low across all projects with enough
    evidence to call it a real gap rather than noise.

# Rules

1. Only cite projects whose sessions count >= 2 when flagging uneven signals.
2. For each anomaly, include a 1-sentence hypothesis about WHY -- but label it
   as a hypothesis, not a fact. Keep it short.
3. Skip anomalies where the evidence is too thin to mean much (one project
   with one session shouldn't drive a finding).
4. Only use canonical dimension names from the matrix.
5. Respond with a SINGLE JSON object:

{{
  "anomalies": [
    {{
      "observation": "Strong steering in demo-api, beginner in demo-ml",
      "dimensions_involved": ["python"],
      "evidence": [{{"project": "demo-api", "level": "strong"}},
                   {{"project": "demo-ml", "level": "beginner"}}],
      "hypothesis": "demo-ml may involve unfamiliar pytorch idioms the user hasn't internalized"
    }}
  ]
}}

# Project metadata

{metadata}

# Matrix (sparse; projects where level=unobserved are omitted from a row)

{matrix}
"""


def _matrix_payload(matrix: dict[str, dict[str, CrossProjectCell]]) -> str:
    return json.dumps(
        {
            dim: {proj: {"level": c.level, "sessions": c.sessions} for proj, c in row.items()}
            for dim, row in sorted(matrix.items())
        },
        indent=2,
        ensure_ascii=False,
    )


def _metadata_payload(syntheses: list[ProjectSynthesis]) -> str:
    rows = []
    for s in sorted(syntheses, key=lambda x: x.project_short):
        rows.append({
            "project_short": s.project_short,
            "session_count": s.session_count,
            "message_count": s.message_count,
            "first_session": s.first_session,
            "last_session": s.last_session,
            "what_it_is": s.dna.what_it_is,
        })
    return json.dumps(rows, indent=2, ensure_ascii=False)


def _input_fingerprint(syntheses: list[ProjectSynthesis]) -> str:
    """Hash project-level summaries so re-runs detect Stage 2 updates."""
    h = hashlib.sha256()
    for s in sorted(syntheses, key=lambda x: x.project_short):
        h.update(s.project_short.encode())
        h.update(b"|")
        h.update(str(s.session_count).encode())
        h.update(b"|")
        h.update(str(s.message_count).encode())
        h.update(b"|")
        h.update(s.last_session.encode())
        # Include the skill profile's level fingerprint too so a re-synthesis
        # with updated skill levels invalidates cross-project cache.
        for dim, e in sorted(s.skill_profile.items()):
            h.update(f"|{dim}:{e.current_level}:{e.peak_level}".encode())
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


@dataclass
class CrossProjectResult:
    cross: CrossProject | None
    cached: bool = False
    error: str | None = None


def build_cross_project(
    syntheses: list[ProjectSynthesis],
    provider: LLMProvider | None,
    *,
    model: str | None = None,
    force: bool = False,
) -> CrossProjectResult:
    """
    Build the cross-project view from all deep-analyzed projects.

    `provider=None` builds the matrix but leaves anomalies empty (no LLM).
    """
    if not syntheses:
        return CrossProjectResult(cross=None, error="no project syntheses provided")

    out_path = get_atlas_dir() / "cross_project.json"
    input_fp = _input_fingerprint(syntheses)

    if not force and out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if existing.get("_input_fingerprint") == input_fp:
                clean = {k: v for k, v in existing.items() if not k.startswith("_")}
                return CrossProjectResult(cross=CrossProject.from_dict(clean), cached=True)
        except (OSError, json.JSONDecodeError):
            pass

    matrix = build_matrix(syntheses)
    anomalies: list[CrossProjectAnomaly] = []

    if provider is not None and matrix:
        prompt = STAGE4_PROMPT.format(
            metadata=_metadata_payload(syntheses),
            matrix=_matrix_payload(matrix),
        )

        def _do_call() -> str:
            return provider.analyze(prompt, model=model)

        raw, err = call_with_retries(_do_call, max_attempts=3)
        if raw is not None:
            parsed = parse_json_lenient(raw) or {}
            raw_anoms = parsed.get("anomalies") or []
            for a in raw_anoms:
                if isinstance(a, dict) and a.get("observation"):
                    anomalies.append(CrossProjectAnomaly.from_dict(a))
        else:
            # Save error sidecar; still proceed with an empty anomaly list so
            # the matrix view works.
            (out_path.parent / "cross_project.error.txt").write_text(
                str(err), encoding="utf-8"
            )

    cross = CrossProject(
        schema_version=SCHEMA_VERSION,
        generated_at=datetime.now().isoformat(timespec="seconds"),
        dimension_matrix=matrix,
        anomalies=anomalies,
    )

    payload = cross.to_dict()
    payload["_input_fingerprint"] = input_fp
    payload["_provider"] = provider.info.name if provider else ""
    payload["_model"] = (model or (provider.info.default_model if provider else ""))
    atomic_write_json(out_path, payload)

    return CrossProjectResult(cross=cross, cached=False)


def load_cross_project() -> CrossProject | None:
    path = get_atlas_dir() / "cross_project.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        clean = {k: v for k, v in raw.items() if not k.startswith("_")}
        return CrossProject.from_dict(clean)
    except (OSError, json.JSONDecodeError):
        return None


__all__ = [
    "CrossProjectResult",
    "build_matrix",
    "build_cross_project",
    "load_cross_project",
]
