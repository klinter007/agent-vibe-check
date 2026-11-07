"""
Project Atlas: a searchable one-page index of every project you've worked on
in Cursor. Answers the question "what folder was that thing I built six months
ago and what did it do?".

For each project with enough conversation history, we collect a handful of
sample user queries + folder metadata, then ask the LLM to classify and
describe it in one sentence. The result is a single markdown file you can
grep or scroll.
"""

from __future__ import annotations

from .redact import strip_secrets

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .extract import (
    iter_matching_sessions,
    project_stats,
    qualifies_for_deep_analysis,
    short_project_name,
)
from .parser import Session
from .paths import get_atlas_dir, get_workspace_storage_dir
from .providers import LLMProvider


ATLAS_FILE = "ATLAS.md"
ATLAS_JSON = "atlas.json"

MAX_SAMPLE_QUERIES = 5
BATCH_SIZE = 12


ATLAS_PROMPT = """You are building an index card for each of the user's coding projects.

For each project below you get:
- folder: the raw Cursor folder name (path-like string)
- session_count: how many AI chat sessions happened there
- sample_queries: a handful of first-user-queries drawn from those sessions

Your job: for each project, return a concise classification and one-sentence
description. Base it strictly on what the queries reveal. If the project is
clearly incidental (e.g. one-off script, no coherent goal), say so.

Return a single JSON object with this shape, no prose around it:

{
  "projects": [
    {
      "folder": "<verbatim folder string from input>",
      "category": "<short tag: e.g. 'web app', 'ml infra', 'cli tool', 'research', 'one-off script', 'unknown'>",
      "description": "<one sentence, concrete, no fluff>",
      "tech_hints": ["<language, framework, or tool names you can infer>", ...]
    },
    ...
  ]
}

Projects:
<projects>
"""


@dataclass
class AtlasEntry:
    short_name: str
    folder: str
    session_count: int
    message_count: int
    max_session_messages: int
    last_active: str
    richness: str = "sparse"  # "deep" if passes the richness rule, else "sparse"
    sample_queries: list[str] = field(default_factory=list)
    real_path: str | None = None
    category: str = "unknown"
    description: str = ""
    tech_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "short_name": self.short_name,
            "folder": self.folder,
            "session_count": self.session_count,
            "message_count": self.message_count,
            "max_session_messages": self.max_session_messages,
            "last_active": self.last_active,
            "richness": self.richness,
            "sample_queries": self.sample_queries,
            "real_path": self.real_path,
            "category": self.category,
            "description": self.description,
            "tech_hints": self.tech_hints,
        }


# ---------------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------------


def _build_workspace_map() -> dict[str, str]:
    """
    Map Cursor workspace hashes -> real filesystem folder by reading each
    workspaceStorage/{hash}/workspace.json. Cursor's folder naming is lossy
    (slashes -> dashes), so this is the authoritative source for real paths.
    """
    ws_dir = get_workspace_storage_dir()
    result: dict[str, str] = {}
    if not ws_dir.is_dir():
        return result

    for hash_dir in ws_dir.iterdir():
        if not hash_dir.is_dir():
            continue
        wsj = hash_dir / "workspace.json"
        if not wsj.is_file():
            continue
        try:
            data = json.loads(wsj.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        folder_uri = data.get("folder")
        if isinstance(folder_uri, str) and folder_uri.startswith("file://"):
            real_path = folder_uri[len("file://"):]
            # Reconstruct the Cursor folder-name convention: strip leading
            # slash, replace '/' with '-'. This gives us a key that matches
            # ~/.cursor/projects/ subfolders.
            # Cursor's folder key also replaces `_` with `-` (lossy); match
            # that so JSONL-sourced project folders look up the same real path.
            key = real_path.lstrip("/").replace("/", "-").replace("_", "-")
            result[key] = real_path
    return result


def _build_candidate(
    project_folder: str,
    sessions: list[Session],
    ws_map: dict[str, str],
    stats_by_short: dict,
) -> AtlasEntry | None:
    if not sessions:
        return None
    total_msgs = sum(s.message_count for s in sessions)
    max_msgs = max(s.message_count for s in sessions)
    last_active = max(s.file_mtime for s in sessions).strftime("%Y-%m-%d")
    short = short_project_name(project_folder)

    # Pick sample queries from different sessions for diversity.
    queries: list[str] = []
    for s in sorted(sessions, key=lambda s: s.file_mtime, reverse=True):
        # This text goes to the LLM provider: drop keys and tokens first.
        q = strip_secrets(s.first_user_query)
        if q and q not in queries:
            queries.append(q)
        if len(queries) >= MAX_SAMPLE_QUERIES:
            break

    # Richness tier: reuse the same rule as the batch-analysis filter. Prefer
    # the project-wide stats (which aggregate across every source for this
    # short name) over this folder's slice, because the badge is a property of
    # the project as a whole, not of one folder key.
    agg = stats_by_short.get(short)
    if agg is not None and qualifies_for_deep_analysis(agg):
        richness = "deep"
    elif agg is None:
        # Fallback: compute from this folder's sessions alone.
        fake = type("T", (), {"sessions": len(sessions),
                              "total_messages": total_msgs,
                              "max_session_messages": max_msgs})()
        richness = "deep" if qualifies_for_deep_analysis(fake) else "sparse"
    else:
        richness = "sparse"

    return AtlasEntry(
        short_name=short,
        folder=project_folder,
        session_count=len(sessions),
        message_count=total_msgs,
        max_session_messages=max_msgs,
        last_active=last_active,
        richness=richness,
        sample_queries=queries,
        real_path=ws_map.get(project_folder),
    )


def collect_candidates(min_sessions: int = 1) -> list[AtlasEntry]:
    """
    Collect one AtlasEntry per project folder across BOTH data sources
    (JSONL agent transcripts + Composer/Chat SQLite DBs).

    Previously this walked `~/.cursor/projects/` only, which silently dropped
    DB-only projects (~75% of projects on a Cmd+L-heavy install). Now we
    enumerate sessions from `iter_matching_sessions(None)` and group them by
    `session.project_folder`, which is populated consistently by both sources.
    """
    ws_map = _build_workspace_map()
    stats_by_short = project_stats()

    # Group every session (both sources) by its project_folder key.
    by_folder: dict[str, list[Session]] = {}
    for session in iter_matching_sessions(None):
        by_folder.setdefault(session.project_folder, []).append(session)

    candidates: list[AtlasEntry] = []
    for folder, sessions in by_folder.items():
        if len(sessions) < min_sessions:
            continue
        c = _build_candidate(folder, sessions, ws_map, stats_by_short)
        if c is not None:
            candidates.append(c)

    # Sort: deep projects first, then by session count.
    candidates.sort(key=lambda e: (e.richness != "deep", -e.session_count))
    return candidates


# ---------------------------------------------------------------------------
# LLM enrichment
# ---------------------------------------------------------------------------


def _prompt_payload(entries: list[AtlasEntry]) -> str:
    slim = [
        {
            "folder": e.folder,
            "session_count": e.session_count,
            "sample_queries": e.sample_queries,
        }
        for e in entries
    ]
    return json.dumps(slim, indent=2, ensure_ascii=False)


def _parse_classification(raw: str) -> dict[str, dict]:
    """Return {folder: {category, description, tech_hints}} from the LLM response."""
    from .pipeline.llm_utils import parse_json_lenient

    parsed = parse_json_lenient(raw)
    if not parsed or not isinstance(parsed.get("projects"), list):
        return {}

    out: dict[str, dict] = {}
    for item in parsed["projects"]:
        folder = item.get("folder")
        if not isinstance(folder, str):
            continue
        out[folder] = {
            "category": str(item.get("category") or "unknown"),
            "description": str(item.get("description") or ""),
            "tech_hints": [str(x) for x in (item.get("tech_hints") or [])],
        }
    return out


def classify(entries: list[AtlasEntry], provider: LLMProvider) -> None:
    """
    Mutate entries in place, filling in category/description/tech_hints by
    batching LLM calls (BATCH_SIZE projects per call).
    """
    if not entries:
        return

    for start in range(0, len(entries), BATCH_SIZE):
        batch = entries[start : start + BATCH_SIZE]
        prompt = ATLAS_PROMPT.replace("<projects>", _prompt_payload(batch))
        raw = provider.analyze(prompt)
        classifications = _parse_classification(raw)
        for entry in batch:
            c = classifications.get(entry.folder, {})
            entry.category = c.get("category") or "unknown"
            entry.description = c.get("description") or ""
            entry.tech_hints = c.get("tech_hints") or []


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_atlas(entries: list[AtlasEntry], out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write ATLAS.md (human) and atlas.json (machine). Return both paths."""
    out_dir = out_dir or get_atlas_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Markdown
    deep_count = sum(1 for e in entries if e.richness == "deep")
    md_lines: list[str] = [
        "# Project Atlas",
        "",
        f"> Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} - "
        f"{len(entries)} projects ({deep_count} deep, {len(entries) - deep_count} sparse)",
        "",
        "_One-line index of every Cursor project you've touched. Grep this file "
        "when you can't remember where a thing lived._",
        "",
        "**Legend:** `[deep]` = substantive history worth revisiting; "
        "`[sparse]` = one-shot or throwaway.",
        "",
    ]

    # Group by category for navigation.
    by_cat: dict[str, list[AtlasEntry]] = {}
    for e in entries:
        by_cat.setdefault(e.category or "unknown", []).append(e)

    for cat in sorted(by_cat.keys()):
        md_lines.append(f"## {cat}")
        md_lines.append("")
        # Within a category: deep projects first, then by session count.
        for e in sorted(
            by_cat[cat],
            key=lambda e: (e.richness != "deep", -e.session_count),
        ):
            badge = "[deep]" if e.richness == "deep" else "[sparse]"
            loc = f"`{e.real_path}`" if e.real_path else f"_{e.folder}_"
            desc = e.description or "(no description)"
            tech = ", ".join(e.tech_hints) if e.tech_hints else ""
            meta = (
                f"{e.session_count} sessions, {e.message_count:,} msgs "
                f"(longest {e.max_session_messages:,}), last {e.last_active}"
            )
            md_lines.append(f"### {badge} {e.short_name}")
            md_lines.append(f"- {desc}")
            md_lines.append(f"- {loc}")
            md_lines.append(f"- {meta}")
            if tech:
                md_lines.append(f"- tech: {tech}")
            md_lines.append("")
        md_lines.append("")

    md_path = out_dir / ATLAS_FILE
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    json_path = out_dir / ATLAS_JSON
    json_path.write_text(
        json.dumps({"generated_at": datetime.now().isoformat(),
                    "entries": [e.to_dict() for e in entries]},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return md_path, json_path


def build_atlas(provider: LLMProvider, min_sessions: int = 1) -> tuple[Path, Path, list[AtlasEntry]]:
    """End-to-end: collect -> classify -> write. Returns (md, json, entries)."""
    entries = collect_candidates(min_sessions=min_sessions)
    classify(entries, provider)
    md, js = write_atlas(entries)
    return md, js, entries
