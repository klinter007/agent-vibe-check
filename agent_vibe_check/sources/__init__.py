"""
Agent-history sources: Cursor, Claude Code, Codex (OpenAI), Gemini CLI.

The user picks which sources the tool may read during onboarding; the choice
lives in config.json under "sources" (list of keys). Nothing outside the
enabled sources is ever read.

Public API:
    all_sources()                 -> every known source
    detect()                      -> [SourceInfo] for onboarding
    enabled_keys() / set_enabled_keys(keys)
    iter_sessions(project_filter) -> Session iterator over enabled sources
    session_stats()               -> cheap per-session (folder, msg_count) via cache
"""

from __future__ import annotations

import json
from typing import Iterator

from ..config import get_setting, set_setting
from ..parser import Session
from .base import SourceInfo, is_noise_cwd, slug_from_cwd
from .claude_code import ClaudeCodeSource
from .codex import CodexSource
from .cursor import CursorSource
from .gemini_cli import GeminiCliSource

SOURCE_LABELS = {
    "cursor": "Cursor",
    "jsonl": "Cursor",
    "composer_db": "Cursor",
    "claude_code": "Claude Code",
    "codex": "Codex",
    "gemini_cli": "Gemini CLI",
}

# Bump a source's revision when its parser changes what the LLM sees; Stage 1
# then re-reviews that source's sessions (other sources stay cached).
SOURCE_REVISIONS = {
    "claude_code": 2,   # drop injected task notifications / compaction summaries
    "codex": 2,         # subagent rollouts detected from the first session_meta
}


def source_revision(session_source: str) -> int:
    return SOURCE_REVISIONS.get(session_source, 1)


# Before onboarding exists (or for old configs) only Cursor is read, which is
# what earlier versions of this tool did.
DEFAULT_ENABLED = ["cursor"]


def all_sources() -> list:
    return [CursorSource(), ClaudeCodeSource(), CodexSource(), GeminiCliSource()]


def detect() -> list[SourceInfo]:
    return [s.info() for s in all_sources()]


def onboarding_done() -> bool:
    return isinstance(get_setting("sources"), list)


def enabled_keys() -> list[str]:
    val = get_setting("sources")
    if isinstance(val, list):
        return [k for k in val if isinstance(k, str)]
    return list(DEFAULT_ENABLED)


def set_enabled_keys(keys: list[str]) -> None:
    known = {s.key for s in all_sources()}
    set_setting("sources", [k for k in keys if k in known])


def enabled_sources() -> list:
    keys = set(enabled_keys())
    return [s for s in all_sources() if s.key in keys]


def tool_label(session_source: str) -> str:
    return SOURCE_LABELS.get(session_source, session_source)


def _matches(session: Session, filter_lc: str | None) -> bool:
    if not filter_lc:
        return True
    from ..extract import short_project_name
    return (filter_lc in short_project_name(session.project_folder).lower()
            or filter_lc in session.project_folder.lower())


def iter_sessions(project_filter: str | None = None) -> Iterator[Session]:
    filter_lc = project_filter.lower() if project_filter else None
    for src in enabled_sources():
        for s in src.iter_sessions():
            if _matches(s, filter_lc):
                yield s


_STATS_CACHE = "source_stats.json"


def file_session_stats() -> Iterator[tuple[str, str, int]]:
    """
    Yield (source_key, project_folder, message_count) for every session in
    the enabled file-based sources (all except Cursor, which has its own
    cheap counters). Results are cached by (path, mtime, size), so only new
    or changed transcripts get parsed. Skipped sessions are cached too.
    """
    from ..pipeline.llm_utils import atomic_write_json
    from ..paths import get_cache_dir

    cache_path = get_cache_dir() / _STATS_CACHE
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if not isinstance(cache, dict):
            cache = {}
    except (OSError, ValueError):
        cache = {}
    fresh: dict[str, list] = {}
    changed = False

    for src in enabled_sources():
        if not hasattr(src, "files"):
            continue
        for path in src.files():
            try:
                st = path.stat()
            except OSError:
                continue
            key = str(path)
            hit = cache.get(key)
            if hit and hit[0] == st.st_mtime_ns and hit[1] == st.st_size:
                entry = hit
            else:
                s = src.parse(path)
                entry = [st.st_mtime_ns, st.st_size,
                         s.project_folder if s else None, s.message_count if s else 0]
                changed = True
            fresh[key] = entry
            if entry[2]:
                yield src.key, entry[2], entry[3]

    if changed or len(fresh) != len(cache):
        atomic_write_json(cache_path, fresh)


__all__ = [
    "SourceInfo", "all_sources", "detect", "enabled_keys", "enabled_sources",
    "iter_sessions", "onboarding_done", "set_enabled_keys", "slug_from_cwd",
    "is_noise_cwd", "tool_label", "source_revision",
]
