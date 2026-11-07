"""
Cursor: agent JSONL transcripts plus Composer/Chat sessions from SQLite.

Thin wrapper over the original readers in `parser.py` and `composer_db.py`.
"""

from __future__ import annotations

from typing import Iterator

from .. import composer_db
from ..parser import Session, iter_all_projects, parse_project
from ..paths import get_cursor_projects_dir, get_workspace_storage_dir
from .base import SourceInfo

SOURCE = "cursor"


class CursorSource:
    key = SOURCE

    def info(self) -> SourceInfo:
        projects_dir = get_cursor_projects_dir()
        n = 0
        for project_dir in iter_all_projects(projects_dir):
            t = project_dir / "agent-transcripts"
            n += sum(1 for d in t.iterdir() if d.is_dir())
        try:
            n += sum(len(v) for v in composer_db.list_db_projects().values())
        except Exception:  # noqa: BLE001 - a broken DB must not break onboarding
            pass
        return SourceInfo(
            key=self.key, label="Cursor", vendor="Anysphere",
            docs_url="https://cursor.com/docs",
            roots=[projects_dir, get_workspace_storage_dir()], session_count=n,
        )

    def iter_sessions(self) -> Iterator[Session]:
        for project_dir in iter_all_projects(get_cursor_projects_dir()):
            # Subagent transcripts are typed by the parent agent, not a person.
            yield from (s for s in parse_project(project_dir) if not s.is_subagent)
        yield from composer_db.iter_db_sessions(project_filter=None)
