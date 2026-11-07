"""
Shared pieces for agent-history sources.

Every source (Cursor, Claude Code, Codex CLI, Gemini CLI) produces the same
`parser.Session` shape so the pipeline never needs to know which tool a
session came from. Project identity is the real working directory, turned
into a Cursor-style folder slug so one project worked on in two tools
merges into one project.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Protocol

from ..parser import Message, Session


# Working directories that are never a real project: temp dirs, scratchpads,
# OS caches. Sessions started there are usually automation, not a human.
_NOISE_PREFIXES = ("/private/tmp", "/tmp", "/var/folders", "/private/var/folders")


def slug_from_cwd(cwd: str) -> str:
    """
    Turn an absolute path into the Cursor-style project folder slug.

        /Users/alice/code_projects/my.app -> Users-alice-code-projects-my-app
    """
    return re.sub(r"[^A-Za-z0-9]+", "-", cwd).strip("-")


def parse_iso(value: str | None):
    """Parse an ISO 8601 timestamp (with optional Z) into a naive local datetime."""
    if not value or not isinstance(value, str):
        return None
    from datetime import datetime
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone().replace(tzinfo=None)
    return dt


def is_noise_cwd(cwd: str | None) -> bool:
    if not cwd:
        return False
    return cwd.startswith(_NOISE_PREFIXES) or "/scratchpad" in cwd


@dataclass
class SourceInfo:
    """What a source found on this machine. Used by onboarding."""
    key: str                 # stable id, e.g. "claude_code"
    label: str               # human name, e.g. "Claude Code"
    vendor: str              # e.g. "Anthropic"
    docs_url: str            # where the user can read about the tool
    roots: list[Path] = field(default_factory=list)   # dirs we would read
    session_count: int = 0

    @property
    def found(self) -> bool:
        return self.session_count > 0


class Source(Protocol):
    key: str

    def info(self) -> SourceInfo: ...

    def iter_sessions(self) -> Iterator[Session]: ...


def make_session(
    *,
    session_id: str,
    cwd: str | None,
    fallback_folder: str,
    path: Path,
    messages: list[Message],
    source: str,
    started_at: str | None = None,
) -> Session | None:
    """Build a Session with fingerprint fields filled from the file on disk."""
    if not messages:
        return None
    try:
        st = path.stat()
        mtime_ns, byte_size = st.st_mtime_ns, st.st_size
    except OSError:
        mtime_ns, byte_size = 0, 0
    return Session(
        session_id=session_id,
        project_folder=slug_from_cwd(cwd) if cwd else fallback_folder,
        transcript_path=path,
        messages=messages,
        source=source,
        cwd=cwd,
        started_at=parse_iso(started_at),
        _mtime_ns=mtime_ns,
        _byte_size=byte_size,
    )
