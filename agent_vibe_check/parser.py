"""
Canonical JSONL parser for Cursor agent transcripts (April 2026 format).

Replaces the legacy extract.py + conversation_extractor.py duo. Works against
the current on-disk format documented by investigation:

    ~/.cursor/projects/{project-folder}/agent-transcripts/{session-uuid}/{session-uuid}.jsonl
    ~/.cursor/projects/{project-folder}/agent-transcripts/{session-uuid}/subagents/{uuid}.jsonl

Each JSONL line is one message object:

    {"role": "user" | "assistant", "message": {"content": [ ...blocks... ]}}

Content blocks are Anthropic-style:

    {"type": "text", "text": "..."}
    {"type": "tool_use", "name": "Shell", "input": {...}}
    {"type": "tool_result", "content": [...], "tool_use_id": "..."}

Note: the legacy `thinking` top-level field no longer exists. Reasoning prose
is embedded inside assistant `text` blocks (typically before a `tool_use`).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


USER_QUERY_RE = re.compile(r"<user_query>\s*(.*?)\s*</user_query>", re.DOTALL)
WRAPPER_TAG_RE = re.compile(
    r"<(user_query|uploaded_documents|manually_attached_skills|attached_files|"
    r"system_reminder|open_and_recently_viewed_files|agent_transcripts|"
    r"git_status|user_info|rules)>.*?</\1>",
    re.DOTALL,
)


@dataclass
class ToolCall:
    """A single tool_use block from an assistant message."""
    name: str
    input: dict

    @classmethod
    def from_block(cls, block: dict) -> "ToolCall":
        return cls(name=block.get("name", ""), input=block.get("input", {}) or {})


@dataclass
class Message:
    """One transcript line, normalized."""
    role: str  # "user" or "assistant"
    text_blocks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    raw: dict | None = None

    @property
    def text(self) -> str:
        """All text content concatenated."""
        return "\n".join(self.text_blocks).strip()

    @property
    def clean_user_text(self) -> str:
        """
        For user messages: extract the actual query from inside <user_query> tags
        if present, otherwise strip known system wrappers.
        """
        raw = self.text
        m = USER_QUERY_RE.search(raw)
        if m:
            return m.group(1).strip()
        return WRAPPER_TAG_RE.sub("", raw).strip()


@dataclass
class Session:
    """
    One chat session. Originally 1-to-1 with a JSONL transcript file; now also
    used for sessions reconstructed from Cursor's SQLite DBs (`source="composer_db"`),
    so `transcript_path` may be None. Fingerprint fields (`_mtime_ns`, `_byte_size`)
    are precomputed at parse time so downstream code doesn't care about the source.
    """
    session_id: str
    project_folder: str  # e.g. "Users-alice-projects-my-app"
    transcript_path: Path | None = None
    messages: list[Message] = field(default_factory=list)
    is_subagent: bool = False
    parent_session_id: str | None = None
    source: str = "jsonl"  # "jsonl" | "composer_db" | "claude_code" | "codex" | "gemini_cli"
    cwd: str | None = None          # real working directory when the source records it
    started_at: datetime | None = None
    # Precomputed for cache fingerprinting. Defaults get overwritten by the
    # factories (parse_transcript for jsonl, composer_db.load_session for DB).
    _mtime_ns: int = 0
    _byte_size: int = 0

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def user_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "user"]

    @property
    def assistant_messages(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]

    @property
    def first_user_query(self) -> str:
        """Best single-line summary of what the session was about."""
        for m in self.user_messages:
            t = m.clean_user_text
            if t:
                return t.split("\n", 1)[0][:200]
        return ""

    @property
    def file_mtime(self) -> datetime:
        if self._mtime_ns:
            return datetime.fromtimestamp(self._mtime_ns / 1e9)
        if self.transcript_path is not None:
            return datetime.fromtimestamp(self.transcript_path.stat().st_mtime)
        return datetime.fromtimestamp(0)

    @property
    def byte_size(self) -> int:
        if self._byte_size:
            return self._byte_size
        if self.transcript_path is not None:
            try:
                return self.transcript_path.stat().st_size
            except OSError:
                return 0
        return 0

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [tc for m in self.assistant_messages for tc in m.tool_calls]


def parse_line(line: str) -> Message | None:
    """Parse a single JSONL line. Returns None for unparseable/empty lines."""
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line, strict=False)
    except json.JSONDecodeError:
        return None

    role = obj.get("role")
    if role not in ("user", "assistant"):
        return None

    msg = obj.get("message") or {}
    content = msg.get("content")

    text_blocks: list[str] = []
    tool_calls: list[ToolCall] = []
    tool_results: list[dict] = []

    if isinstance(content, str):
        text_blocks.append(content)
    elif isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "text":
                t = block.get("text", "")
                if t:
                    text_blocks.append(t)
            elif btype == "tool_use":
                tool_calls.append(ToolCall.from_block(block))
            elif btype == "tool_result":
                tool_results.append(block)

    return Message(
        role=role,
        text_blocks=text_blocks,
        tool_calls=tool_calls,
        tool_results=tool_results,
        raw=obj,
    )


def parse_transcript(path: Path, project_folder: str | None = None) -> Session | None:
    """
    Parse one JSONL transcript file into a Session.
    Returns None if the file is empty or totally unparseable.
    """
    session_id = path.stem
    is_subagent = path.parent.name == "subagents"
    parent_session_id = path.parent.parent.name if is_subagent else None

    if project_folder is None:
        # .../projects/{project-folder}/agent-transcripts/{session}/[subagents/]{uuid}.jsonl
        project_folder = _infer_project_folder(path)

    messages: list[Message] = []
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = parse_line(line)
                if m is not None:
                    messages.append(m)
    except OSError:
        return None

    if not messages:
        return None

    try:
        st = path.stat()
        mtime_ns, byte_size = st.st_mtime_ns, st.st_size
    except OSError:
        mtime_ns, byte_size = 0, 0

    return Session(
        session_id=session_id,
        project_folder=project_folder,
        transcript_path=path,
        messages=messages,
        is_subagent=is_subagent,
        parent_session_id=parent_session_id,
        source="jsonl",
        _mtime_ns=mtime_ns,
        _byte_size=byte_size,
    )


def _infer_project_folder(path: Path) -> str:
    """Walk up from a transcript path to find the project folder name."""
    for parent in path.parents:
        if parent.parent.name == "projects":
            return parent.name
    return ""


def iter_project_transcripts(project_dir: Path) -> Iterator[Path]:
    """
    Yield all transcript JSONL files for a single project directory.
    Covers both main sessions and subagent sessions.
    """
    transcripts_dir = project_dir / "agent-transcripts"
    if not transcripts_dir.is_dir():
        return
    for session_dir in transcripts_dir.iterdir():
        if not session_dir.is_dir():
            continue
        main = session_dir / f"{session_dir.name}.jsonl"
        if main.is_file():
            yield main
        subagents_dir = session_dir / "subagents"
        if subagents_dir.is_dir():
            for sub in subagents_dir.glob("*.jsonl"):
                yield sub


def parse_project(project_dir: Path) -> list[Session]:
    """Parse all transcripts for one project directory."""
    sessions: list[Session] = []
    project_folder = project_dir.name
    for path in iter_project_transcripts(project_dir):
        s = parse_transcript(path, project_folder=project_folder)
        if s is not None:
            sessions.append(s)
    return sessions


def iter_all_projects(cursor_projects_dir: Path) -> Iterator[Path]:
    """Yield project directories that actually contain a non-empty agent-transcripts folder."""
    if not cursor_projects_dir.is_dir():
        return
    for project_dir in cursor_projects_dir.iterdir():
        if not project_dir.is_dir():
            continue
        t = project_dir / "agent-transcripts"
        if t.is_dir() and any(t.iterdir()):
            yield project_dir
