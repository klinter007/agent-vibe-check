"""
Extraction layer: turn parsed Cursor transcripts into the compact JSONL format
consumed by the analysis pipeline.

Public contract (kept stable so `interactive.py` / `analyze.py` don't need to change):

    extract_to_jsonl(output_path, project_filter=None) -> stats dict
    get_session_metadata() -> {session_id: {project, topic, mode, ...}}
    list_projects() -> {short_project_name: session_count}

Backed by `parser.py`, which reads JSONL transcripts at
`~/.cursor/projects/{folder}/agent-transcripts/{session}/*.jsonl`.

Compact JSONL output format (unchanged from legacy):
    {"session_start": "id", "project": "name", "topic": "...", "date": "YYYY-MM-DD"}
    {"r": "u", "t": "user text"}
    {"r": "a", "t": "assistant text"}
    {"session_end": "id"}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .parser import (
    Session,
    iter_all_projects,
)
from .paths import get_cursor_projects_dir
from . import composer_db
from . import sources


MIN_TEXT_LEN = 5


def short_project_name(folder: str) -> str:
    """
    Derive a human-readable short name from a Cursor project folder name.

    The folder name is a path with `/` replaced by `-`, so true disambiguation
    requires workspace.json. This is a best-effort heuristic that matches what
    the legacy code returned (the final path component).

    Examples:
        Users-alice-projects-my-app         -> my-app
        Users-alice-Desktop-side-project    -> side-project
        Users-alice                         -> home
        1772910391913                       -> 1772910391913 (numeric IDs kept as-is)
    """
    if folder.isdigit():
        return folder
    tokens = folder.split("-")
    if not tokens:
        return folder
    # Heuristic: known "project root" markers split the path
    for marker in ("projects", "Desktop", "Downloads", "Documents"):
        if marker in tokens:
            idx = tokens.index(marker)
            tail = tokens[idx + 1 :]
            if tail:
                return "-".join(tail)
    # Home folders: macOS "Users-<me>-...", Linux "home-<me>-...",
    # Windows "C-Users-<me>-...". Drop the user name from the short name.
    if len(tokens) >= 2 and tokens[0] in ("Users", "home"):
        return "home" if len(tokens) <= 2 else "-".join(tokens[2:])
    if len(tokens) >= 3 and len(tokens[0]) == 1 and tokens[1] == "Users":
        return "home" if len(tokens) <= 3 else "-".join(tokens[3:])
    return folder


def _assistant_text_for_jsonl(session: Session) -> list[tuple[str, str]]:
    """
    Convert a Session into a flat [(role, text), ...] list for compact JSONL.

    For assistant messages, we concatenate text blocks. Tool calls are
    summarized inline so downstream analysis still sees what tools ran.
    """
    out: list[tuple[str, str]] = []
    for msg in session.messages:
        if msg.role == "user":
            text = msg.clean_user_text
            if len(text) >= MIN_TEXT_LEN:
                out.append(("u", text))
        else:  # assistant
            parts: list[str] = []
            if msg.text_blocks:
                parts.append("\n".join(msg.text_blocks).strip())
            if msg.tool_calls:
                tool_summary = ", ".join(
                    tc.name for tc in msg.tool_calls if tc.name
                )
                if tool_summary:
                    parts.append(f"[tools: {tool_summary}]")
            text = "\n".join(p for p in parts if p).strip()
            if len(text) >= MIN_TEXT_LEN:
                out.append(("a", text))
    return out


def _session_start_date(session: Session) -> str:
    """Approximate session start date from file mtime (YYYY-MM-DD)."""
    return session.file_mtime.strftime("%Y-%m-%d")


def _session_topic(session: Session) -> str:
    """Use the first user query (truncated) as the session topic."""
    topic = session.first_user_query
    return topic[:80] if topic else ""


def iter_matching_sessions(project_filter: str | None) -> list[Session]:
    """
    Return all parsed sessions from every source the user enabled
    (Cursor JSONL + SQLite, Claude Code, Codex, Gemini CLI).

    Optionally filter to projects whose short name (or folder string) matches
    `project_filter` (case-insensitive substring match).
    """
    return list(sources.iter_sessions(project_filter))


def get_session_metadata() -> dict[str, dict]:
    """
    Return {session_id: {project, topic, mode, ...}} for every session
    discoverable across the enabled sources.
    """
    out: dict[str, dict] = {}
    for session in sources.iter_sessions(None):
        if session.session_id in out:
            continue
        if session.source == "composer_db":
            mode = "composer"
        elif session.is_subagent:
            mode = "subagent"
        else:
            mode = "agent"
        out[session.session_id] = {
            "project": short_project_name(session.project_folder),
            "topic": _session_topic(session),
            "mode": mode,
            "transcript_path": str(session.transcript_path) if session.transcript_path else None,
            "date": _session_start_date(session),
            "message_count": session.message_count,
            "source": session.source,
        }
    return out


def list_projects() -> dict[str, int]:
    """
    Return {short_project_name: session_count}, sorted descending by count,
    across the enabled sources.
    """
    counts = {short: st.sessions for short, st in project_stats().items()}
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


def top_n_projects(n: int = 5, min_sessions: int = 3) -> list[tuple[str, int]]:
    """
    Return the top N projects by session count, keeping only those with at
    least `min_sessions`. Used by the meta-report / atlas to focus on
    projects with enough conversation history to be worth analyzing.
    """
    ranked = [(name, c) for name, c in list_projects().items() if c >= min_sessions]
    return ranked[:n]


@dataclass
class ProjectStats:
    """
    Lightweight richness summary for a single project, aggregated across both
    JSONL and composer-DB sources. All counts are cheap approximations
    sufficient for filtering (we count JSONL file lines and DB bubble-header
    list lengths rather than fully parsing sessions).
    """
    short_name: str
    sessions: int = 0
    total_messages: int = 0
    max_session_messages: int = 0

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.sessions, self.total_messages, self.max_session_messages)


def _count_jsonl_lines(path: Path) -> int:
    """Cheap message-count approximation for a JSONL transcript. Counts non-empty lines."""
    try:
        with path.open("rb") as f:
            n = 0
            for line in f:
                if line.strip():
                    n += 1
            return n
    except OSError:
        return 0


def project_stats() -> dict[str, ProjectStats]:
    """
    Return {short_project_name: ProjectStats} merged across JSONL transcripts
    and composer DB sessions. Designed to be cheap (no full session parsing,
    no bubble fetching from the 8 GB global DB) so it can run every TUI start.

    On the author's machine this takes ~1 second for 111 projects / 680
    sessions - dominated by ~421 point lookups against the global DB for
    composer header lengths.
    """
    from . import composer_db
    from .paths import get_global_state_db

    stats: dict[str, ProjectStats] = {}

    def _bump(short: str, msg_count: int) -> None:
        st = stats.setdefault(short, ProjectStats(short_name=short))
        st.sessions += 1
        st.total_messages += msg_count
        if msg_count > st.max_session_messages:
            st.max_session_messages = msg_count

    use_cursor = "cursor" in sources.enabled_keys()

    # --- Claude Code / Codex / Gemini CLI: cached per-file counts. ---
    for _key, folder, n in sources.file_session_stats():
        _bump(short_project_name(folder), n)

    # --- Cursor JSONL side: walk files and line-count each. ---
    projects_dir = get_cursor_projects_dir()
    for project_dir in (iter_all_projects(projects_dir) if use_cursor else ()):
        short = short_project_name(project_dir.name)
        t = project_dir / "agent-transcripts"
        if not t.is_dir():
            continue
        for session_dir in t.iterdir():
            if not session_dir.is_dir():
                continue
            main = session_dir / f"{session_dir.name}.jsonl"
            if main.is_file():
                _bump(short, _count_jsonl_lines(main))
            # Subagent transcripts are not read (an agent typed them).

    # --- Composer DB side: one point query per composer for its header count. ---
    global_db = get_global_state_db()
    if use_cursor and global_db.is_file():
        import sqlite3
        try:
            conn = sqlite3.connect(f"file:{global_db}?mode=ro", uri=True, timeout=5.0)
        except sqlite3.Error:
            conn = None

        if conn is not None:
            try:
                for folder, composers in composer_db.list_db_projects().items():
                    short = short_project_name(folder)
                    for meta in composers:
                        cid = meta.get("composerId")
                        if not isinstance(cid, str):
                            continue
                        row = conn.execute(
                            "SELECT value FROM cursorDiskKV WHERE key = ?",
                            (f"composerData:{cid}",),
                        ).fetchone()
                        if not row or not row[0]:
                            continue
                        try:
                            data = json.loads(row[0], strict=False)
                        except (ValueError, TypeError):
                            continue
                        headers = data.get("fullConversationHeadersOnly") or []
                        n = len(headers) if isinstance(headers, list) else 0
                        _bump(short, n)
            finally:
                conn.close()

    return stats


# Richness rule for deep analysis. Either one real conversation (>= 10
# messages in a single session) OR enough aggregate content (>= 20 total
# messages across sessions). Empirically these are the cut-offs where the
# analyze.py prompt starts producing non-generic output.
DEFAULT_MIN_SESSION_MESSAGES = 10
DEFAULT_MIN_TOTAL_MESSAGES = 20


def qualifies_for_deep_analysis(
    st: ProjectStats,
    min_session_messages: int = DEFAULT_MIN_SESSION_MESSAGES,
    min_total_messages: int = DEFAULT_MIN_TOTAL_MESSAGES,
) -> bool:
    """Apply the richness rule. Either axis is sufficient."""
    if st.max_session_messages >= min_session_messages:
        return True
    if st.total_messages >= min_total_messages:
        return True
    return False


def qualifying_projects(
    stats: dict[str, ProjectStats] | None = None,
    min_session_messages: int = DEFAULT_MIN_SESSION_MESSAGES,
    min_total_messages: int = DEFAULT_MIN_TOTAL_MESSAGES,
) -> list[ProjectStats]:
    """Return projects that pass the richness filter, sorted by total_messages desc."""
    s = stats if stats is not None else project_stats()
    keep = [
        st for st in s.values()
        if qualifies_for_deep_analysis(st, min_session_messages, min_total_messages)
    ]
    keep.sort(key=lambda st: st.total_messages, reverse=True)
    return keep


def extract_to_jsonl(
    output_path: Path,
    project_filter: str | None = None,
) -> dict:
    """
    Extract conversations to compact JSONL for downstream analysis.

    Returns stats dict: {sessions, total_messages, user_messages,
    assistant_messages, output_path, file_size_bytes}.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sessions = iter_matching_sessions(project_filter)
    # Sort by start date for deterministic output
    sessions.sort(key=_session_start_date)

    total_messages = 0
    total_user = 0
    total_assistant = 0
    written_sessions = 0

    with output_path.open("w", encoding="utf-8") as f:
        for session in sessions:
            flat = _assistant_text_for_jsonl(session)
            if not flat:
                continue
            written_sessions += 1

            header = {
                "session_start": session.session_id,
                "project": short_project_name(session.project_folder),
                "topic": _session_topic(session),
                "date": _session_start_date(session),
            }
            if session.is_subagent:
                header["mode"] = "subagent"
                if session.parent_session_id:
                    header["parent_session_id"] = session.parent_session_id

            f.write(json.dumps(header, ensure_ascii=False) + "\n")

            for role, text in flat:
                f.write(
                    json.dumps({"r": role, "t": text}, ensure_ascii=False) + "\n"
                )
                total_messages += 1
                if role == "u":
                    total_user += 1
                else:
                    total_assistant += 1

            f.write(
                json.dumps({"session_end": session.session_id}, ensure_ascii=False)
                + "\n"
            )

    return {
        "sessions": written_sessions,
        "total_messages": total_messages,
        "user_messages": total_user,
        "assistant_messages": total_assistant,
        "output_path": str(output_path),
        "file_size_bytes": output_path.stat().st_size if output_path.exists() else 0,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
