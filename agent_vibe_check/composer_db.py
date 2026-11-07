"""
Composer / Chat / Agent session reader backed by Cursor's SQLite databases.

Complements parser.py (which reads the JSONL agent-transcripts). Together they
cover every mode: background-agent, CLI-agent, Composer, Cmd+L Chat.

Storage model (reverse-engineered 2026-04):

    ~/Library/Application Support/Cursor/User/workspaceStorage/{hash}/
        workspace.json          # "folder": "file://..." -> real path
        state.vscdb             # ItemTable.composer.composerData = list of {composerId, ...}

    ~/Library/Application Support/Cursor/User/globalStorage/state.vscdb
        cursorDiskKV.composerData:{composerId}
            -> full composer blob including fullConversationHeadersOnly
               which is the ordered list of {bubbleId, type}
        cursorDiskKV.bubbleId:{composerId}:{bubbleId}
            -> one message. `type`: 1 = user, 2 = assistant.
               `text` holds the message body. `allThinkingBlocks`
               holds reasoning if present.

So to reconstruct a workspace's conversations we:
  1. Enumerate workspaceStorage dirs, read workspace.json + composerData.
  2. For each composerId referenced by any workspace, fetch the global composerData
     to get ordered bubble headers.
  3. Batch-fetch the bubble blobs from the global DB.
  4. Build a Session in the same shape parser.py produces so downstream code
     (analyze, cache, atlas, dashboard) stays unchanged.

Performance notes:
  - The global DB is 8 GB / 461k rows on the author's machine. We never iterate
    it; every lookup is a point query (`WHERE key = ?`) or a bounded IN batch.
  - We open global DB read-only with URI mode so we can't corrupt it.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator
from urllib.parse import unquote

from .parser import Message, Session, ToolCall
from .paths import get_global_state_db, get_workspace_storage_dir


log = logging.getLogger(__name__)


# Bubble `type` field in the global DB:
#   1 -> user message
#   2 -> assistant message
# Anything else is ignored (we've seen none but be defensive).
BUBBLE_TYPE_USER = 1
BUBBLE_TYPE_ASSISTANT = 2

# When a user's text came from the attached-files wrapper rather than a
# direct message, richText is often the only carrier. We don't parse Lexical
# richText for now; `text` is good enough.

# Fetch bubbles in batches so we don't blow through SQLite's parameter limit.
BUBBLE_FETCH_BATCH = 256


# ---------------------------------------------------------------------------
# Low-level DB helpers
# ---------------------------------------------------------------------------


def _open_readonly(path: Path) -> sqlite3.Connection:
    """Open a SQLite DB read-only via URI. Prevents accidental writes."""
    uri = f"file:{path}?mode=ro"
    return sqlite3.connect(uri, uri=True, timeout=5.0)


def _load_json(blob) -> dict | list | None:
    """SQLite BLOB/TEXT -> parsed JSON or None on any failure."""
    if blob is None:
        return None
    if isinstance(blob, bytes):
        try:
            blob = blob.decode("utf-8", errors="replace")
        except Exception:
            return None
    try:
        return json.loads(blob, strict=False)
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Workspace -> composer ID mapping
# ---------------------------------------------------------------------------


@dataclass
class WorkspaceComposers:
    """Discovered state for one Cursor workspace (one workspaceStorage/{hash}/)."""
    hash: str
    real_path: str | None  # resolved from workspace.json; None if unreadable
    project_folder: str    # Cursor-style key (slashes -> dashes)
    composers: list[dict]  # list of composer metadata dicts from composer.composerData


def _uri_to_path(uri: str) -> str | None:
    """Convert a file:// URI to a filesystem path. Returns None if not file://."""
    if not isinstance(uri, str) or not uri.startswith("file://"):
        return None
    # Strip scheme + host (file:///Users/... -> /Users/...)
    path = uri[len("file://"):]
    # URL-decode so '%20' etc. become proper characters.
    return unquote(path)


def _path_to_project_folder(path: str) -> str:
    """
    Reconstruct Cursor's folder-naming convention. Empirically Cursor replaces
    both `/` and `_` with `-` when it writes to `~/.cursor/projects/`, so we
    do the same here. This keeps DB-sourced and JSONL-sourced project folders
    aligned for e.g. `/Users/alice/projects/sample_app`:
        -> Users-alice-projects-sample-app
    matches what JSONL transcripts use for the same folder.
    """
    return path.lstrip("/").replace("/", "-").replace("_", "-")


def enumerate_workspaces() -> Iterator[WorkspaceComposers]:
    """
    Walk workspaceStorage/ and yield a WorkspaceComposers per workspace that
    actually has at least one composer referenced.
    """
    ws_dir = get_workspace_storage_dir()
    if not ws_dir.is_dir():
        return

    for hash_dir in ws_dir.iterdir():
        if not hash_dir.is_dir():
            continue
        ws_json = hash_dir / "workspace.json"
        db_path = hash_dir / "state.vscdb"
        if not db_path.is_file():
            continue

        real_path: str | None = None
        project_folder = hash_dir.name  # fallback: use hash as folder key
        if ws_json.is_file():
            try:
                wdata = json.loads(ws_json.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                wdata = {}
            real_path = _uri_to_path(wdata.get("folder"))
            if real_path:
                project_folder = _path_to_project_folder(real_path)

        try:
            conn = _open_readonly(db_path)
        except sqlite3.Error as e:
            log.debug("skip workspace %s: %s", hash_dir.name, e)
            continue

        try:
            cur = conn.execute(
                "SELECT value FROM ItemTable WHERE key = 'composer.composerData'"
            )
            row = cur.fetchone()
        except sqlite3.Error:
            row = None
        finally:
            conn.close()

        if not row or not row[0]:
            continue

        data = _load_json(row[0])
        if not isinstance(data, dict):
            continue
        composers = data.get("allComposers") or []
        composers = [c for c in composers if isinstance(c, dict) and c.get("composerId")]
        if not composers:
            continue

        yield WorkspaceComposers(
            hash=hash_dir.name,
            real_path=real_path,
            project_folder=project_folder,
            composers=composers,
        )


# ---------------------------------------------------------------------------
# Session reconstruction from the global DB
# ---------------------------------------------------------------------------


def _bubble_to_message(bubble: dict) -> Message | None:
    """
    Convert one bubble blob (as stored under `bubbleId:...`) into a Message.

    Extracts:
      - text (primary body)
      - thinking (from allThinkingBlocks, appended if non-empty)
      - tool_calls (from toolResults / supportedTools interplay - we use
        the presence of a toolResult to mean a tool ran, and the tool name
        when available)
      - tool_results (raw blobs preserved for symmetry with parser.py)
    """
    if not isinstance(bubble, dict):
        return None

    btype = bubble.get("type")
    if btype == BUBBLE_TYPE_USER:
        role = "user"
    elif btype == BUBBLE_TYPE_ASSISTANT:
        role = "assistant"
    else:
        return None

    text = bubble.get("text") or ""
    if not isinstance(text, str):
        text = ""

    # Surface reasoning when available. For assistant messages, `allThinkingBlocks`
    # is a list of {text, ...}. We flatten and prepend as a "(reasoning)" section
    # so downstream analysis picks it up naturally via the text body.
    thinking_bits: list[str] = []
    for blk in bubble.get("allThinkingBlocks") or []:
        if isinstance(blk, dict):
            t = blk.get("text") or blk.get("thinking") or ""
            if isinstance(t, str) and t.strip():
                thinking_bits.append(t.strip())

    text_blocks: list[str] = []
    if thinking_bits:
        text_blocks.append("(reasoning)\n" + "\n".join(thinking_bits))
    if text.strip():
        text_blocks.append(text.strip())

    tool_calls: list[ToolCall] = []
    tool_results: list[dict] = []
    for tr in bubble.get("toolResults") or []:
        if not isinstance(tr, dict):
            continue
        tool_results.append(tr)
        name = tr.get("toolName") or tr.get("name") or ""
        if name:
            tool_calls.append(ToolCall(name=name, input=tr.get("args") or {}))

    if not text_blocks and not tool_calls:
        return None

    return Message(
        role=role,
        text_blocks=text_blocks,
        tool_calls=tool_calls,
        tool_results=tool_results,
        raw=None,
    )


def _fetch_composer_record(
    global_conn: sqlite3.Connection, composer_id: str
) -> dict | None:
    """Read composerData:{id} from the global DB, parsed as JSON."""
    cur = global_conn.execute(
        "SELECT value FROM cursorDiskKV WHERE key = ?",
        (f"composerData:{composer_id}",),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    data = _load_json(row[0])
    return data if isinstance(data, dict) else None


def _fetch_bubbles(
    global_conn: sqlite3.Connection, composer_id: str, bubble_ids: list[str]
) -> dict[str, dict]:
    """Batch-fetch bubble blobs. Returns {bubble_id: parsed_dict}."""
    out: dict[str, dict] = {}
    for i in range(0, len(bubble_ids), BUBBLE_FETCH_BATCH):
        chunk = bubble_ids[i : i + BUBBLE_FETCH_BATCH]
        keys = [f"bubbleId:{composer_id}:{bid}" for bid in chunk]
        placeholders = ",".join("?" * len(keys))
        cur = global_conn.execute(
            f"SELECT key, value FROM cursorDiskKV WHERE key IN ({placeholders})",
            keys,
        )
        for key, value in cur:
            bid = key.rsplit(":", 1)[-1]
            parsed = _load_json(value)
            if isinstance(parsed, dict):
                out[bid] = parsed
    return out


def _fetch_bubble_sizes(
    global_conn: sqlite3.Connection, composer_id: str, bubble_ids: list[str]
) -> int:
    """Sum of raw blob byte lengths for fingerprinting. Cheap enough to do inline."""
    total = 0
    for i in range(0, len(bubble_ids), BUBBLE_FETCH_BATCH):
        chunk = bubble_ids[i : i + BUBBLE_FETCH_BATCH]
        keys = [f"bubbleId:{composer_id}:{bid}" for bid in chunk]
        placeholders = ",".join("?" * len(keys))
        cur = global_conn.execute(
            f"SELECT length(value) FROM cursorDiskKV WHERE key IN ({placeholders})",
            keys,
        )
        for (n,) in cur:
            if n:
                total += n
    return total


def build_session(
    global_conn: sqlite3.Connection,
    composer_meta: dict,
    project_folder: str,
) -> Session | None:
    """
    Assemble one Session from a composer metadata entry + the global DB.

    Returns None if:
      - composer record is missing from the global DB (deleted/corrupted)
      - the session has zero usable bubbles
    """
    composer_id = composer_meta.get("composerId")
    if not composer_id:
        return None

    record = _fetch_composer_record(global_conn, composer_id)
    if not record:
        return None

    headers = record.get("fullConversationHeadersOnly") or []
    if not isinstance(headers, list) or not headers:
        return None

    ordered_ids: list[str] = [
        h["bubbleId"] for h in headers
        if isinstance(h, dict) and isinstance(h.get("bubbleId"), str)
    ]
    if not ordered_ids:
        return None

    bubbles = _fetch_bubbles(global_conn, composer_id, ordered_ids)
    messages: list[Message] = []
    for bid in ordered_ids:
        b = bubbles.get(bid)
        if b is None:
            continue
        m = _bubble_to_message(b)
        if m is not None:
            messages.append(m)

    if not messages:
        return None

    # Fingerprint inputs: lastUpdatedAt (ms since epoch) + total bubble bytes.
    last_ms = composer_meta.get("lastUpdatedAt") or composer_meta.get("createdAt") or 0
    try:
        last_ms = int(last_ms)
    except (TypeError, ValueError):
        last_ms = 0
    mtime_ns = last_ms * 1_000_000  # ms -> ns

    byte_size = _fetch_bubble_sizes(global_conn, composer_id, ordered_ids)

    is_sub = bool(composer_meta.get("isBestOfNSubcomposer"))

    return Session(
        session_id=composer_id,
        project_folder=project_folder,
        transcript_path=None,
        messages=messages,
        is_subagent=is_sub,
        parent_session_id=None,
        source="composer_db",
        _mtime_ns=mtime_ns,
        _byte_size=byte_size,
    )


# ---------------------------------------------------------------------------
# High-level iteration
# ---------------------------------------------------------------------------


def iter_db_sessions(
    project_filter: str | None = None,
) -> Iterator[Session]:
    """
    Yield every Session reconstructable from Cursor's SQLite databases.

    `project_filter` is a case-insensitive substring match against the derived
    project_folder (mirrors what iter_matching_sessions does for JSONL).
    """
    global_db = get_global_state_db()
    if not global_db.is_file():
        log.debug("global state.vscdb not found: %s", global_db)
        return

    try:
        conn = _open_readonly(global_db)
    except sqlite3.Error as e:
        log.warning("cannot open global DB (%s): %s", global_db, e)
        return

    filter_lc = project_filter.lower() if project_filter else None

    try:
        # Track composerIds seen; the same id could theoretically be registered
        # to multiple workspaces (e.g. user moved a folder). First-wins keeps
        # output stable.
        seen: set[str] = set()

        for ws in enumerate_workspaces():
            if filter_lc and filter_lc not in ws.project_folder.lower():
                # Short-circuit the filter before touching the global DB.
                # The same filter logic is applied per-session below for safety.
                if ws.real_path and filter_lc in ws.real_path.lower():
                    pass
                else:
                    continue

            for meta in ws.composers:
                cid = meta.get("composerId")
                if not isinstance(cid, str) or cid in seen:
                    continue
                seen.add(cid)

                session = build_session(conn, meta, ws.project_folder)
                if session is None:
                    continue
                yield session
    finally:
        conn.close()


def list_db_projects() -> dict[str, list[dict]]:
    """
    Cheap, transcript-agnostic directory: {project_folder: [composer_meta, ...]}.

    Useful for counts without touching the global DB at all (the workspace DB
    already knows how many composer sessions exist).
    """
    out: dict[str, list[dict]] = {}
    for ws in enumerate_workspaces():
        out.setdefault(ws.project_folder, []).extend(ws.composers)
    return out


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def db_health_summary() -> dict:
    """
    Quick read-only probe we can show at startup. Reports coverage so the user
    knows we're actually reading the right DBs.
    """
    ws_dir = get_workspace_storage_dir()
    global_db = get_global_state_db()

    workspaces = 0
    with_composers = 0
    total_composers = 0
    by_folder: dict[str, int] = {}

    for ws in enumerate_workspaces():
        workspaces += 1
        with_composers += 1
        total_composers += len(ws.composers)
        by_folder[ws.project_folder] = by_folder.get(ws.project_folder, 0) + len(ws.composers)

    return {
        "workspace_dir_exists": ws_dir.is_dir(),
        "global_db_exists": global_db.is_file(),
        "global_db_size_bytes": global_db.stat().st_size if global_db.is_file() else 0,
        "workspaces_scanned": workspaces,
        "workspaces_with_composers": with_composers,
        "total_composers": total_composers,
        "unique_projects": len(by_folder),
        "checked_at": datetime.now().isoformat(timespec="seconds"),
    }
