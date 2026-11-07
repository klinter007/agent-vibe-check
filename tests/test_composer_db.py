"""
Tests for composer_db.py: synthetic SQLite fixtures for workspace + global DBs,
then assert we reconstruct Sessions correctly.

All tests are fully offline - no dependence on the user's real Cursor install.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent_vibe_check import composer_db


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _init_db(path: Path) -> sqlite3.Connection:
    """Create a fresh Cursor-shaped SQLite DB at `path`."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    conn.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
    return conn


def _put_item(conn: sqlite3.Connection, key: str, value: dict | list) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO ItemTable (key, value) VALUES (?, ?)",
        (key, json.dumps(value).encode("utf-8")),
    )


def _put_disk_kv(conn: sqlite3.Connection, key: str, value: dict | list) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (key, json.dumps(value).encode("utf-8")),
    )


def _build_fake_cursor_install(root: Path) -> tuple[Path, Path]:
    """
    Build a miniature Cursor install under `root`:
      - one workspace referencing two composers
      - the global DB containing both composers' data
    Returns (workspace_storage_dir, global_storage_dir).
    """
    ws_dir = root / "workspaceStorage"
    ws_dir.mkdir(parents=True)
    gs_dir = root / "globalStorage"
    gs_dir.mkdir(parents=True)

    # workspaceStorage/abc123/
    hash_dir = ws_dir / "abc123"
    hash_dir.mkdir()
    (hash_dir / "workspace.json").write_text(
        json.dumps({"folder": "file:///Users/me/projects/demo_app"}),
        encoding="utf-8",
    )

    ws_conn = _init_db(hash_dir / "state.vscdb")
    _put_item(ws_conn, "composer.composerData", {
        "allComposers": [
            {"composerId": "comp-alpha", "name": "alpha", "lastUpdatedAt": 1_700_000_000_000},
            {"composerId": "comp-beta",  "name": "beta",  "lastUpdatedAt": 1_700_100_000_000},
            {"composerId": "comp-ghost", "name": "ghost", "lastUpdatedAt": 1_700_200_000_000},
            # "comp-ghost" intentionally has NO record in the global DB.
        ],
    })
    ws_conn.commit()
    ws_conn.close()

    # globalStorage/state.vscdb
    gconn = _init_db(gs_dir / "state.vscdb")

    # --- comp-alpha: 3 bubbles, user + assistant + user ---
    _put_disk_kv(gconn, "composerData:comp-alpha", {
        "composerId": "comp-alpha",
        "fullConversationHeadersOnly": [
            {"bubbleId": "b1", "type": 1},
            {"bubbleId": "b2", "type": 2},
            {"bubbleId": "b3", "type": 1},
        ],
    })
    _put_disk_kv(gconn, "bubbleId:comp-alpha:b1", {
        "type": 1, "text": "How do I wire this up?", "bubbleId": "b1",
    })
    _put_disk_kv(gconn, "bubbleId:comp-alpha:b2", {
        "type": 2,
        "text": "You can use a context manager.",
        "bubbleId": "b2",
        "allThinkingBlocks": [
            {"text": "User wants the simplest solution. Context manager it is."},
        ],
        "toolResults": [{"toolName": "Shell", "args": {"cmd": "ls"}}],
    })
    _put_disk_kv(gconn, "bubbleId:comp-alpha:b3", {
        "type": 1, "text": "Thanks!", "bubbleId": "b3",
    })

    # --- comp-beta: 1 bubble, assistant-only (edge case) ---
    _put_disk_kv(gconn, "composerData:comp-beta", {
        "composerId": "comp-beta",
        "fullConversationHeadersOnly": [{"bubbleId": "x1", "type": 2}],
    })
    _put_disk_kv(gconn, "bubbleId:comp-beta:x1", {
        "type": 2, "text": "One-sided message.", "bubbleId": "x1",
    })

    # --- comp-ghost: referenced by workspace, absent from global (deleted) ---
    # (no rows)

    gconn.commit()
    gconn.close()

    return ws_dir, gs_dir


@pytest.fixture
def fake_cursor(tmp_path, monkeypatch):
    """Patch the cursor paths to our synthetic install and return paths."""
    ws_dir, gs_dir = _build_fake_cursor_install(tmp_path)
    monkeypatch.setattr(composer_db, "get_workspace_storage_dir", lambda: ws_dir)
    monkeypatch.setattr(composer_db, "get_global_state_db", lambda: gs_dir / "state.vscdb")
    return ws_dir, gs_dir


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_path_to_project_folder_matches_cursor_encoding():
    # Both `/` and `_` become `-`, leading `/` is stripped.
    got = composer_db._path_to_project_folder("/Users/me/projects/sample_app")
    assert got == "Users-me-projects-sample-app"


def test_uri_to_path_handles_percent_encoding():
    assert composer_db._uri_to_path("file:///Users/me/my%20app") == "/Users/me/my app"
    assert composer_db._uri_to_path("not-a-uri") is None
    assert composer_db._uri_to_path(None) is None  # type: ignore[arg-type]


def test_bubble_to_message_user():
    m = composer_db._bubble_to_message({"type": 1, "text": "hello"})
    assert m is not None
    assert m.role == "user"
    assert m.text == "hello"


def test_bubble_to_message_assistant_surfaces_thinking():
    m = composer_db._bubble_to_message({
        "type": 2,
        "text": "Here is the answer.",
        "allThinkingBlocks": [{"text": "I weighed A vs B."}, {"text": "Picked A."}],
    })
    assert m is not None
    assert m.role == "assistant"
    # Reasoning gets its own text block, main answer follows.
    assert len(m.text_blocks) == 2
    assert "(reasoning)" in m.text_blocks[0]
    assert "I weighed A vs B." in m.text_blocks[0]
    assert "Picked A." in m.text_blocks[0]
    assert m.text_blocks[1] == "Here is the answer."


def test_bubble_to_message_unknown_type_returns_none():
    assert composer_db._bubble_to_message({"type": 99, "text": "?"}) is None


def test_bubble_to_message_empty_body_returns_none():
    # No text + no thinking + no tools -> we don't synthesize an empty message.
    assert composer_db._bubble_to_message({"type": 1}) is None


def test_enumerate_workspaces_returns_composers(fake_cursor):
    ws_list = list(composer_db.enumerate_workspaces())
    assert len(ws_list) == 1
    ws = ws_list[0]
    assert ws.hash == "abc123"
    assert ws.real_path == "/Users/me/projects/demo_app"
    assert ws.project_folder == "Users-me-projects-demo-app"
    assert {c["composerId"] for c in ws.composers} == {"comp-alpha", "comp-beta", "comp-ghost"}


def test_iter_db_sessions_reconstructs_ordered_conversation(fake_cursor):
    sessions = list(composer_db.iter_db_sessions())
    # comp-ghost is referenced but missing from global -> dropped.
    # comp-beta has 1 bubble (assistant-only) -> still a valid session.
    assert len(sessions) == 2
    ids = {s.session_id for s in sessions}
    assert ids == {"comp-alpha", "comp-beta"}

    alpha = next(s for s in sessions if s.session_id == "comp-alpha")
    assert alpha.source == "composer_db"
    assert alpha.transcript_path is None
    assert alpha.message_count == 3
    # Preserves the header order.
    roles = [m.role for m in alpha.messages]
    assert roles == ["user", "assistant", "user"]
    # first_user_query pulls from the first user message.
    assert alpha.first_user_query.startswith("How do I wire this up")
    # Assistant has a tool call extracted.
    assert any("Shell" == tc.name for tc in alpha.tool_calls)

    beta = next(s for s in sessions if s.session_id == "comp-beta")
    assert beta.message_count == 1
    assert beta.messages[0].role == "assistant"


def test_session_fingerprint_fields_populated(fake_cursor):
    sessions = list(composer_db.iter_db_sessions())
    alpha = next(s for s in sessions if s.session_id == "comp-alpha")
    # lastUpdatedAt = 1_700_000_000_000 ms -> 1_700_000_000_000 * 1_000_000 ns
    assert alpha._mtime_ns == 1_700_000_000_000 * 1_000_000
    # Bubbles are JSON-serialized dicts; nonzero and stable for a given payload.
    assert alpha._byte_size > 0


def test_project_filter_applied(fake_cursor):
    # Matches nothing.
    assert list(composer_db.iter_db_sessions(project_filter="nothing-matches")) == []
    # Substring match works.
    hits = list(composer_db.iter_db_sessions(project_filter="demo"))
    assert len(hits) == 2


def test_list_db_projects_counts_workspaces(fake_cursor):
    mapping = composer_db.list_db_projects()
    assert "Users-me-projects-demo-app" in mapping
    assert len(mapping["Users-me-projects-demo-app"]) == 3  # includes ghost
