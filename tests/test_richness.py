"""
Tests for the richness-based deep-analysis filter.

Verifies:
  - project_stats() aggregates cheaply from both JSONL and DB sources
  - qualifies_for_deep_analysis() applies the (any-of) rule correctly
  - qualifying_projects() sorts by total_messages desc
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from agent_vibe_check import composer_db, extract
from agent_vibe_check.extract import (
    DEFAULT_MIN_SESSION_MESSAGES,
    DEFAULT_MIN_TOTAL_MESSAGES,
    ProjectStats,
    qualifies_for_deep_analysis,
    qualifying_projects,
    project_stats,
)


# ---------------------------------------------------------------------------
# Pure-function tests for the richness rule
# ---------------------------------------------------------------------------


def test_rule_passes_on_single_long_session():
    # One session with 50 messages -> easily >= 10 max_session_messages.
    st = ProjectStats("foo", sessions=1, total_messages=50, max_session_messages=50)
    assert qualifies_for_deep_analysis(st) is True


def test_rule_passes_on_aggregate_volume_even_with_short_sessions():
    # Five sessions of 5 messages each -> 25 total, max 5. Fails "max >= 10"
    # but passes "total >= 20".
    st = ProjectStats("foo", sessions=5, total_messages=25, max_session_messages=5)
    assert qualifies_for_deep_analysis(st) is True


def test_rule_fails_on_throwaway_projects():
    # Three "fix this typo" sessions, 2 messages each.
    st = ProjectStats("foo", sessions=3, total_messages=6, max_session_messages=2)
    assert qualifies_for_deep_analysis(st) is False


def test_rule_uses_defaults_that_catch_real_content():
    # Regression guard: the defaults should accept a 1-session/200-msg project
    # (the exact motivating case that the old "3-session floor" was dropping).
    st = ProjectStats("real-work", sessions=1, total_messages=200, max_session_messages=200)
    assert qualifies_for_deep_analysis(st) is True


def test_rule_thresholds_are_configurable():
    # Tightening the rule should exclude borderline cases.
    st = ProjectStats("edge", sessions=1, total_messages=10, max_session_messages=10)
    assert qualifies_for_deep_analysis(st) is True
    assert qualifies_for_deep_analysis(st, min_session_messages=50, min_total_messages=100) is False


def test_qualifying_projects_sorts_by_total_messages_desc():
    stats = {
        "small": ProjectStats("small", 1, 30, 30),
        "big":   ProjectStats("big", 5, 500, 200),
        "mid":   ProjectStats("mid", 2, 100, 80),
        "tiny":  ProjectStats("tiny", 1, 2, 2),  # fails filter
    }
    got = qualifying_projects(stats)
    assert [s.short_name for s in got] == ["big", "mid", "small"]


# ---------------------------------------------------------------------------
# project_stats() integration: combined JSONL + DB aggregation
# ---------------------------------------------------------------------------


def _write_jsonl_session(project_root: Path, session_id: str, n_messages: int) -> None:
    """Create a fake JSONL transcript with `n_messages` non-empty lines."""
    session_dir = project_root / "agent-transcripts" / session_id
    session_dir.mkdir(parents=True)
    path = session_dir / f"{session_id}.jsonl"
    with path.open("w") as f:
        for i in range(n_messages):
            f.write(json.dumps({
                "role": "user" if i % 2 == 0 else "assistant",
                "message": {"content": [{"type": "text", "text": f"msg {i}"}]},
            }) + "\n")


def _build_synth_db_install(root: Path) -> None:
    """Minimal DB-backed install: one workspace, one composer with 12 bubble headers."""
    ws_dir = root / "workspaceStorage"
    gs_dir = root / "globalStorage"
    ws_dir.mkdir()
    gs_dir.mkdir()

    hash_dir = ws_dir / "h1"
    hash_dir.mkdir()
    (hash_dir / "workspace.json").write_text(
        json.dumps({"folder": "file:///Users/me/proj/db-only"}),
        encoding="utf-8",
    )

    def _init(path: Path) -> sqlite3.Connection:
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE ItemTable (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
        c.execute("CREATE TABLE cursorDiskKV (key TEXT UNIQUE ON CONFLICT REPLACE, value BLOB)")
        return c

    # Workspace DB
    w = _init(hash_dir / "state.vscdb")
    w.execute(
        "INSERT INTO ItemTable (key, value) VALUES (?, ?)",
        ("composer.composerData",
         json.dumps({"allComposers": [{"composerId": "cA", "lastUpdatedAt": 1}]}).encode()),
    )
    w.commit()
    w.close()

    # Global DB
    g = _init(gs_dir / "state.vscdb")
    g.execute(
        "INSERT INTO cursorDiskKV (key, value) VALUES (?, ?)",
        (
            "composerData:cA",
            json.dumps({
                "composerId": "cA",
                "fullConversationHeadersOnly": [
                    {"bubbleId": f"b{i}", "type": 1 if i % 2 == 0 else 2}
                    for i in range(12)
                ],
            }).encode(),
        ),
    )
    g.commit()
    g.close()


def test_project_stats_combines_both_sources(tmp_path, monkeypatch):
    # JSONL side: one project with two sessions of 7 messages each.
    jsonl_dir = tmp_path / "cursor_projects"
    jsonl_dir.mkdir()
    jsonl_project = jsonl_dir / "Users-me-proj-jsonl_only"  # short name -> "jsonl-only"
    _write_jsonl_session(jsonl_project, "s1", 7)
    _write_jsonl_session(jsonl_project, "s2", 7)

    # DB side: separate install with one composer of 12 messages.
    db_root = tmp_path / "db"
    db_root.mkdir()
    _build_synth_db_install(db_root)

    monkeypatch.setattr(extract, "get_cursor_projects_dir", lambda: jsonl_dir)
    monkeypatch.setattr(composer_db, "get_workspace_storage_dir", lambda: db_root / "workspaceStorage")
    # project_stats() imports get_global_state_db lazily from .paths, so patch there.
    from agent_vibe_check import paths as paths_mod
    monkeypatch.setattr(
        paths_mod, "get_global_state_db",
        lambda: db_root / "globalStorage" / "state.vscdb",
    )

    stats = project_stats()

    # JSONL project: 2 sessions, 14 total, max 7.
    jsonl_short = [k for k in stats if "jsonl" in k][0]
    s_jsonl = stats[jsonl_short]
    assert s_jsonl.sessions == 2
    assert s_jsonl.total_messages == 14
    assert s_jsonl.max_session_messages == 7

    # DB project: 1 session, 12 total, max 12.
    db_short = [k for k in stats if "db-only" in k][0]
    s_db = stats[db_short]
    assert s_db.sessions == 1
    assert s_db.total_messages == 12
    assert s_db.max_session_messages == 12

    # Richness filter: jsonl-only passes via total_messages >= 20? No, 14 < 20
    # AND max 7 < 10. So it should FAIL. db-only has max 12 >= 10, PASSES.
    assert qualifies_for_deep_analysis(s_jsonl) is False
    assert qualifies_for_deep_analysis(s_db) is True


def test_defaults_are_the_documented_values():
    # Guard against accidental threshold drift in the public API.
    assert DEFAULT_MIN_SESSION_MESSAGES == 10
    assert DEFAULT_MIN_TOTAL_MESSAGES == 20
