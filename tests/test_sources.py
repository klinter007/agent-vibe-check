"""Hermetic tests for the Claude Code, Codex and Gemini CLI source readers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_vibe_check import sources
from agent_vibe_check.sources import claude_code, codex, gemini_cli
from agent_vibe_check.sources.base import is_noise_cwd, slug_from_cwd


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path / "home"))
    return tmp_path


# -- base -------------------------------------------------------------------


def test_slug_matches_cursor_style():
    assert slug_from_cwd("/Users/alice/code_projects/my.app") == "Users-alice-code-projects-my-app"


def test_noise_cwd():
    assert is_noise_cwd("/private/tmp/x")
    assert is_noise_cwd("/Users/a/scratchpad/y")
    assert not is_noise_cwd("/Users/a/projects/app")
    assert not is_noise_cwd(None)


# -- Claude Code ------------------------------------------------------------


def _cc_rows(entrypoint: str = "cli", cwd: str = "/Users/alice/projects/app") -> list[dict]:
    base = {"cwd": cwd, "sessionId": "s1", "entrypoint": entrypoint}
    return [
        {"type": "mode", "mode": "default", "sessionId": "s1"},
        {**base, "type": "user", "message": {"role": "user", "content": "fix the login bug"}},
        {**base, "type": "assistant", "message": {"id": "m1", "content": [
            {"type": "thinking", "thinking": "hidden"},
        ]}},
        {**base, "type": "assistant", "message": {"id": "m1", "content": [
            {"type": "text", "text": "Reading the file."},
        ]}},
        {**base, "type": "assistant", "message": {"id": "m1", "content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "a.py"}},
        ]}},
        {**base, "type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "print(1)"},
        ]}},
        {**base, "type": "user", "isSidechain": True, "message": {"content": "subagent noise"}},
        {**base, "type": "assistant", "message": {"id": "m2", "content": [
            {"type": "text", "text": "Fixed."},
        ]}},
    ]


def test_claude_code_parses_and_merges_split_messages(tmp_path):
    path = _write_jsonl(tmp_path / "-Users-alice-projects-app" / "s1.jsonl", _cc_rows())
    s = claude_code.parse_file(path)
    assert s is not None
    assert s.source == "claude_code"
    assert s.project_folder == "Users-alice-projects-app"
    roles = [m.role for m in s.messages]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert s.messages[1].text == "Reading the file."
    assert s.messages[1].tool_calls[0].name == "Read"
    assert s.messages[2].tool_results
    assert all("subagent noise" not in m.text for m in s.messages)


def test_claude_code_skips_sdk_and_temp_sessions(tmp_path):
    sdk = _write_jsonl(tmp_path / "p" / "a.jsonl", _cc_rows(entrypoint="sdk-cli"))
    tmp = _write_jsonl(tmp_path / "p" / "b.jsonl", _cc_rows(cwd="/private/tmp/x"))
    assert claude_code.parse_file(sdk) is None
    assert claude_code.parse_file(sdk, include_automated=True) is not None
    assert claude_code.parse_file(tmp) is None


# -- Codex ------------------------------------------------------------------


def _codex_rows(source="cli", originator="codex-tui") -> list[dict]:
    return [
        {"type": "session_meta", "payload": {
            "id": "cx1", "cwd": "/Users/alice/projects/app",
            "originator": originator, "source": source}},
        {"type": "response_item", "payload": {"type": "message", "role": "developer",
            "content": [{"type": "input_text", "text": "system prompt"}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "# AGENTS.md instructions\n..."}]}},
        {"type": "response_item", "payload": {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": "add a test"}]}},
        {"type": "response_item", "payload": {"type": "reasoning", "summary": []}},
        {"type": "response_item", "payload": {"type": "message", "role": "assistant",
            "content": [{"type": "output_text", "text": "Running pytest."}]}},
        {"type": "response_item", "payload": {"type": "function_call", "name": "shell",
            "arguments": json.dumps({"command": ["pytest"]})}},
        {"type": "response_item", "payload": {"type": "function_call_output",
            "output": "1 passed"}},
        {"type": "event_msg", "payload": {"type": "token_count"}},
    ]


def test_codex_parses(tmp_path):
    path = _write_jsonl(tmp_path / "2026" / "10" / "01" / "rollout-x.jsonl", _codex_rows())
    s = codex.parse_file(path)
    assert s is not None and s.source == "codex"
    assert s.session_id == "cx1"
    assert s.project_folder == "Users-alice-projects-app"
    assert [m.role for m in s.messages] == ["user", "assistant", "user"]
    assert s.messages[0].text == "add a test"
    assert s.messages[1].tool_calls[0].input == {"command": ["pytest"]}
    assert s.messages[2].tool_results[0]["content"] == "1 passed"


def test_codex_skips_subagents_and_exec(tmp_path):
    sub = _write_jsonl(tmp_path / "rollout-a.jsonl",
                       _codex_rows(source={"subagent": {"thread_spawn": {}}}))
    ex = _write_jsonl(tmp_path / "rollout-b.jsonl", _codex_rows(originator="codex_exec"))
    assert codex.parse_file(sub) is None
    assert codex.parse_file(ex) is None


# -- Gemini CLI -------------------------------------------------------------


def test_gemini_jsonl_dedupes_and_rewinds(tmp_path):
    proj = tmp_path / "app"
    proj.mkdir()
    (proj / ".project_root").write_text("/Users/alice/projects/app\n")
    rows = [
        {"sessionId": "g1", "projectHash": "h", "startTime": "2026-10-01T00:00:00Z", "kind": "main"},
        {"id": "u0", "type": "user", "content": [{"text": "<session_context>\nsetup"}]},
        {"id": "u1", "type": "user", "content": [{"text": "make a cube"}]},
        {"$set": {"lastUpdated": "x"}},
        {"id": "g-a", "type": "gemini", "content": "", "thoughts": []},
        {"id": "g-a", "type": "gemini", "content": "",
         "toolCalls": [{"name": "run_shell_command", "args": {"command": "ls"}}]},
        {"id": "u2", "type": "user", "content": [{"text": "wrong, undo"}]},
        {"$rewindTo": "u2"},
        {"id": "g-b", "type": "gemini", "content": "Done."},
        {"id": "i1", "type": "info", "content": "Request cancelled."},
    ]
    path = _write_jsonl(proj / "chats" / "session-1.jsonl", rows)
    s = gemini_cli.parse_file(path)
    assert s is not None and s.source == "gemini_cli"
    assert s.session_id == "g1"
    assert s.project_folder == "Users-alice-projects-app"
    texts = [m.text for m in s.messages]
    assert texts == ["make a cube", "", "Done."]
    assert s.messages[1].tool_calls[0].name == "run_shell_command"


# -- registry ---------------------------------------------------------------


def test_enabled_keys_default_and_persist(isolated_home):
    assert sources.enabled_keys() == ["cursor"]
    assert not sources.onboarding_done()
    sources.set_enabled_keys(["claude_code", "bogus", "codex"])
    assert sources.enabled_keys() == ["claude_code", "codex"]
    assert sources.onboarding_done()


def test_file_session_stats_uses_cache(isolated_home, monkeypatch):
    root = isolated_home / "claude"
    _write_jsonl(root / "-Users-alice-projects-app" / "s1.jsonl", _cc_rows())
    monkeypatch.setattr(claude_code, "default_root", lambda: root)
    sources.set_enabled_keys(["claude_code"])

    first = list(sources.file_session_stats())
    assert first == [("claude_code", "Users-alice-projects-app", 4)]

    calls = []
    real = claude_code.parse_file
    monkeypatch.setattr(claude_code, "parse_file", lambda *a, **k: calls.append(1) or real(*a, **k))
    assert list(sources.file_session_stats()) == first
    assert calls == []  # served from cache


def test_claude_code_drops_injected_user_text():
    from agent_vibe_check.sources.claude_code import clean_user_text
    assert clean_user_text("<task-notification><task-id>x</task-id></task-notification>") == (None, None)
    assert clean_user_text("This session is being continued from a previous conversation...") == (None, None)
    assert clean_user_text("<local-command-stdout>ok</local-command-stdout>") == (None, None)
    assert clean_user_text(
        "<command-message>review</command-message>\n<command-name>/review</command-name>\n<command-args>high</command-args>"
    ) == ("/review high", None)
    assert clean_user_text("<bash-input>ls</bash-input>") == ("! ls", None)
    assert clean_user_text("<bash-stdout>a.py</bash-stdout><bash-stderr></bash-stderr>") == (None, "a.py\n")
    assert clean_user_text("fix it <system-reminder>noise</system-reminder>") == ("fix it", None)


def test_claude_code_skips_compact_summary_lines(tmp_path):
    rows = _cc_rows()
    rows.insert(1, {"type": "user", "isCompactSummary": True, "cwd": "/Users/alice/projects/app",
                    "message": {"content": "summary of earlier work"}})
    s = claude_code.parse_file(_write_jsonl(tmp_path / "p" / "s.jsonl", rows))
    assert all("summary of earlier work" not in m.text for m in s.messages)


def test_codex_subagent_detected_from_first_meta(tmp_path):
    rows = _codex_rows()
    rows[0] = {"type": "session_meta", "payload": {"id": "child", "cwd": "/Users/alice/projects/app",
               "originator": "codex-tui", "source": {"subagent": {"thread_spawn": {}}}}}
    rows.insert(1, {"type": "session_meta", "payload": {"id": "parent", "cwd": "/Users/alice/projects/app",
                    "originator": "codex-tui", "source": "vscode"}})
    path = _write_jsonl(tmp_path / "rollout-c.jsonl", rows)
    assert codex.parse_file(path) is None
    s = codex.parse_file(path, include_automated=True)
    assert s.session_id == "child"


def test_short_names_drop_the_user_on_every_os():
    from agent_vibe_check.extract import short_project_name
    assert short_project_name("home-alice-code-myapp") == "code-myapp"
    assert short_project_name("C-Users-alice-code-myapp") == "code-myapp"
    assert short_project_name("Users-alice-code-myapp") == "code-myapp"


def test_parse_json_lenient_only_returns_dicts():
    from agent_vibe_check.pipeline.llm_utils import parse_json_lenient
    assert parse_json_lenient('[{"a": 1}]') == {"a": 1}
    assert parse_json_lenient('[1, 2]') is None
    assert parse_json_lenient('"text"') is None
    assert parse_json_lenient('{"a": [1, 2,],}') == {"a": [1, 2]}


def test_card_strips_control_characters():
    import xml.etree.ElementTree as ET
    from agent_vibe_check.renderers.card import render_card_svg
    prof = {"stats": {}, "archetype": {"key": "x", "name": "Bad\x1bName", "tagline": "t\x07"},
            "narrative": {"card_rules": ["rule \x1b[31m red"]}}
    ET.fromstring(render_card_svg(prof))
