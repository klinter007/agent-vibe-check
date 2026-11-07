"""Parser tests using synthetic JSONL fixtures (no Cursor install needed)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_vibe_check.parser import (
    Session,
    parse_transcript,
    parse_project,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PROJECT = FIXTURES / "sample_project"
SAMPLE_JSONL = (
    SAMPLE_PROJECT / "agent-transcripts" / "aaa-bbb-ccc" / "aaa-bbb-ccc.jsonl"
)


def test_fixture_present():
    assert SAMPLE_JSONL.exists(), f"Missing fixture: {SAMPLE_JSONL}"


def test_parse_transcript_returns_session():
    session = parse_transcript(SAMPLE_JSONL)
    assert isinstance(session, Session)
    assert session.session_id == "aaa-bbb-ccc"
    assert len(session.messages) == 6


def test_user_query_wrapper_stripped():
    session = parse_transcript(SAMPLE_JSONL)
    first_user = session.messages[0]
    assert first_user.role == "user"
    cleaned = first_user.clean_user_text
    assert "<user_query>" not in cleaned
    assert "refactor the auth module" in cleaned


def test_tool_calls_extracted():
    session = parse_transcript(SAMPLE_JSONL)
    tool_calls = [tc for m in session.messages for tc in m.tool_calls]
    assert len(tool_calls) == 2
    assert tool_calls[0].name == "Read"
    assert tool_calls[0].input == {"path": "auth.py"}
    assert tool_calls[1].name == "StrReplace"


def test_assistant_reasoning_preserved_in_text():
    """The 'why' lives in assistant text now (no top-level thinking field)."""
    session = parse_transcript(SAMPLE_JSONL)
    assistant_texts = [m.text for m in session.messages if m.role == "assistant"]
    combined = " ".join(assistant_texts)
    assert "JWT requires" in combined  # reasoning captured


def test_parse_project_aggregates_sessions():
    sessions = list(parse_project(SAMPLE_PROJECT))
    assert len(sessions) == 1
    assert sessions[0].session_id == "aaa-bbb-ccc"


def test_malformed_jsonl_is_skipped(tmp_path):
    """Bad lines should not kill the whole parse."""
    t = tmp_path / "agent-transcripts" / "bad" / "bad.jsonl"
    t.parent.mkdir(parents=True)
    t.write_text(
        "\n".join([
            "not json at all",
            '{"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}',
            "{broken",
        ])
        + "\n"
    )
    session = parse_transcript(t)
    assert len(session.messages) == 1
    assert session.messages[0].text == "hi"
