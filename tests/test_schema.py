"""Schema drift detector + top-N project selection."""

from __future__ import annotations

from pathlib import Path

from agent_vibe_check import schema as schema_check

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_PROJECT = FIXTURES / "sample_project"


# ---------- Schema check ----------


def test_schema_check_passes_on_valid_fixture():
    result = schema_check.check_project(SAMPLE_PROJECT, sample_size=5, seed=0)
    assert result.sampled_files == 1
    assert result.total_lines > 0
    assert result.parse_rate == 1.0
    assert result.ok is True


def test_schema_check_fails_on_garbage(tmp_path):
    t_dir = tmp_path / "fake_project" / "agent-transcripts" / "session-1"
    t_dir.mkdir(parents=True)
    (t_dir / "session-1.jsonl").write_text("\n".join(["garbage"] * 20))

    result = schema_check.check_project(tmp_path / "fake_project", sample_size=5, seed=0)
    assert result.parse_rate == 0.0
    assert result.ok is False
    assert "drift" in result.reason.lower() or "changed" in result.reason.lower()


def test_schema_check_handles_empty_project(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    result = schema_check.check_project(empty)
    assert result.sampled_files == 0
    assert result.ok is False


# ---------- Top-N ----------


def test_top_n_smoke(monkeypatch, tmp_path):
    projects = tmp_path / "projects"
    for name, n in [("big", 5), ("medium", 3), ("small", 1)]:
        for i in range(n):
            d = projects / f"fake-{name}" / "agent-transcripts" / f"sess-{i}"
            d.mkdir(parents=True)
            (d / f"sess-{i}.jsonl").write_text(
                '{"role": "user", "message": {"content": [{"type": "text", "text": "hi"}]}}\n'
            )

    from agent_vibe_check import extract, composer_db
    monkeypatch.setattr(extract, "get_cursor_projects_dir", lambda: projects)
    monkeypatch.setattr(composer_db, "list_db_projects", lambda: {})

    top = extract.top_n_projects(n=5, min_sessions=3)
    names = [name for name, _ in top]
    assert any("big" in n for n in names)
    assert any("medium" in n for n in names)
    assert not any("small" in n for n in names)
