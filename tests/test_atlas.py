"""Project Atlas tests (no live LLM calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_vibe_check import atlas
from agent_vibe_check.providers import LLMProvider, ProviderInfo


class FakeProvider(LLMProvider):
    """Returns canned responses; lets tests avoid real API calls."""

    info = ProviderInfo(
        name="fake",
        display_name="Fake",
        env_var="FAKE_KEY",
        default_model="fake-model",
        signup_url="https://example.com",
    )

    def __init__(self, response: str):
        self._response = response

    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        return self._response


# ---------- Atlas: entry construction ----------

def test_atlas_entry_to_dict():
    e = atlas.AtlasEntry(
        short_name="foo",
        folder="Users-me-foo",
        session_count=3,
        message_count=42,
        max_session_messages=20,
        last_active="2026-04-17",
    )
    d = e.to_dict()
    assert d["short_name"] == "foo"
    assert d["session_count"] == 3
    assert d["max_session_messages"] == 20
    assert d["richness"] == "sparse"  # default


def test_atlas_parse_classification_well_formed():
    raw = json.dumps({
        "projects": [
            {"folder": "A", "category": "web app", "description": "...", "tech_hints": ["next"]},
            {"folder": "B", "category": "cli", "description": "...", "tech_hints": []},
        ]
    })
    out = atlas._parse_classification(raw)
    assert set(out.keys()) == {"A", "B"}
    assert out["A"]["category"] == "web app"
    assert out["A"]["tech_hints"] == ["next"]


def test_atlas_parse_classification_handles_fences_and_prose():
    raw = "here you go:\n```json\n" + json.dumps({
        "projects": [{"folder": "X", "category": "ml", "description": "d"}]
    }) + "\n```"
    out = atlas._parse_classification(raw)
    assert "X" in out
    assert out["X"]["category"] == "ml"


def test_atlas_parse_classification_garbage_returns_empty():
    assert atlas._parse_classification("nope") == {}


def test_classify_populates_entries_in_place():
    entries = [
        atlas.AtlasEntry(short_name="a", folder="A", session_count=5,
                         message_count=100, max_session_messages=40,
                         last_active="2026-01-01", sample_queries=["do X"]),
        atlas.AtlasEntry(short_name="b", folder="B", session_count=3,
                         message_count=50, max_session_messages=30,
                         last_active="2026-02-01", sample_queries=["do Y"]),
    ]
    fake = FakeProvider(json.dumps({
        "projects": [
            {"folder": "A", "category": "web", "description": "the A app", "tech_hints": ["ts"]},
            {"folder": "B", "category": "cli", "description": "the B tool", "tech_hints": ["py"]},
        ]
    }))
    atlas.classify(entries, fake)
    assert entries[0].category == "web"
    assert entries[0].description == "the A app"
    assert entries[1].tech_hints == ["py"]


def test_write_atlas_produces_markdown_and_json(tmp_path):
    entries = [
        atlas.AtlasEntry(
            short_name="alpha", folder="Users-me-alpha", session_count=5,
            message_count=200, max_session_messages=80,
            last_active="2026-04-17",
            richness="deep",
            real_path="/Users/me/alpha",
            category="web app", description="an app",
            tech_hints=["specification", "specification"],
        ),
        atlas.AtlasEntry(
            short_name="beta", folder="Users-me-beta", session_count=3,
            message_count=5, max_session_messages=2,
            last_active="2026-01-02",
            richness="sparse",
            category="cli tool", description="a cli",
        ),
    ]
    md_path, js_path = atlas.write_atlas(entries, out_dir=tmp_path)
    assert md_path.exists()
    assert js_path.exists()

    md = md_path.read_text()
    assert "# Project Atlas" in md
    assert "## cli tool" in md
    assert "## web app" in md
    assert "alpha" in md
    assert "/Users/me/alpha" in md  # resolved path shown when available
    # Richness badges are rendered in the markdown.
    assert "[deep] alpha" in md
    assert "[sparse] beta" in md
    # Header counts deep/sparse projects.
    assert "1 deep" in md and "1 sparse" in md

    j = json.loads(js_path.read_text())
    assert len(j["entries"]) == 2
    # JSON shape includes new fields.
    fields = set(j["entries"][0].keys())
    assert {"richness", "max_session_messages"} <= fields

def test_atlas_sample_queries_drop_secrets():
    from agent_vibe_check.parser import Message, Session
    key = "sk-" + "ant-" + "api03-" + "A1b2C3d4E5f6G7h8I9j0kL"
    s = Session(session_id="s1", project_folder="Users-me-projects-demo", _mtime_ns=1_700_000_000_000_000_000,
                messages=[Message(role="user", text_blocks=[f"deploy with {key} now"])] * 3)
    entry = atlas._build_candidate("Users-me-projects-demo", [s], {}, {})
    assert entry is not None
    assert all(key not in q for q in entry.sample_queries)
