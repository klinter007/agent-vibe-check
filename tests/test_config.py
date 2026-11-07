"""Tests for the persistent config / API-key store."""

from __future__ import annotations

import json
import os
import stat

import pytest

from agent_vibe_check import config as vibe_config


@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    """Point VIBE_CHECK_HOME at a temp dir so nothing leaks."""
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path))
    # Also make sure no real keys shadow the test.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return tmp_path


# ---------------------------------------------------------------------------
# Basic load/save
# ---------------------------------------------------------------------------


def test_load_config_empty_when_missing(isolated_home):
    assert vibe_config.load_config() == {}


def test_save_then_load_roundtrips(isolated_home):
    vibe_config.save_config({"api_keys": {"ANTHROPIC_API_KEY": "sk-test"}})
    assert vibe_config.load_config() == {
        "api_keys": {"ANTHROPIC_API_KEY": "sk-test"}
    }


def test_save_file_is_0600(isolated_home):
    vibe_config.save_config({"api_keys": {"X": "y"}})
    path = isolated_home / "config.json"
    assert path.exists()
    mode = stat.S_IMODE(path.stat().st_mode)
    # On POSIX we expect exactly 0600. Skip strict check on Windows.
    if os.name == "posix":
        assert mode == 0o600


def test_load_survives_corrupt_file(isolated_home):
    path = isolated_home / "config.json"
    path.write_text("{not valid json")
    assert vibe_config.load_config() == {}


def test_load_survives_non_dict_root(isolated_home):
    path = isolated_home / "config.json"
    path.write_text(json.dumps(["a", "list"]))
    assert vibe_config.load_config() == {}


# ---------------------------------------------------------------------------
# API key helpers
# ---------------------------------------------------------------------------


def test_save_api_key_persists_across_calls(isolated_home):
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "sk-ant-xyz")
    keys = vibe_config.load_api_keys()
    assert keys == {"ANTHROPIC_API_KEY": "sk-ant-xyz"}


def test_save_multiple_keys_coexist(isolated_home):
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "sk-a")
    vibe_config.save_api_key("GEMINI_API_KEY", "g-b")
    assert vibe_config.load_api_keys() == {
        "ANTHROPIC_API_KEY": "sk-a",
        "GEMINI_API_KEY": "g-b",
    }


def test_save_api_key_replaces_existing(isolated_home):
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "sk-old")
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "sk-new")
    assert vibe_config.load_api_keys()["ANTHROPIC_API_KEY"] == "sk-new"


def test_save_api_key_rejects_empty(isolated_home):
    with pytest.raises(ValueError):
        vibe_config.save_api_key("ANTHROPIC_API_KEY", "")


def test_delete_api_key_removes_and_reports(isolated_home):
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "sk-a")
    assert vibe_config.delete_api_key("ANTHROPIC_API_KEY") is True
    assert vibe_config.load_api_keys() == {}


def test_delete_missing_key_returns_false(isolated_home):
    assert vibe_config.delete_api_key("NOPE") is False


def test_load_api_keys_ignores_non_string_values(isolated_home):
    path = isolated_home / "config.json"
    path.write_text(json.dumps({
        "api_keys": {"OK": "yes", "BAD": 42, "EMPTY": ""}
    }))
    assert vibe_config.load_api_keys() == {"OK": "yes"}


# ---------------------------------------------------------------------------
# get_api_key precedence: env var > config > None
# ---------------------------------------------------------------------------


def test_get_api_key_prefers_env(isolated_home, monkeypatch):
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "from-config")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
    assert vibe_config.get_api_key("ANTHROPIC_API_KEY") == "from-env"


def test_get_api_key_falls_back_to_config(isolated_home):
    vibe_config.save_api_key("ANTHROPIC_API_KEY", "from-config")
    assert vibe_config.get_api_key("ANTHROPIC_API_KEY") == "from-config"


def test_get_api_key_returns_none_when_missing(isolated_home):
    assert vibe_config.get_api_key("ANTHROPIC_API_KEY") is None


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_mask_key_long():
    masked = vibe_config.mask_key(
        "sk-ant-api03-abc123def456ghi789jkl012mno345pqr678stu901vwxyz"
    )
    assert masked.startswith("sk-ant-api")
    assert masked.endswith("wxyz")
    assert "..." in masked


def test_mask_key_short():
    # Keys <= 12 chars are fully masked, not previewed.
    assert vibe_config.mask_key("abc") == "***"


def test_mask_key_empty():
    assert vibe_config.mask_key("") == "(empty)"


# ---------------------------------------------------------------------------
# Provider integration: build_provider should resolve saved keys
# ---------------------------------------------------------------------------


def test_build_provider_uses_saved_key(isolated_home, monkeypatch):
    """detect_available and build_provider should pick up saved keys."""
    from agent_vibe_check import providers

    vibe_config.save_api_key("ANTHROPIC_API_KEY", "sk-saved")
    available = providers.detect_available()
    assert "claude" in available

    # We don't actually instantiate the Anthropic client; just verify that
    # the missing-key branch doesn't fire when only config has the key.
    # ClaudeProvider construction imports anthropic which may or may not be
    # installed in the test env, so we patch it to avoid that dependency.
    class StubClaude:
        def __init__(self, api_key):
            self.api_key = api_key

    monkeypatch.setattr(providers, "_PROVIDERS", providers._PROVIDERS)
    import agent_vibe_check.providers.claude as claude_mod
    monkeypatch.setattr(claude_mod, "ClaudeProvider", StubClaude)

    inst = providers.build_provider(name="claude")
    assert inst.api_key == "sk-saved"
