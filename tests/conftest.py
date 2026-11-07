"""Keep every test hermetic: never read or write the real ~/.vibe-check."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_vibe_home(tmp_path_factory, monkeypatch):
    monkeypatch.setenv("VIBE_CHECK_HOME", str(tmp_path_factory.mktemp("vibe_home")))
