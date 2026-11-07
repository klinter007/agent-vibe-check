"""
Persistent config for Vibe Check.

Right now the only thing stored here is API keys so users can paste a key
once in the TUI and not have to edit shell dotfiles. Future settings
(preferred provider, model overrides, etc.) can go in the same file.

Security posture:
  - File lives under the user's Vibe Check home directory (default
    ~/.vibe-check/config.json). Same practical security as
    ~/.zshrc: readable only by the owning user.
  - File mode is explicitly set to 0o600 on every write to keep it
    user-only even if the umask is permissive.
  - Parent directory is chmod'd to 0o700 for the same reason, but only
    if it already belongs to the current user (we don't want to chmod
    something we don't own if $VIBE_CHECK_HOME points somewhere
    shared).
  - We don't use the OS keychain to keep the tool dependency-free and
    cross-platform. Anyone with read access to your user account can
    read $ANTHROPIC_API_KEY from .zshrc anyway, so this is equivalent.
"""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any

from .paths import get_vibe_check_home


CONFIG_FILENAME = "config.json"


def _config_path() -> Path:
    return get_vibe_check_home() / CONFIG_FILENAME


def _tighten_perms(path: Path) -> None:
    """Set 0600 on a file. Best-effort on non-POSIX (no-op on Windows)."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        pass


def _tighten_dir_perms_if_owned(path: Path) -> None:
    """Set 0700 on a directory if we own it. Best-effort."""
    try:
        st = path.stat()
    except OSError:
        return
    if hasattr(os, "geteuid") and st.st_uid != os.geteuid():
        return
    try:
        os.chmod(path, stat.S_IRWXU)
    except OSError:
        pass


def load_config() -> dict[str, Any]:
    """
    Load the config dict from disk. Returns {} on first run, corrupt file,
    or permission error. We never crash the tool over a missing / bad
    config file: the TUI should still boot.
    """
    path = _config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def save_config(data: dict[str, Any]) -> None:
    """
    Atomically write config with 0600 perms. Mirrors the approach used in
    llm_utils.atomic_write_json so a crash mid-write never corrupts the
    file.
    """
    path = _config_path()
    _tighten_dir_perms_if_owned(path.parent)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = Path(tmp.name)

    _tighten_perms(tmp_name)
    os.replace(tmp_name, path)
    _tighten_perms(path)


# API keys ---------------------------------------------------------------------


def load_api_keys() -> dict[str, str]:
    """Return {env_var_name: key} for every saved API key."""
    cfg = load_config()
    keys = cfg.get("api_keys")
    if not isinstance(keys, dict):
        return {}
    return {k: v for k, v in keys.items()
            if isinstance(k, str) and isinstance(v, str) and v}


def get_api_key(env_var: str) -> str | None:
    """
    Resolve an API key for the given env-var name, preferring the live
    environment over saved config. This keeps CI and shell-exported
    workflows working untouched.
    """
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    return load_api_keys().get(env_var)


def save_api_key(env_var: str, api_key: str) -> None:
    """Persist a key under the given env-var name."""
    if not api_key:
        raise ValueError("api_key must be non-empty")
    cfg = load_config()
    keys = cfg.get("api_keys")
    if not isinstance(keys, dict):
        keys = {}
    keys[env_var] = api_key
    cfg["api_keys"] = keys
    save_config(cfg)


def delete_api_key(env_var: str) -> bool:
    """Remove a saved key. Returns True if one was removed."""
    cfg = load_config()
    keys = cfg.get("api_keys")
    if not isinstance(keys, dict) or env_var not in keys:
        return False
    del keys[env_var]
    cfg["api_keys"] = keys
    save_config(cfg)
    return True


def mask_key(key: str) -> str:
    """Return a display-safe preview like 'sk-ant-api03...MQAA'."""
    if not key:
        return "(empty)"
    if len(key) <= 12:
        return "*" * len(key)
    return f"{key[:10]}...{key[-4:]}"


# Settings --------------------------------------------------------------------


DEFAULT_SETTINGS: dict[str, Any] = {
    "max_workers": 4,
    "models": {},  # {"ANTHROPIC_API_KEY": "claude-sonnet-4-6", ...}
}


def get_setting(key: str, default: Any = None) -> Any:
    """Read a scalar setting; falls back to DEFAULT_SETTINGS then to `default`."""
    cfg = load_config()
    settings = cfg.get("settings")
    if isinstance(settings, dict) and key in settings:
        return settings[key]
    if key in DEFAULT_SETTINGS:
        return DEFAULT_SETTINGS[key]
    return default


def set_setting(key: str, value: Any) -> None:
    cfg = load_config()
    settings = cfg.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    settings[key] = value
    cfg["settings"] = settings
    save_config(cfg)


def get_model_for(env_var: str) -> str | None:
    """Return the user's model override for a provider, or None for default."""
    cfg = load_config()
    settings = cfg.get("settings")
    if not isinstance(settings, dict):
        return None
    models = settings.get("models")
    if not isinstance(models, dict):
        return None
    v = models.get(env_var)
    return v if isinstance(v, str) and v else None


def set_model_for(env_var: str, model: str | None) -> None:
    """Set or clear (None) the model override for a provider."""
    cfg = load_config()
    settings = cfg.get("settings")
    if not isinstance(settings, dict):
        settings = {}
    models = settings.get("models")
    if not isinstance(models, dict):
        models = {}
    if model:
        models[env_var] = model
    elif env_var in models:
        del models[env_var]
    settings["models"] = models
    cfg["settings"] = settings
    save_config(cfg)


__all__ = [
    "load_config",
    "save_config",
    "load_api_keys",
    "get_api_key",
    "save_api_key",
    "delete_api_key",
    "mask_key",
    "DEFAULT_SETTINGS",
    "get_setting",
    "set_setting",
    "get_model_for",
    "set_model_for",
]
