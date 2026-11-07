"""
LLM provider abstraction.

Keeps `analyze.py` provider-agnostic so Gemini, Claude, or future backends
(Ollama, OpenAI, etc.) can be swapped without touching prompt logic.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass


class ProviderError(RuntimeError):
    """Raised when a provider cannot be constructed or fails irrecoverably."""


class BatchGoneError(ProviderError):
    """A saved batch no longer exists (expired, deleted, or another account)."""


@dataclass
class ProviderInfo:
    """Metadata for surfacing available providers in the UI."""
    name: str
    display_name: str
    env_var: str
    default_model: str
    signup_url: str


class LLMProvider(ABC):
    """
    Minimal contract every provider must satisfy.

    The contract is deliberately narrow: send a prompt, get back a raw string
    that the caller will try to parse as JSON. JSON repair and result
    unpacking live in `analyze.py`, not here.
    """

    info: ProviderInfo

    supports_batch: bool = False

    @abstractmethod
    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        """
        Send prompt, return raw string response (expected JSON).

        `system` carries instructions that are identical across many calls
        (Stage 1 rules, taxonomy). Providers that support prompt caching put
        it in a cached prefix; others just prepend it.
        """

    def batch_analyze(
        self,
        items: list[tuple[str, str, str | None]],
        *,
        model: str | None = None,
        on_status=None,
        should_stop=None,
    ) -> dict[str, str | Exception]:
        """
        Run many independent prompts. `items` is [(custom_id, prompt, system)].
        Returns {custom_id: response_text | Exception}. The default runs them
        one by one; providers with a batch API override this (cheaper, async).
        """
        out: dict[str, str | Exception] = {}
        for i, (cid, prompt, system) in enumerate(items, start=1):
            if should_stop and should_stop():
                break
            try:
                out[cid] = self.analyze(prompt, model=model, system=system)
            except Exception as e:  # noqa: BLE001
                out[cid] = e
            if on_status:
                on_status(f"{i}/{len(items)} done")
        return out


# Registry ---------------------------------------------------------------------

_PROVIDERS: dict[str, ProviderInfo] = {
    "claude": ProviderInfo(
        name="claude",
        display_name="Claude (Anthropic)",
        env_var="ANTHROPIC_API_KEY",
        default_model="claude-sonnet-5-5",
        signup_url="https://console.anthropic.com/settings/keys",
    ),
    "gemini": ProviderInfo(
        name="gemini",
        display_name="Gemini (Google)",
        env_var="GEMINI_API_KEY",
        # Alias that tracks Google's current Pro model, so the default does
        # not go stale when a preview is retired.
        default_model="gemini-pro-latest",
        signup_url="https://aistudio.google.com/app/apikey",
    ),
}


def list_providers() -> list[ProviderInfo]:
    """Return all registered providers in preferred order."""
    return list(_PROVIDERS.values())


def get_provider_info(name: str) -> ProviderInfo:
    info = _PROVIDERS.get(name.lower())
    if info is None:
        raise ProviderError(
            f"Unknown provider '{name}'. Available: {', '.join(_PROVIDERS)}"
        )
    return info


def detect_available() -> list[str]:
    """
    Return provider names with a reachable API key. Env vars take
    precedence over the saved config so CI / shell workflows stay intact,
    but a key saved via the TUI settings menu is equally valid.
    """
    from ..config import get_api_key  # local to avoid import cycles
    return [p.name for p in _PROVIDERS.values() if get_api_key(p.env_var)]


def build_provider(
    name: str | None = None,
    api_key: str | None = None,
) -> LLMProvider:
    """
    Factory. If `name` is None, picks the first provider with a live API key.

    Key resolution order:
      1. `api_key` argument (explicit override)
      2. Environment variable (ANTHROPIC_API_KEY / GEMINI_API_KEY)
      3. Saved config (~/.vibe-check/config.json)
      4. ProviderError with signup URL
    """
    from ..config import get_api_key  # local to avoid import cycles

    if name is None:
        available = detect_available()
        if not available:
            keys = ", ".join(p.env_var for p in _PROVIDERS.values())
            raise ProviderError(
                f"No provider API key found. Set one of: {keys} "
                f"(or paste it in the TUI via Settings)."
            )
        name = available[0]

    info = get_provider_info(name)
    key = api_key or get_api_key(info.env_var)
    if not key:
        raise ProviderError(
            f"{info.display_name} requires {info.env_var}. Get a key at {info.signup_url}"
        )

    if info.name == "claude":
        from .claude import ClaudeProvider
        return ClaudeProvider(api_key=key)
    if info.name == "gemini":
        from .gemini import GeminiProvider
        return GeminiProvider(api_key=key)

    raise ProviderError(f"No implementation for provider '{info.name}'")
