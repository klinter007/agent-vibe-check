"""
One provider for every server that speaks the OpenAI "chat completions" API:
OpenAI, xAI (Grok), OpenRouter, Groq, Ollama and LM Studio (local), and any
other compatible endpoint.

Stdlib only (urllib), so supporting many vendors adds no dependency.
Model names change often, so there is no hard-coded default: the user picks
one from the server's own /models list once, and it is saved in config.
"""

from __future__ import annotations

import json
import types
import urllib.error
import urllib.request

from . import LLMProvider, ProviderError, ProviderInfo

JSON_SYSTEM_PROMPT = (
    "You are a precise technical analyst. "
    "Always respond with a single valid JSON object and nothing else. "
    "No prose, no markdown fences, no explanation outside the JSON."
)
TIMEOUT_SECONDS = 600

# Model ids that are clearly not chat models (images, audio, embeddings...).
_NOT_CHAT = ("embed", "image", "audio", "tts", "whisper", "transcribe", "moderation",
             "realtime", "dall-e", "davinci", "babbage", "search", "rerank", "live", "vision-preview")


class HTTPProviderError(ProviderError):
    """Carries the HTTP status in its message and the headers for retry-after."""

    def __init__(self, status: int, body: str, headers=None):
        super().__init__(f"HTTP {status}: {body[:300]}")
        self.status = status
        self.response = types.SimpleNamespace(headers=headers or {})


class OpenAICompatProvider(LLMProvider):
    supports_batch = False

    def __init__(self, info: ProviderInfo, *, api_key: str | None, base_url: str, model: str | None = None):
        self.info = info
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._model = model
        self._json_mode = True   # switched off if the server rejects response_format

    # -- HTTP --------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        headers = {"Content-Type": "application/json", "User-Agent": "agent-vibe-check"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(self._base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
                return json.loads(resp.read().decode("utf-8", errors="replace"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise HTTPProviderError(e.code, body, e.headers) from e
        except urllib.error.URLError as e:
            raise ProviderError(f"cannot reach {self._base}: {e.reason}") from e

    # -- API ---------------------------------------------------------------------

    def list_models(self) -> list[str]:
        """Chat-capable model ids the server offers, newest first (A-Z when no dates)."""
        data = self._request("GET", "/models")
        rows = [m for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]
        rows = [m for m in rows if not any(k in m["id"].lower() for k in _NOT_CHAT)]
        rows.sort(key=lambda m: m["id"])
        rows.sort(key=lambda m: m.get("created") or 0, reverse=True)
        return [m["id"] for m in rows]

    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        model_name = model or self._model
        if not model_name:
            raise ProviderError(
                f"No model chosen for {self.info.display_name}. Pick one in Settings -> API keys / models."
            )
        sys_text = JSON_SYSTEM_PROMPT + ("\n\n" + system if system else "")
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": prompt},
            ],
        }
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        try:
            data = self._request("POST", "/chat/completions", payload)
        except HTTPProviderError as e:
            # Some servers/models do not support JSON mode: retry once without it.
            if self._json_mode and e.status == 400 and "response_format" in str(e).lower():
                self._json_mode = False
                payload.pop("response_format", None)
                data = self._request("POST", "/chat/completions", payload)
            else:
                raise
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"unexpected response from {self.info.display_name}: {str(data)[:300]}") from e


__all__ = ["OpenAICompatProvider", "HTTPProviderError"]
