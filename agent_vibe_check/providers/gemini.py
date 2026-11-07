"""Gemini provider implementation."""

from __future__ import annotations

from . import LLMProvider, ProviderError, get_provider_info


class GeminiProvider(LLMProvider):
    info = get_provider_info("gemini")

    def __init__(self, api_key: str):
        try:
            import google.generativeai as genai
        except ImportError as e:
            raise ProviderError(
                "Gemini support is optional. Install it with: pip install \"agent-vibe-check[gemini]\" "
                "(or with uvx: uvx --with google-generativeai --from git+https://github.com/klinter007/agent-vibe-check vibe-check)"
            ) from e

        genai.configure(api_key=api_key)
        self._genai = genai

    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        model_name = model or self.info.default_model
        if system:
            prompt = f"{system}\n\n{prompt}"
        model_instance = self._genai.GenerativeModel(model_name)
        response = model_instance.generate_content(
            prompt,
            generation_config=self._genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.2,
            ),
        )
        return response.text
