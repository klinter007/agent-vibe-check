"""Claude (Anthropic) provider implementation."""

from __future__ import annotations

import time

from . import BatchGoneError, LLMProvider, ProviderError, get_provider_info


JSON_SYSTEM_PROMPT = (
    "You are a precise technical analyst. "
    "Always respond with a single valid JSON object and nothing else. "
    "No prose, no markdown fences, no explanation outside the JSON."
)

# Analysis is judgment over a transcript, not hard reasoning: medium effort
# keeps thinking tokens (billed as output) low without hurting quality.
EFFORT = "medium"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"
BATCH_POLL_SECONDS = 30


def _system_blocks(system: str | None) -> list[dict]:
    blocks: list[dict] = [{"type": "text", "text": JSON_SYSTEM_PROMPT}]
    if system:
        # Same instructions for every session -> cache them.
        blocks.append({"type": "text", "text": system, "cache_control": {"type": "ephemeral"}})
    return blocks


def _text_of(message) -> str:
    return "".join(b.text for b in message.content if getattr(b, "type", "") == "text")


class ClaudeProvider(LLMProvider):
    info = get_provider_info("claude")
    supports_batch = True

    def __init__(self, api_key: str):
        try:
            import anthropic
        except ImportError as e:
            raise ProviderError(
                "anthropic not installed. Run: pip install anthropic"
            ) from e

        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=api_key)

    def _params(self, prompt: str, model: str | None, system: str | None) -> dict:
        return {
            "model": model or self.info.default_model,
            "max_tokens": MAX_TOKENS,
            "system": _system_blocks(system),
            "output_config": {"effort": EFFORT},
            "messages": [{"role": "user", "content": prompt}],
        }

    def analyze(self, prompt: str, *, model: str | None = None, system: str | None = None) -> str:
        params = self._params(prompt, model, system)
        # Server-side fallback: if a safety classifier declines (e.g. a
        # transcript about security work), the API re-runs the request on a
        # recommended fallback model instead of returning a refusal.
        response = self._client.beta.messages.create(
            betas=[FALLBACK_BETA], extra_body={"fallbacks": "default"}, **params,
        )
        if response.stop_reason == "refusal":
            raise ProviderError("model declined the request (refusal)")
        return _text_of(response)

    # -- Message Batches: submit / wait / results ---------------------------

    def batch_submit(self, items, *, model=None) -> str:
        """Submit [(custom_id, prompt, system)] as one batch; return its id."""
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
        from anthropic.types.messages.batch_create_params import Request

        requests = [
            Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**self._params(p, model, s)))
            for cid, p, s in items
        ]
        return self._client.messages.batches.create(requests=requests).id

    def batch_wait(self, batch_id: str, *, on_status=None, should_stop=None) -> str:
        """Poll until the batch ends. On stop, cancel and keep polling until it ends."""
        cancel_sent = False
        while True:
            try:
                batch = self._client.messages.batches.retrieve(batch_id)
            except (self._anthropic.NotFoundError, self._anthropic.PermissionDeniedError) as e:
                raise BatchGoneError(f"batch {batch_id} is gone: {e}") from e
            if batch.processing_status == "ended":
                return "ended"
            if should_stop and should_stop() and not cancel_sent:
                self._client.messages.batches.cancel(batch_id)
                cancel_sent = True
                if on_status:
                    on_status(f"batch {batch_id}: cancel requested; finished results are kept")
            elif on_status:
                c = batch.request_counts
                on_status(f"batch {batch_id}: {c.succeeded} done, {c.processing} running, {c.errored} errors")
            time.sleep(5 if cancel_sent else BATCH_POLL_SECONDS)

    def batch_results(self, batch_id: str, items_by_id: dict | None = None,
                      model: str | None = None) -> dict[str, str | Exception]:
        """
        Results of an ended batch, {custom_id: text | Exception}. Refusals are
        retried live with fallbacks when the original prompt is known.
        """
        out: dict[str, str | Exception] = {}
        refused: list[str] = []
        try:
            stream = self._client.messages.batches.results(batch_id)
            for result in stream:
                r = result.result
                if r.type == "succeeded":
                    if r.message.stop_reason == "refusal":
                        refused.append(result.custom_id)
                    else:
                        out[result.custom_id] = _text_of(r.message)
                elif r.type == "errored":
                    etype = getattr(getattr(r, "error", None), "type", None) or str(getattr(r, "error", ""))
                    out[result.custom_id] = ProviderError(f"batch error: {etype}")
                else:
                    out[result.custom_id] = ProviderError(f"batch request {r.type}")
        except (self._anthropic.NotFoundError, self._anthropic.PermissionDeniedError) as e:
            raise BatchGoneError(f"batch {batch_id} is gone: {e}") from e
        except self._anthropic.AnthropicError as e:
            raise ProviderError(f"could not read results of batch {batch_id}: {e}") from e
        for cid in refused:
            item = (items_by_id or {}).get(cid)
            if item is None:
                out[cid] = ProviderError("model declined the request (refusal)")
                continue
            _, p, s = item
            try:
                out[cid] = self.analyze(p, model=model, system=s)
            except Exception as e:  # noqa: BLE001
                out[cid] = e
        return out

    def batch_analyze(self, items, *, model=None, on_status=None, should_stop=None):
        """Message Batches API: 50% cheaper, results within ~1h (max 24h)."""
        if not items:
            return {}
        batch_id = self.batch_submit(items, model=model)
        if on_status:
            on_status(f"batch {batch_id} submitted ({len(items)} requests)")
        self.batch_wait(batch_id, on_status=on_status, should_stop=should_stop)
        return self.batch_results(batch_id, {cid: (cid, p, s) for cid, p, s in items}, model=model)
