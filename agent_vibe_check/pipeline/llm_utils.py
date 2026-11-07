"""
Shared LLM helpers for the pipeline stages.

Kept deliberately thin: the provider abstraction in
`agent_vibe_check.providers` already handles transport and auth. This module
adds the stage-agnostic plumbing around it: lenient JSON parsing, retry policy,
and atomic file writes.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable


FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]+?)```")
OBJECT_RE = re.compile(r"\{[\s\S]*\}")


def parse_json_lenient(raw: str) -> dict[str, Any] | None:
    """
    Best-effort JSON parsing. Handles markdown fences, leading/trailing prose,
    and trailing commas (which Claude occasionally emits).

    Always returns a dict or None: a one-element list holding a dict is
    unwrapped (some providers wrap JSON mode output in an array); any other
    shape counts as a failure, so callers never call .get on a list.
    """
    def _as_dict(v):
        if isinstance(v, list) and len(v) == 1:
            v = v[0]
        return v if isinstance(v, dict) else None

    candidates: list[str] = [raw]
    fenced = FENCE_RE.search(raw)
    if fenced:
        candidates.append(fenced.group(1))
    obj = OBJECT_RE.search(raw)
    if obj:
        candidates.append(obj.group())

    for text in candidates:
        for attempt in (text, re.sub(r",(\s*[}\]])", r"\1", text)):
            try:
                d = _as_dict(json.loads(attempt))
            except json.JSONDecodeError:
                continue
            if d is not None:
                return d
    return None


def atomic_write_json(path: Path, data: Any) -> None:
    """
    Write JSON to `path` atomically: serialize to a tmp file in the same
    directory, fsync, then rename. Guarantees readers never see a half-written
    file even if we crash mid-write.
    """
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
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def _classify_error(e: Exception) -> tuple[str, float | None]:
    """
    Inspect an exception from a provider and decide how to handle it.

    Returns (category, suggested_wait_seconds). Categories:
      - "rate_limit": Anthropic 429 or equivalent; respect retry-after.
      - "retryable":  5xx, connection, timeout; back off.
      - "fatal":      4xx auth/schema/prompt errors; don't waste tokens retrying.

    `suggested_wait_seconds` is None unless the server sent a hint.
    """
    name = type(e).__name__
    msg = str(e).lower()

    # Try to read retry-after off the underlying HTTP response if present.
    wait_hint: float | None = None
    for attr in ("response", "_response"):
        resp = getattr(e, attr, None)
        if resp is None:
            continue
        headers = getattr(resp, "headers", None) or {}
        for header_key in ("retry-after", "Retry-After", "anthropic-ratelimit-requests-reset"):
            val = headers.get(header_key) if hasattr(headers, "get") else None
            if val is None:
                continue
            try:
                wait_hint = float(val)
                break
            except (TypeError, ValueError):
                pass
        if wait_hint is not None:
            break

    # anthropic.RateLimitError, openai.RateLimitError, etc.
    if "ratelimit" in name.lower() or "429" in msg or "rate limit" in msg:
        return "rate_limit", wait_hint

    # Auth / bad request -> don't retry. Wastes tokens + time.
    if any(k in name.lower() for k in ("authentication", "permission", "invalidrequest",
                                        "badrequest", "notfound")):
        return "fatal", None
    if any(code in msg for code in ("401", "403", "404", "400")):
        return "fatal", None

    # Everything else (5xx, timeout, connection, unknown) -> retry.
    return "retryable", wait_hint


def _default_on_retry(category: str, wait: float, attempt: int, err: Exception) -> None:
    """
    Default retry notifier: prints to stderr so long waits are visible in the
    TUI (which renders stderr inline between rich Live contexts) and in CI logs.
    Pipeline code doesn't need to pass anything for this to work.
    """
    label = {"rate_limit": "rate-limited",
             "retryable": "transient error"}.get(category, category)
    short = type(err).__name__
    print(
        f"  [pipeline] {label} ({short}); waiting {wait:.0f}s "
        f"before retry {attempt}...",
        file=sys.stderr,
        flush=True,
    )


def call_with_retries(
    fn: Callable[[], str],
    *,
    max_attempts: int = 5,
    base_delay: float = 3.0,
    max_delay: float = 120.0,
    on_retry: Callable[[str, float, int, Exception], None] | None = _default_on_retry,
) -> tuple[str | None, Exception | None]:
    """
    Run `fn` with adaptive backoff. Returns (result, last_error).

    - Rate-limit errors: wait retry-after (if present) or 60s, whichever longer.
    - Retryable errors: exponential backoff (3s, 6s, 12s, ...) capped at max_delay.
    - Fatal errors (auth/schema): fail immediately, don't waste tokens.

    `on_retry(category, wait_seconds, attempt, exception)` is invoked before
    each sleep so the TUI can surface "waiting 62s for rate limit" instead
    of looking frozen.
    """
    last_err: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return fn(), None
        except Exception as e:  # noqa: BLE001 - provider errors are varied
            last_err = e
            category, wait_hint = _classify_error(e)

            if category == "fatal":
                break
            if attempt == max_attempts - 1:
                break

            if category == "rate_limit":
                wait = max(wait_hint or 0.0, 60.0)
            else:
                wait = min(base_delay * (2 ** attempt), max_delay)

            if on_retry is not None:
                try:
                    on_retry(category, wait, attempt + 1, e)
                except Exception:  # noqa: BLE001 - never let UI hooks crash the pipeline
                    pass
            time.sleep(wait)
    return None, last_err


__all__ = [
    "parse_json_lenient",
    "atomic_write_json",
    "call_with_retries",
    "_classify_error",
]
