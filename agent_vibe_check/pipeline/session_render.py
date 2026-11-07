"""
Render a parsed Session as a compact, msg_idx-addressable text block for Stage 1.

Goals:
  - Each message gets a stable `[N]` prefix so the LLM can cite evidence by index.
  - Keep signal, drop noise: tool_result blobs get truncated hard.
  - Stay under a character budget so a single session doesn't blow the context
    window. When we have to truncate, drop from the middle and tag the gap.

The character budget is a proxy for tokens: ~4 chars/token is a safe rule of
thumb for English + code, so 80k chars is ~20k input tokens which leaves
plenty of room for the prompt scaffolding and response.
"""

from __future__ import annotations

from ..redact import strip_secrets

import json
from dataclasses import dataclass

from ..parser import Message, Session


MAX_RENDER_CHARS = 80_000
MAX_TEXT_PER_BLOCK = 4_000
MAX_TOOL_ARG_CHARS = 600
MAX_TOOL_RESULT_CHARS = 400


@dataclass
class RenderedSession:
    text: str
    message_count: int  # messages actually included (same as Session.message_count unless truncated)
    truncated: bool
    chars: int

    # True `msg_idx` range used, so callers can trust evidence citations.
    msg_idx_range: tuple[int, int]


def _shorten(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    head = s[: limit // 2]
    tail = s[-limit // 2 :]
    return f"{head}\n... [{len(s) - limit} chars snipped] ...\n{tail}"


def _render_tool_call(name: str, args: dict) -> str:
    try:
        arg_text = json.dumps(args, ensure_ascii=False)
    except (TypeError, ValueError):
        arg_text = str(args)
    arg_text = _shorten(strip_secrets(arg_text), MAX_TOOL_ARG_CHARS)
    return f"<<tool:{name}>> {arg_text}"


def _render_tool_result(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        chunks: list[str] = []
        for c in content:
            if isinstance(c, dict) and c.get("type") == "text":
                chunks.append(str(c.get("text", "")))
            elif isinstance(c, str):
                chunks.append(c)
        raw = "\n".join(chunks)
    elif isinstance(content, str):
        raw = content
    else:
        raw = ""
    return f"<<result>> {_shorten(strip_secrets(raw), MAX_TOOL_RESULT_CHARS)}"


def _render_message(idx: int, msg: Message) -> str:
    role_tag = "USER" if msg.role == "user" else "AI"
    lines: list[str] = [f"[{idx}] {role_tag}:"]

    if msg.role == "user":
        body = msg.clean_user_text or msg.text
        if body:
            lines.append(_shorten(strip_secrets(body), MAX_TEXT_PER_BLOCK))
    else:
        if msg.text:
            lines.append(_shorten(strip_secrets(msg.text), MAX_TEXT_PER_BLOCK))
        for tc in msg.tool_calls:
            lines.append(_render_tool_call(tc.name, tc.input))

    for tr in msg.tool_results:
        lines.append(_render_tool_result(tr))

    return "\n".join(lines)


def render_session(session: Session, *, max_chars: int = MAX_RENDER_CHARS) -> RenderedSession:
    """
    Build the final prompt-ready string for a session.

    Returns the string plus metadata the extractor needs to know whether
    truncation happened (affects LLM prompt hinting).
    """
    rendered: list[str] = []
    total = 0
    for i, msg in enumerate(session.messages):
        # Keys and tokens are removed before truncation (inside _render_message,
        # so a cut cannot expose half a key) and once more on the whole block.
        block = strip_secrets(_render_message(i, msg))
        rendered.append(block)
        total += len(block) + 2  # +2 for the join separator

    if total <= max_chars:
        body = "\n\n".join(rendered)
        return RenderedSession(
            text=body,
            message_count=session.message_count,
            truncated=False,
            chars=len(body),
            msg_idx_range=(0, max(0, session.message_count - 1)),
        )

    # Over budget: keep head and tail, drop the middle. We preserve original
    # msg_idx numbering in the [N] prefix so evidence citations survive.
    half = max_chars // 2 - 200  # reserve ~200 chars for the gap marker
    head: list[str] = []
    head_chars = 0
    for block in rendered:
        if head_chars + len(block) + 2 > half:
            break
        head.append(block)
        head_chars += len(block) + 2

    tail: list[str] = []
    tail_chars = 0
    for block in reversed(rendered):
        if tail_chars + len(block) + 2 > half:
            break
        tail.append(block)
        tail_chars += len(block) + 2
    tail.reverse()

    kept = len(head) + len(tail)
    dropped = session.message_count - kept
    gap = f"\n\n--- [{dropped} messages omitted for length] ---\n\n"
    body = "\n\n".join(head) + gap + "\n\n".join(tail)
    return RenderedSession(
        text=body,
        message_count=kept,
        truncated=True,
        chars=len(body),
        msg_idx_range=(0, max(0, session.message_count - 1)),
    )


__all__ = ["RenderedSession", "render_session", "MAX_RENDER_CHARS"]
