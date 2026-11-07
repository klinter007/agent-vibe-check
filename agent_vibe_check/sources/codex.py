"""
OpenAI Codex CLI / Codex Desktop transcripts.

    ~/.codex/sessions/YYYY/MM/DD/rollout-<timestamp>-<uuid>.jsonl

Line 1 is `{"type": "session_meta", "payload": {"cwd", "originator", "source", ...}}`.
Conversation lives in `{"type": "response_item", "payload": {...}}` lines:

    message (role user | assistant | developer), content [{type: input_text|output_text, text}]
    function_call / custom_tool_call            {name, arguments | input}
    function_call_output / custom_tool_call_output {output}

Skipped by default: subagent threads (`source.subagent`), `codex exec`
batch runs, injected AGENTS.md / environment-context user messages.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from ..parser import Message, ToolCall
from .base import SourceInfo, is_noise_cwd, make_session

SOURCE = "codex"

_INJECTED_PREFIXES = (
    "# AGENTS.md instructions",
    "<environment_context>",
    "<user_instructions>",
    "<permissions instructions>",
    "<turn_aborted>",
)


def default_root() -> Path:
    env = os.environ.get("CODEX_HOME")
    base = Path(env) if env else Path.home() / ".codex"
    return base / "sessions"


def iter_transcript_files(root: Path) -> Iterator[Path]:
    if root.is_dir():
        yield from sorted(root.rglob("rollout-*.jsonl"))


def _content_text(content) -> list[str]:
    out: list[str] = []
    if isinstance(content, str):
        return [content] if content else []
    if isinstance(content, list):
        for b in content:
            if isinstance(b, dict) and isinstance(b.get("text"), str) and b["text"]:
                out.append(b["text"])
    return out


def _is_automated(meta: dict) -> bool:
    src = meta.get("source")
    if isinstance(src, dict) and "subagent" in src:
        return True
    return meta.get("originator") == "codex_exec" or src == "exec"


def parse_file(path: Path, include_automated: bool = False) -> "Session | None":
    meta: dict = {}
    messages: list[Message] = []
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return None
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line, strict=False)
            except json.JSONDecodeError:
                continue
            kind = obj.get("type")
            p = obj.get("payload") or {}
            if kind == "session_meta":
                # Subagent rollouts carry their own meta first, then the
                # parent's. The first one describes THIS file.
                meta = meta or p
                continue
            if kind != "response_item":
                continue
            ptype = p.get("type")
            if ptype == "message":
                role = p.get("role")
                if role not in ("user", "assistant"):
                    continue
                texts = _content_text(p.get("content"))
                if role == "user":
                    texts = [t for t in texts if not t.lstrip().startswith(_INJECTED_PREFIXES)]
                if not texts:
                    continue
                if role == "assistant" and messages and messages[-1].role == "assistant":
                    messages[-1].text_blocks.extend(texts)
                else:
                    messages.append(Message(role=role, text_blocks=texts))
            elif ptype in ("function_call", "custom_tool_call", "local_shell_call"):
                raw = p.get("arguments") if "arguments" in p else p.get("input")
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw)
                    except json.JSONDecodeError:
                        args = {"input": raw}
                else:
                    args = raw if isinstance(raw, dict) else {"input": raw}
                if not isinstance(args, dict):
                    args = {"input": args}
                tc = ToolCall(name=p.get("name") or ptype, input=args)
                if messages and messages[-1].role == "assistant":
                    messages[-1].tool_calls.append(tc)
                else:
                    messages.append(Message(role="assistant", tool_calls=[tc]))
            elif ptype in ("function_call_output", "custom_tool_call_output"):
                out = p.get("output")
                text = "\n".join(_content_text(out)) if isinstance(out, list) else str(out or "")
                result = {"type": "tool_result", "content": text}
                prev = messages[-1] if messages else None
                if prev is not None and prev.role == "user" and prev.tool_results and not prev.text_blocks:
                    prev.tool_results.append(result)
                else:
                    messages.append(Message(role="user", tool_results=[result]))

    if not include_automated and _is_automated(meta):
        return None
    cwd = meta.get("cwd")
    if is_noise_cwd(cwd):
        return None
    return make_session(
        session_id=str(meta.get("id") or meta.get("session_id") or path.stem),
        cwd=cwd,
        fallback_folder="codex-unknown",
        path=path,
        messages=messages,
        source=SOURCE,
        started_at=meta.get("timestamp"),
    )


class CodexSource:
    key = SOURCE

    def __init__(self, root: Path | None = None, include_automated: bool = False):
        self.root = root or default_root()
        self.include_automated = include_automated

    def info(self) -> SourceInfo:
        n = sum(1 for _ in iter_transcript_files(self.root))
        return SourceInfo(
            key=self.key, label="Codex", vendor="OpenAI",
            docs_url="https://learn.chatgpt.com/docs/codex/cli",
            roots=[self.root], session_count=n,
        )

    def files(self) -> Iterator[Path]:
        return iter_transcript_files(self.root)

    def parse(self, path: Path) -> "Session | None":
        return parse_file(path, include_automated=self.include_automated)

    def iter_sessions(self) -> Iterator["Session"]:
        for path in self.files():
            s = self.parse(path)
            if s is not None:
                yield s
