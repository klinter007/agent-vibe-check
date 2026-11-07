"""
Google Gemini CLI transcripts.

    ~/.gemini/tmp/<project>/.project_root          real project path
    ~/.gemini/tmp/<project>/chats/session-*.jsonl  current format
    ~/.gemini/tmp/<project>/chats/session-*.json   older single-JSON format

JSONL format: line 1 is session metadata (`sessionId`, `startTime`, `kind`).
Then records with `id` and `type` (user | gemini | info). A record id can
appear more than once (later copy adds `toolCalls`); the last copy wins.
`{"$rewindTo": id}` drops that record and everything after it.
`{"$set": {...}}` lines are metadata updates and are ignored.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from ..parser import Message, ToolCall
from .base import SourceInfo, is_noise_cwd, make_session

SOURCE = "gemini_cli"


def default_root() -> Path:
    env = os.environ.get("GEMINI_CLI_HOME")
    base = Path(env) / ".gemini" if env else Path.home() / ".gemini"
    return base / "tmp"


def iter_transcript_files(root: Path) -> Iterator[Path]:
    if not root.is_dir():
        return
    for project_dir in sorted(root.iterdir()):
        chats = project_dir / "chats"
        if chats.is_dir():
            yield from sorted(p for p in chats.iterdir()
                              if p.suffix in (".jsonl", ".json") and p.name.startswith("session-"))


def _project_root(chat_file: Path) -> str | None:
    marker = chat_file.parent.parent / ".project_root"
    try:
        return marker.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _load_records(path: Path) -> tuple[dict, list[dict]]:
    """Return (metadata, ordered records) for either file format."""
    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, json.JSONDecodeError):
            return {}, []
        if isinstance(data, dict):
            return data, [r for r in data.get("messages", []) if isinstance(r, dict)]
        return {}, []

    meta: dict = {}
    order: list[str] = []
    by_id: dict[str, dict] = {}
    try:
        fh = path.open("r", encoding="utf-8", errors="replace")
    except OSError:
        return {}, []
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line, strict=False)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or "$set" in obj:
                continue
            if "$rewindTo" in obj:
                target = obj["$rewindTo"]
                if target in order:
                    cut = order.index(target)
                    for rid in order[cut:]:
                        by_id.pop(rid, None)
                    order = order[:cut]
                continue
            if "type" not in obj and "sessionId" in obj:
                meta = obj
                continue
            rid = obj.get("id")
            if not rid:
                continue
            if rid not in by_id:
                order.append(rid)
            by_id[rid] = obj
    return meta, [by_id[r] for r in order]


def _user_parts(content) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    results: list[dict] = []
    parts = content if isinstance(content, list) else [content]
    for part in parts:
        if isinstance(part, str):
            if part:
                texts.append(part)
        elif isinstance(part, dict):
            if isinstance(part.get("text"), str) and part["text"]:
                texts.append(part["text"])
            elif "functionResponse" in part:
                resp = part["functionResponse"].get("response")
                results.append({"type": "tool_result", "content": json.dumps(resp)[:2000]})
    return texts, results


def parse_file(path: Path) -> "Session | None":
    meta, records = _load_records(path)
    messages: list[Message] = []
    for r in records:
        rtype = r.get("type")
        if rtype == "user":
            texts, results = _user_parts(r.get("content"))
            texts = [t for t in texts if not t.lstrip().startswith("<session_context>")]
            if texts or results:
                messages.append(Message(role="user", text_blocks=texts, tool_results=results))
        elif rtype in ("gemini", "model", "assistant"):
            content = r.get("content")
            texts = [content] if isinstance(content, str) and content else []
            calls = [
                ToolCall(name=tc.get("name", ""), input=tc.get("args") or {})
                for tc in (r.get("toolCalls") or []) if isinstance(tc, dict)
            ]
            if texts or calls:
                messages.append(Message(role="assistant", text_blocks=texts, tool_calls=calls))

    cwd = _project_root(path)
    if is_noise_cwd(cwd):
        return None
    return make_session(
        session_id=str(meta.get("sessionId") or path.stem),
        cwd=cwd,
        fallback_folder=f"gemini-{path.parent.parent.name}",
        path=path,
        messages=messages,
        source=SOURCE,
        started_at=meta.get("startTime"),
    )


class GeminiCliSource:
    key = SOURCE

    def __init__(self, root: Path | None = None):
        self.root = root or default_root()

    def info(self) -> SourceInfo:
        n = sum(1 for _ in iter_transcript_files(self.root))
        return SourceInfo(
            key=self.key, label="Gemini CLI", vendor="Google",
            docs_url="https://github.com/google-gemini/gemini-cli",
            roots=[self.root], session_count=n,
        )

    def files(self) -> Iterator[Path]:
        return iter_transcript_files(self.root)

    def parse(self, path: Path) -> "Session | None":
        return parse_file(path)

    def iter_sessions(self) -> Iterator["Session"]:
        for path in self.files():
            s = self.parse(path)
            if s is not None:
                yield s
