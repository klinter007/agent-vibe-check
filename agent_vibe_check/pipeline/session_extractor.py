"""
Stage 1: per-session fact extraction.

Input:  agent_vibe_check.parser.Session
Output: agent_vibe_check.pipeline.schemas.SessionFacts
Side effects:
  - writes extractions/<short>/sessions/<session_id>.json  (SessionFacts)
  - updates extractions/<short>/sessions/_index.json        (fingerprint cache)
  - on failure, writes  <session_id>.raw.txt with the raw LLM response

Caching contract:
  A session is considered fresh if (a) its fact file exists and (b) the
  index's fingerprint matches the current (mtime_ns, byte_size, message_count).
  Matching sessions are NOT re-sent to the LLM.

Error handling:
  - Very short sessions (< MIN_MESSAGES) are skipped entirely (cheap, no LLM).
  - Transient LLM errors are retried with exponential backoff.
  - Permanent failures produce a `.raw.txt` sidecar and return None; the run
    continues to the next session.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..extract import short_project_name
from ..parser import Session
from ..paths import get_extractions_dir
from ..providers import LLMProvider
from .llm_utils import atomic_write_json, call_with_retries, parse_json_lenient
from .schemas import SCHEMA_VERSION, SessionFacts
from .session_render import render_session
from .taxonomy import (
    CONFIDENCES,
    LEVELS,
    dimension_list_for_prompt,
)


MIN_MESSAGES = 4  # anything shorter is noise (Cursor session that died at hello)
INDEX_FILE = "_index.json"


# The instructions are identical for every session, so they go in the system
# prompt where the provider can cache them. Only the transcript varies.
STAGE1_SYSTEM = """You are reviewing one session between a person and an AI coding agent.
You are not grading their code: the agent wrote most of it. You are judging
how well the person DIRECTED the agent, where time was lost, and whether the
session reached its goal. Your output feeds a report the person reads about
themselves, so be specific, fair and evidence-based. Never flatter.

Cite messages by the `[N]` prefix in the transcript (`msg_idx`). If you cannot
cite a message for a claim, leave the claim out.

# Who is driving

Decide who typed the USER turns:
- `human`: a person typing (typos, short asks, reactions, changes of mind).
- `agent`: another program or agent typed them: briefs from a coordinator,
  status pings, DONE / STUCK protocols, round-based multi-author prompts,
  messages that quote other agents' output.
- `mixed`: an agent drove it, but a person stepped in.
When `agent` or `mixed`, return an empty `vibes` list: the report scores only
how the person directs agents, not what agents write to each other. Still
fill in outcome, friction and the rest.

# Lens 1: Vibes (how the person directs the agent)

Rate only the signals you actually saw in this session. Omit the rest.

{vibes}

Levels: beginner, working, strong, expert.
Calibration (important, earlier versions inflated scores):
- `working` is the normal level for competent, unremarkable behavior.
- `strong` needs at least 2 separate pieces of evidence in this session.
- `expert` only for behavior another developer could copy as a model example.
- A short or routine session usually supports 0-2 signals, not all 6.

# Lens 2: Friction (where time and effort were lost)

A friction event is a stretch of the session where progress stalled.
Types: {friction_types}
Causes: {friction_causes}

For each event give the message range, the type, the most likely cause, how
many messages it cost before progress resumed, and ONE sentence the person
could have said or written down up front that would have prevented it.
Do not count normal back-and-forth as friction. A clean session has none.

# Outcome

Did the session reach the goal the person set?
- `shipped`: goal met, with evidence (confirmed working, tests pass, commit, PR, deploy, file delivered).
- `partial`: some of the goal met, or met with known open issues.
- `abandoned`: the person gave up, reverted, or the session ended on an unresolved failure.
- `unknown`: no evidence either way (the session just stops). Prefer this to guessing.

# Rules for the person's agent instructions

If a friction event is likely to happen again, write up to 3 rules the person
could paste into their agent instructions file (CLAUDE.md, AGENTS.md,
.cursorrules, GEMINI.md). Each rule is one imperative sentence addressed to
the agent and tied to the friction it prevents. Set `scope`:
- `general`: useful in any project ("Run the tests before you say a fix works.").
- `project`: only makes sense for this codebase or machine.
No rule for a one-off accident.

# Privacy

Never copy secrets, API keys, tokens, passwords, email addresses, internal
URLs, hostnames or people's names into your output. Describe them generically
("an API key", "a teammate", "the internal dashboard").

# Output

Return ONE JSON object, no prose, no markdown fences:

{{
  "topic": "3-8 word title",
  "goal": "what the person wanted, one sentence, or empty if unclear",
  "summary": "1-2 factual sentences on what happened",
  "driver": "human|agent|mixed",
  "task_type": "one of {task_types}",
  "topics_touched": ["<=6 concrete technical topics"],
  "outcome": {{
    "status": "shipped|partial|abandoned|unknown",
    "confidence": "low|med|high",
    "evidence": [{{"msg_idx": 88, "signal": "user confirms the export works"}}]
  }},
  "vibes": [
    {{
      "signal": "verification",
      "level": "strong",
      "confidence": "med",
      "evidence": [
        {{"msg_idx": 14, "behavior": "asked for the failing test output before accepting the fix"}},
        {{"msg_idx": 31, "behavior": "caught that the agent edited the wrong config file"}}
      ]
    }}
  ],
  "friction": [
    {{
      "type": "loop",
      "cause": "missing-context",
      "msg_idx_range": [40, 58],
      "cost_messages": 18,
      "what_happened": "agent retried the same import fix 4 times; the real cause was a second venv",
      "prevention": "State which Python environment the project uses before asking for fixes."
    }}
  ],
  "wins": [{{"description": "user gave the exact error and file, fix landed first try", "msg_idxs": [5, 7]}}],
  "ai_failures": [{{"what_ai_did_wrong": "claimed tests passed without running them", "msg_idx": 35, "user_caught_it": true}}],
  "decisions": [{{"decision": "use SQLite over Postgres", "rationale": "single-user local tool", "msg_idx": 42}}],
  "rule_candidates": [
    {{"rule": "Run the tests and show the output before you say a fix works.", "prevents": "agent claimed success without checking", "scope": "general", "msg_idx": 35}}
  ]
}}
"""

STAGE1_USER = """# Context

Project: {project_short}
Tool: {tool}
Messages: {message_count}{truncation_note}

# Transcript

{transcript}
"""


def stage1_system_prompt() -> str:
    from .taxonomy import FRICTION_CAUSES, FRICTION_TYPES, TASK_TYPES
    return STAGE1_SYSTEM.format(
        vibes=dimension_list_for_prompt(),
        friction_types=", ".join(FRICTION_TYPES),
        friction_causes=", ".join(FRICTION_CAUSES),
        task_types="|".join(TASK_TYPES),
    )


@dataclass
class _IndexEntry:
    mtime_ns: int
    byte_size: int
    message_count: int
    last_processed_at: str
    provider: str
    model: str
    truncated: bool = False
    status: str = "ok"  # ok | failed | skipped
    parser_rev: int = 1

    def to_dict(self) -> dict:
        return {
            "mtime_ns": self.mtime_ns,
            "byte_size": self.byte_size,
            "message_count": self.message_count,
            "last_processed_at": self.last_processed_at,
            "provider": self.provider,
            "model": self.model,
            "truncated": self.truncated,
            "status": self.status,
            "parser_rev": self.parser_rev,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "_IndexEntry":
        return cls(
            mtime_ns=int(d.get("mtime_ns", 0)),
            byte_size=int(d.get("byte_size", 0)),
            message_count=int(d.get("message_count", 0)),
            last_processed_at=str(d.get("last_processed_at", "")),
            provider=str(d.get("provider", "")),
            model=str(d.get("model", "")),
            truncated=bool(d.get("truncated", False)),
            status=str(d.get("status", "ok")),
            parser_rev=int(d.get("parser_rev", 1) or 1),
        )


def _sessions_dir(project_short: str) -> Path:
    d = get_extractions_dir() / project_short / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_index(project_short: str) -> dict[str, _IndexEntry]:
    idx_path = _sessions_dir(project_short) / INDEX_FILE
    if not idx_path.exists():
        return {}
    try:
        raw = json.loads(idx_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        k: _IndexEntry.from_dict(v)
        for k, v in (raw.get("entries") or {}).items()
        if isinstance(v, dict)
    }


def _save_index(project_short: str, index: dict[str, _IndexEntry]) -> None:
    idx_path = _sessions_dir(project_short) / INDEX_FILE
    payload = {
        "schema_version": SCHEMA_VERSION,
        "entries": {k: v.to_dict() for k, v in index.items()},
    }
    atomic_write_json(idx_path, payload)


def _fingerprint_matches(session: Session, entry: _IndexEntry) -> bool:
    return (
        entry.mtime_ns == getattr(session, "_mtime_ns", 0)
        and entry.byte_size == getattr(session, "_byte_size", 0)
        and entry.message_count == session.message_count
    )


def _session_timestamp(session: Session) -> str:
    """ISO timestamp for when this session started (file mtime as fallback)."""
    started = getattr(session, "started_at", None)
    if started is not None:
        return started.isoformat(timespec="seconds")
    try:
        return session.file_mtime.isoformat(timespec="seconds")
    except Exception:  # noqa: BLE001
        return datetime.now().isoformat(timespec="seconds")


def _build_prompt(session: Session) -> tuple[str, bool]:
    """Return (user prompt, truncated). The system prompt is `stage1_system_prompt()`."""
    from ..sources import tool_label

    rendered = render_session(session)
    truncation_note = (
        f" (transcript was truncated to fit; {session.message_count - rendered.message_count} "
        f"messages omitted from the middle)"
        if rendered.truncated
        else ""
    )
    prompt = STAGE1_USER.format(
        project_short=short_project_name(session.project_folder),
        tool=tool_label(session.source),
        message_count=session.message_count,
        truncation_note=truncation_note,
        transcript=rendered.text,
    )
    return prompt, rendered.truncated


def _sanitize_facts(raw: dict, session: Session) -> SessionFacts:
    """
    Normalize an LLM response into a SessionFacts, filling in stable metadata
    from the Session object (which the LLM doesn't need to re-assert).
    """
    from ..sources import tool_label
    from .taxonomy import DIMENSION_BY_NAME

    raw = dict(raw)  # shallow copy, don't mutate caller
    raw["schema_version"] = SCHEMA_VERSION
    raw["session_id"] = session.session_id
    raw["project_folder"] = session.project_folder
    raw["project_short"] = short_project_name(session.project_folder)
    raw["timestamp"] = _session_timestamp(session)
    raw["ended_at"] = session.file_mtime.isoformat(timespec="seconds")
    raw["source"] = session.source
    raw["tool"] = tool_label(session.source)
    raw["message_count"] = session.message_count

    # v2 prompt says "vibes"/"signal"/"wins"; storage keeps the v1 field names
    # (skills_observed/dimension/vibing_moments) so Stages 2-4 aggregate as before.
    vibes = raw.get("vibes")
    if vibes is None:
        vibes = raw.get("skills_observed") or []
    clean = []
    for v in vibes:
        if not isinstance(v, dict):
            continue
        dim = v.get("signal") or v.get("dimension")
        if dim not in DIMENSION_BY_NAME:
            continue
        v = dict(v, dimension=dim)
        if v.get("level") not in LEVELS:
            v["level"] = "unobserved"
        if v.get("confidence") not in CONFIDENCES:
            v["confidence"] = "low"
        clean.append(v)
    raw["skills_observed"] = clean
    if "wins" in raw and "vibing_moments" not in raw:
        raw["vibing_moments"] = raw.get("wins") or []

    facts = SessionFacts.from_dict(raw)
    facts.outcome.git_commits_after = _git_commits_after(session)
    return facts


def _git_commits_after(session: Session) -> int:
    """
    Local outcome evidence: commits in the session's repo from session start
    to 48h after its last activity. -1 when there is no repo or no cwd.
    Read-only `git log`; nothing leaves the machine.
    """
    import subprocess
    from datetime import timedelta

    cwd = getattr(session, "cwd", None)
    if not cwd or not Path(cwd).is_dir():
        return -1
    start = getattr(session, "started_at", None) or session.file_mtime
    end = session.file_mtime + timedelta(hours=48)
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "log", "--oneline", "--all",
             f"--since={start.isoformat(timespec='seconds')}",
             f"--until={end.isoformat(timespec='seconds')}"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return -1
    if out.returncode != 0:
        return -1
    return sum(1 for line in out.stdout.splitlines() if line.strip())


@dataclass
class ExtractionResult:
    facts: SessionFacts | None
    cached: bool = False
    skipped_reason: str | None = None
    error: str | None = None
    truncated: bool = False


def _cache_hit(
    session: Session, index: dict[str, _IndexEntry], force: bool,
) -> ExtractionResult | None:
    """Return the cached result if this session is fresh, else None."""
    if force:
        return None
    short = short_project_name(session.project_folder)
    facts_path = _sessions_dir(short) / f"{session.session_id}.json"
    entry = index.get(session.session_id)
    from ..sources import source_revision
    if not (entry and entry.status.startswith("ok") and _fingerprint_matches(session, entry)):
        return None
    if entry.parser_rev < source_revision(session.source):
        return None  # the parser for this source changed since the review
    if not facts_path.exists():
        return None
    try:
        raw = json.loads(facts_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
        return None  # older prompt/schema: re-extract
    return ExtractionResult(facts=SessionFacts.from_dict(raw), cached=True, truncated=entry.truncated)


def _finish(
    session: Session,
    raw_response: str | None,
    err: Exception | None,
    *,
    provider_name: str,
    model: str,
    truncated: bool,
    index: dict[str, _IndexEntry],
) -> ExtractionResult:
    """Parse, validate and persist one LLM response (live or batch)."""
    short = short_project_name(session.project_folder)
    if raw_response is None:
        err_msg = f"{type(err).__name__}: {err}" if err else "unknown error"
        _record_failure(short, session, provider_name, model, truncated, err_msg, index)
        return ExtractionResult(facts=None, error=err_msg, truncated=truncated)

    parsed = parse_json_lenient(raw_response)
    if parsed is None:
        raw_sidecar = _sessions_dir(short) / f"{session.session_id}.raw.txt"
        raw_sidecar.write_text(raw_response, encoding="utf-8")
        err_msg = f"bad_json (raw saved to {raw_sidecar.name})"
        _record_failure(short, session, provider_name, model, truncated, err_msg, index)
        return ExtractionResult(facts=None, error=err_msg, truncated=truncated)

    try:
        facts = _sanitize_facts(parsed, session)
    except Exception as e:  # noqa: BLE001 - one odd response must not kill a 1000-session run
        raw_sidecar = _sessions_dir(short) / f"{session.session_id}.raw.txt"
        raw_sidecar.write_text(raw_response, encoding="utf-8")
        err_msg = f"bad_shape: {type(e).__name__}: {e}"
        _record_failure(short, session, provider_name, model, truncated, err_msg, index)
        return ExtractionResult(facts=None, error=err_msg, truncated=truncated)
    issues = facts.validate()
    # Soft failure: save the file anyway but flag in index so aggregators can
    # choose to skip. Validators mostly catch LLM slop (unknown dims,
    # out-of-range msg_idx) which we'd rather note than lose entirely.
    status = "ok" if not issues else "ok_with_warnings"
    atomic_write_json(_sessions_dir(short) / f"{session.session_id}.json", facts.to_dict())
    index[session.session_id] = _IndexEntry(
        mtime_ns=getattr(session, "_mtime_ns", 0),
        byte_size=getattr(session, "_byte_size", 0),
        message_count=session.message_count,
        last_processed_at=datetime.now().isoformat(timespec="seconds"),
        provider=provider_name,
        model=model,
        truncated=truncated,
        status=status,
        parser_rev=_parser_rev(session),
    )
    return ExtractionResult(facts=facts, cached=False, truncated=truncated)


def _parser_rev(session: Session) -> int:
    from ..sources import source_revision
    return source_revision(session.source)


def extract_session(
    session: Session,
    provider: LLMProvider,
    *,
    model: str | None = None,
    force: bool = False,
    min_messages: int = MIN_MESSAGES,
    index: dict[str, _IndexEntry] | None = None,
) -> ExtractionResult:
    """
    Extract facts for one session with a live LLM call. Honors the
    per-project cache unless `force=True`. `index` may be passed in to
    amortize index I/O across many sessions in the same project.
    """
    short = short_project_name(session.project_folder)
    if session.message_count < min_messages:
        return ExtractionResult(facts=None, skipped_reason=f"too_short ({session.message_count})")

    owns_index = index is None
    if index is None:
        index = _load_index(short)

    hit = _cache_hit(session, index, force)
    if hit is not None:
        return hit

    prompt, truncated = _build_prompt(session)
    system = stage1_system_prompt()

    def _do_call() -> str:
        return provider.analyze(prompt, model=model, system=system)

    raw_response, err = call_with_retries(_do_call, max_attempts=3)
    result = _finish(
        session, raw_response, err,
        provider_name=provider.info.name, model=model or provider.info.default_model,
        truncated=truncated, index=index,
    )
    if owns_index:
        _save_index(short, index)
    return result


# One batch is capped at 100k requests / 256 MB. A prompt is at most ~100 KB,
# so 1,500 requests stay far below the size limit.
BATCH_CHUNK = 1500
PENDING_BATCHES_FILE = "pending_batches.json"


def _batch_custom_id(session: Session) -> str:
    """Stable across runs: same session + same content -> same id."""
    import hashlib
    key = f"{session.source}|{session.project_folder}|{session.session_id}|{getattr(session, '_mtime_ns', 0)}"
    return "v" + hashlib.sha1(key.encode()).hexdigest()[:40]


def _pending_path() -> Path:
    from ..paths import get_cache_dir
    return get_cache_dir() / PENDING_BATCHES_FILE


def _load_pending_batches() -> list[dict]:
    try:
        data = json.loads(_pending_path().read_text(encoding="utf-8"))
        return [b for b in data if isinstance(b, dict) and b.get("batch_id")]
    except (OSError, ValueError):
        return []


def _save_pending_batches(batches: list[dict]) -> None:
    atomic_write_json(_pending_path(), batches)


def extract_sessions_batch(
    sessions_by_project: dict[str, list[Session]],
    provider: LLMProvider,
    *,
    model: str | None = None,
    force: bool = False,
    on_status=None,
    should_stop=None,
) -> dict[str, "ProjectExtractionStats"]:
    """
    Stage 1 for many projects as provider batches (Claude Message Batches:
    50% cheaper). Cached and too-short sessions never reach a batch.

    Batch ids are saved to cache/pending_batches.json the moment they are
    submitted. If the run stops (Ctrl-C, closed terminal, crash), the next
    run collects those results instead of paying for the same sessions twice.
    """
    indexes = {short: _load_index(short) for short in sessions_by_project}
    stats = {
        short: ProjectExtractionStats(project_short=short, total_sessions=len(ss))
        for short, ss in sessions_by_project.items()
    }
    pending: dict[str, tuple[Session, str, bool]] = {}
    system = stage1_system_prompt()
    for short, ss in sessions_by_project.items():
        st = stats[short]
        for session in ss:
            if session.message_count < MIN_MESSAGES:
                st.skipped += 1
                continue
            hit = _cache_hit(session, indexes[short], force)
            if hit is not None:
                st.ok += 1
                st.cached += 1
                continue
            prompt, truncated = _build_prompt(session)
            pending.setdefault(_batch_custom_id(session), (session, prompt, truncated))

    model_name = model or provider.info.default_model
    done: set[str] = set()

    def _apply(responses: dict) -> None:
        for cid, resp in responses.items():
            item = pending.get(cid)
            if item is None or cid in done:
                continue
            session, _, truncated = item
            short = short_project_name(session.project_folder)
            text, err = (None, resp) if isinstance(resp, Exception) else (resp, None)
            result = _finish(
                session, text, err, provider_name=provider.info.name, model=model_name,
                truncated=truncated, index=indexes[short],
            )
            done.add(cid)
            st = stats[short]
            if result.facts is None:
                st.failed += 1
                if result.error:
                    st.errors.append(f"{session.session_id}: {result.error}")
            else:
                st.ok += 1
            if truncated:
                st.truncated_calls += 1

    supports_resume = all(hasattr(provider, a) for a in ("batch_submit", "batch_wait", "batch_results"))
    try:
        if supports_resume:
            # 1. Collect batches an earlier run submitted but never read.
            from ..providers import BatchGoneError
            held: set[str] = set()   # sessions in a batch we could not read now
            for b in _load_pending_batches():
                ids = [cid for cid in b.get("custom_ids", []) if cid in pending]
                if not ids:
                    continue
                if on_status:
                    on_status(f"resuming batch {b['batch_id']} from an earlier run")
                try:
                    provider.batch_wait(b["batch_id"], on_status=on_status, should_stop=should_stop)
                    _apply(provider.batch_results(
                        b["batch_id"], {cid: (cid, pending[cid][1], system) for cid in ids}, model=model))
                except BatchGoneError as e:
                    # Expired or from another account: forget it and review those sessions again.
                    if on_status:
                        on_status(f"{e}; its sessions will be sent again")
                except Exception as e:  # noqa: BLE001 - network etc.: keep it for the next run
                    held.update(ids)
                    if on_status:
                        on_status(f"could not read batch {b['batch_id']} now ({e}); will retry next run")
                    continue
                _save_pending_batches([x for x in _load_pending_batches() if x["batch_id"] != b["batch_id"]])
            # 2. Submit what is left, in chunks; save each id before waiting.
            # Sessions held in an unreadable batch are not paid for twice.
            todo = [cid for cid in pending if cid not in done and cid not in held]
            for i in range(0, len(todo), BATCH_CHUNK):
                if should_stop and should_stop():
                    break
                chunk = todo[i:i + BATCH_CHUNK]
                items = [(cid, pending[cid][1], system) for cid in chunk]
                batch_id = provider.batch_submit(items, model=model)
                _save_pending_batches(_load_pending_batches() + [{
                    "batch_id": batch_id, "custom_ids": chunk,
                    "submitted_at": datetime.now().isoformat(timespec="seconds"),
                }])
                if on_status:
                    on_status(f"batch {batch_id} submitted ({len(chunk)} requests)")
                provider.batch_wait(batch_id, on_status=on_status, should_stop=should_stop)
                _apply(provider.batch_results(
                    batch_id, {cid: (cid, pending[cid][1], system) for cid in chunk}, model=model))
                _save_pending_batches([x for x in _load_pending_batches() if x["batch_id"] != batch_id])
        elif pending:
            _apply(provider.batch_analyze(
                [(cid, p, system) for cid, (_, p, _) in pending.items()],
                model=model, on_status=on_status, should_stop=should_stop,
            ))
    finally:
        # Anything without a result (cancelled, expired, stopped early) is
        # counted, so totals always add up. It is retried on the next run.
        for cid, (session, _, _) in pending.items():
            if cid not in done:
                st = stats[short_project_name(session.project_folder)]
                st.failed += 1
                st.errors.append(f"{session.session_id}: no result yet (stopped, cancelled or expired)")
        for short, idx in indexes.items():
            _save_index(short, idx)
    return stats


def _record_failure(
    project_short: str,
    session: Session,
    provider_name: str,
    model: str,
    truncated: bool,
    err_msg: str,
    index: dict[str, _IndexEntry],
) -> None:
    index[session.session_id] = _IndexEntry(
        mtime_ns=getattr(session, "_mtime_ns", 0),
        byte_size=getattr(session, "_byte_size", 0),
        message_count=session.message_count,
        last_processed_at=datetime.now().isoformat(timespec="seconds"),
        provider=provider_name,
        model=model,
        truncated=truncated,
        status=f"failed:{err_msg[:100]}",
    )


@dataclass
class ProjectExtractionStats:
    project_short: str
    total_sessions: int = 0
    ok: int = 0
    cached: int = 0
    skipped: int = 0
    failed: int = 0
    truncated_calls: int = 0
    errors: list[str] = field(default_factory=list)


def extract_project_sessions(
    project_short: str,
    sessions: list[Session],
    provider: LLMProvider,
    *,
    model: str | None = None,
    force: bool = False,
    on_progress=None,
    should_stop=None,
) -> ProjectExtractionStats:
    """
    Extract all sessions for one project. Batches index I/O so we do at most
    one read + one write per project even with hundreds of sessions.

    `on_progress`, if given, is called as `on_progress(done, total, result)`.
    """
    stats = ProjectExtractionStats(project_short=project_short, total_sessions=len(sessions))
    index = _load_index(project_short)

    try:
        for i, session in enumerate(sessions, start=1):
            if should_stop is not None and should_stop():
                break  # Ctrl-C: no new LLM calls; what is done stays cached
            result = extract_session(
                session, provider, model=model, force=force, index=index,
            )
            if result.facts is None:
                if result.skipped_reason:
                    stats.skipped += 1
                else:
                    stats.failed += 1
                    if result.error:
                        stats.errors.append(
                            f"{session.session_id}: {result.error}"
                        )
            else:
                stats.ok += 1
                if result.cached:
                    stats.cached += 1
            if result.truncated:
                stats.truncated_calls += 1

            if on_progress:
                try:
                    on_progress(i, len(sessions), result)
                except Exception:  # noqa: BLE001
                    # Never let a progress hook crash the pipeline.
                    pass
    finally:
        # Save index even if we bail mid-loop — partial progress > lost work.
        _save_index(project_short, index)

    return stats


def load_project_session_facts(project_short: str) -> list[SessionFacts]:
    """
    Load all on-disk SessionFacts for a project. Used by Stage 2+.
    Silently skips unreadable files so one bad write doesn't tank a run.
    """
    d = _sessions_dir(project_short)
    out: list[SessionFacts] = []
    for p in sorted(d.glob("*.json")):
        if p.name == INDEX_FILE:
            continue
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if int(raw.get("schema_version", 0)) != SCHEMA_VERSION:
                continue  # older prompt/schema
            out.append(SessionFacts.from_dict(raw))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  warn: couldn't read {p.name}: {e}", file=sys.stderr)
    return out


__all__ = [
    "ExtractionResult",
    "ProjectExtractionStats",
    "extract_session",
    "extract_project_sessions",
    "extract_sessions_batch",
    "stage1_system_prompt",
    "load_project_session_facts",
    "MIN_MESSAGES",
]
