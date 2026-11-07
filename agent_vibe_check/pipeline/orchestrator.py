"""
Pipeline orchestrator.

Drives the four stages end-to-end so the TUI/CLI doesn't have to care about
ordering or resume semantics. Each stage writes atomically, so interrupting
anywhere and re-running picks up right where we left off (no re-sending an
already-extracted session to the LLM).

Entry point `run_full_pipeline` is idempotent: given unchanged inputs it will
spend zero tokens and zero seconds of provider latency.

Single-project variant `run_one_project` is what "analyze one project" in the
TUI calls.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from ..extract import (
    iter_matching_sessions,
    project_stats,
    qualifies_for_deep_analysis,
    qualifying_projects,
    short_project_name,
)
from ..parser import Session
from ..providers import LLMProvider
from ..atlas import _build_workspace_map
from .cross_project import build_cross_project
from .project_synthesizer import (
    load_project_synthesis,
    synthesize_project,
)
from .schemas import ProjectSynthesis, SessionFacts
from .profile import build_profile
from .session_extractor import (
    ProjectExtractionStats,
    extract_project_sessions,
    extract_sessions_batch,
    load_project_session_facts,
)
from .timeline import build_timeline

ProgressHook = Callable[[str, dict], None]

DEFAULT_MAX_WORKERS = 4


def _noop_progress(event: str, data: dict) -> None:  # pragma: no cover - default
    return


class CancellationToken:
    """
    Cooperative cancellation flag.

    The TUI sets this on Ctrl-C; workers check it at safe boundaries and
    skip new work when tripped. In-flight LLM calls are allowed to finish
    (we can't yank a blocking HTTP request cleanly, and atomic writes mean
    their output is still useful on the next resume). A second Ctrl-C
    restores Python's default SIGINT handler so the user can still hard-
    kill the process if a call really hangs.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ProjectRunResult:
    project_short: str
    project_folder: str
    stats: ProjectExtractionStats
    synthesis: ProjectSynthesis | None = None
    error: str | None = None


@dataclass
class PipelineRunResult:
    per_project: list[ProjectRunResult] = field(default_factory=list)
    all_session_facts: list[SessionFacts] = field(default_factory=list)
    all_syntheses: list[ProjectSynthesis] = field(default_factory=list)
    timeline_cached: bool = False
    cross_cached: bool = False
    errors: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bucket_sessions_by_short() -> dict[str, list[Session]]:
    """
    Single pass over every data source, grouped by short project name.
    Replaces the N-projects-x-full-rescan loop.
    """
    buckets: dict[str, list[Session]] = {}
    for s in iter_matching_sessions(None):
        key = short_project_name(s.project_folder)
        buckets.setdefault(key, []).append(s)
    return buckets


def _pick_project_folder(sessions: list[Session]) -> str:
    """
    A single "short name" can come from multiple source folders (JSONL vs
    composer DB). Pick the one that appears most so the project synthesis has
    a stable `project_folder` key and the workspace-path resolver has a
    chance to hit.
    """
    counts: dict[str, int] = {}
    for s in sessions:
        counts[s.project_folder] = counts.get(s.project_folder, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


# ---------------------------------------------------------------------------
# Per-project run
# ---------------------------------------------------------------------------


def run_one_project(
    project_short: str,
    sessions: list[Session],
    provider: LLMProvider,
    *,
    model: str | None = None,
    force: bool = False,
    progress: ProgressHook = _noop_progress,
    cancel_token: CancellationToken | None = None,
    stage1_stats: ProjectExtractionStats | None = None,
) -> ProjectRunResult:
    """
    Stage 1 + Stage 2 for one project. Pass `stage1_stats` when Stage 1
    already ran (batch mode) to go straight to Stage 2. Stages 3-4 are global; call those
    separately from `run_full_pipeline`.

    If `cancel_token` is set before we start, we return immediately with
    error="cancelled" so the worker pool drains without submitting more
    LLM calls. We do NOT check mid-stage: once Stage 1 starts its inner
    loop, it runs to completion (its own atomic writes mean partial
    progress is preserved for the next run).
    """
    if cancel_token is not None and cancel_token.is_cancelled:
        empty_stats = ProjectExtractionStats(project_short=project_short)
        return ProjectRunResult(
            project_short=project_short,
            project_folder="",
            stats=empty_stats,
            error="cancelled",
        )

    if not sessions:
        empty_stats = ProjectExtractionStats(project_short=project_short)
        return ProjectRunResult(
            project_short=project_short,
            project_folder="",
            stats=empty_stats,
            error="no sessions",
        )

    project_folder = _pick_project_folder(sessions)

    # Stage 1
    def _on_stage1_progress(done: int, total: int, _result) -> None:
        progress("stage1", {
            "project": project_short, "done": done, "total": total,
        })

    if stage1_stats is not None:
        stats = stage1_stats  # batch mode already reported stage1_done
    else:
        stats = extract_project_sessions(
            project_short, sessions, provider,
            model=model, force=force, on_progress=_on_stage1_progress,
            should_stop=(lambda: cancel_token.is_cancelled) if cancel_token is not None else None,
        )
        if cancel_token is not None and cancel_token.is_cancelled:
            return ProjectRunResult(project_short=project_short, project_folder=project_folder,
                                    stats=stats, error="cancelled")
        progress("stage1_done", {
            "project": project_short,
            "ok": stats.ok, "cached": stats.cached,
            "failed": stats.failed, "skipped": stats.skipped,
        })

    # Stage 2: only facts for sessions that are still in an enabled source
    # (a source turned off, or a session now skipped, must not count).
    current_ids = {sess.session_id for sess in sessions}
    facts = [f for f in load_project_session_facts(project_short) if f.session_id in current_ids]
    if not facts:
        return ProjectRunResult(
            project_short=project_short,
            project_folder=project_folder,
            stats=stats,
            error="no session facts produced",
        )

    progress("stage2", {"project": project_short, "facts": len(facts)})
    # Claude Code / Codex / Gemini CLI record the real working directory.
    cwds: dict[str, int] = {}
    for sess in sessions:
        if getattr(sess, "cwd", None):
            cwds[sess.cwd] = cwds.get(sess.cwd, 0) + 1
    real_path_hint = max(cwds.items(), key=lambda kv: kv[1])[0] if cwds else None
    syn_result = synthesize_project(
        project_short, project_folder, facts, provider,
        model=model, force=force, min_sessions=1, real_path_hint=real_path_hint,
    )
    progress("stage2_done", {
        "project": project_short,
        "cached": syn_result.cached,
        "synthesis": syn_result.synthesis is not None,
    })

    if syn_result.synthesis is None:
        return ProjectRunResult(
            project_short=project_short,
            project_folder=project_folder,
            stats=stats,
            error=syn_result.error or syn_result.skipped_reason or "stage 2 failed",
        )

    return ProjectRunResult(
        project_short=project_short,
        project_folder=project_folder,
        stats=stats,
        synthesis=syn_result.synthesis,
    )


# ---------------------------------------------------------------------------
# Full portfolio run
# ---------------------------------------------------------------------------


def select_portfolio(
    *,
    min_session_messages: int = 10,
    min_total_messages: int = 20,
) -> list[str]:
    """Return short names of every project that qualifies for deep analysis."""
    stats = project_stats()
    picked = qualifying_projects(
        stats,
        min_session_messages=min_session_messages,
        min_total_messages=min_total_messages,
    )
    return [p.short_name for p in picked]


def run_full_pipeline(
    provider: LLMProvider,
    *,
    model: str | None = None,
    portfolio: list[str] | None = None,
    force: bool = False,
    progress: ProgressHook = _noop_progress,
    max_workers: int = DEFAULT_MAX_WORKERS,
    cancel_token: CancellationToken | None = None,
    use_batch: bool = False,
    max_sessions_per_project: int | None = None,
) -> PipelineRunResult:
    """
    Full end-to-end run.

    - `portfolio`: specific project short names to include. If None, auto-
      selects via the richness filter.
    - `force`: re-run every stage even if fingerprints match.
    - `max_workers`: parallel workers for per-project Stage 1+2. Default 4.
      Global Stages 3 and 4 are always single-threaded (they need all
      per-project outputs and are only one LLM call each, so there's
      nothing to parallelize).
    - `cancel_token`: cooperative cancellation. Workers that haven't
      started skip immediately; in-flight workers finish their current
      Stage 1+2 so atomic writes complete. Global stages are skipped if
      the token is tripped before they start.
    - `max_sessions_per_project`: quick mode, keep only the N most recent
      sessions of each project (cheaper first look).
    - `use_batch`: run Stage 1 for every project in ONE provider batch
      (Claude Message Batches: 50% cheaper, results usually within an
      hour). Falls back to live calls when the provider has no batch API.
    - `progress`: callback for UI updates. Events:
        "batch_status"     {message}
        "portfolio"        {count, projects, max_workers}
        "project_start"    {project, i, total, sessions}
        "stage1_done"      {project, ok, cached, failed, skipped}
        "stage2_done"      {project, cached, synthesis}
        "project_done"     {project, status}
        "cancelled"        {remaining}
        "stage3"           {}
        "stage3_done"      {cached}
        "stage4"           {}
        "stage4_done"      {cached}

    Thread safety notes:
      - `progress` may be called concurrently from worker threads. The
        default stdout/stderr printers are fine; rich's Console takes an
        internal lock. If you pass your own hook, make it thread-safe.
      - Append-to-list in this function is only done from the main thread
        (we collect results via as_completed).
    """
    result = PipelineRunResult()

    portfolio = portfolio or select_portfolio()
    progress("portfolio", {
        "count": len(portfolio),
        "projects": portfolio,
        "max_workers": max_workers,
    })
    if not portfolio:
        result.errors.append("no projects qualify for deep analysis")
        return result

    buckets = _bucket_sessions_by_short()
    if max_sessions_per_project:
        # Quick mode: only the most recent N sessions per project.
        def _when(sess: Session):
            return sess.started_at or sess.file_mtime
        buckets = {
            k: sorted(v, key=_when, reverse=True)[:max_sessions_per_project]
            for k, v in buckets.items()
        }

    # ----- Optional: Stage 1 for everything in one batch -----
    batch_stats: dict[str, ProjectExtractionStats] = {}
    if use_batch and getattr(provider, "supports_batch", False):
        progress("batch_status", {"message": "preparing Stage 1 batch"})
        try:
            batch_stats = extract_sessions_batch(
                {short: buckets.get(short, []) for short in portfolio},
                provider, model=model, force=force,
                on_status=lambda m: progress("batch_status", {"message": m}),
                should_stop=(lambda: cancel_token.is_cancelled) if cancel_token else None,
            )
        except Exception as e:  # noqa: BLE001 - batch ids are saved; next run resumes
            result.errors.append(f"stage 1 batch: {type(e).__name__}: {e}")
            progress("batch_status", {"message": f"batch failed: {e}. Re-run to resume."})
            return result
        for short, st in batch_stats.items():
            progress("stage1_done", {
                "project": short, "ok": st.ok, "cached": st.cached,
                "failed": st.failed, "skipped": st.skipped,
            })

    # ----- Per-project Stage 1 + 2 (parallel across projects) -----

    # project_start events fire on submission so the user sees the queue
    # fill up at dispatch time. Inner stage_done events fire as they
    # naturally complete in each worker.
    def _run(short: str, i: int) -> ProjectRunResult:
        sessions = buckets.get(short, [])
        progress("project_start", {
            "project": short, "i": i, "total": len(portfolio),
            "sessions": len(sessions),
        })
        pr = run_one_project(
            short, sessions, provider,
            model=model, force=force,
            progress=progress, cancel_token=cancel_token,
            stage1_stats=batch_stats.get(short),
        )
        progress("project_done", {
            "project": short,
            "status": "ok" if pr.synthesis else ("cancelled" if pr.error == "cancelled" else "failed"),
        })
        return pr

    # Serial fallback for tests / max_workers=1. Keeps ordering deterministic
    # and avoids thread overhead when parallelism isn't wanted.
    if max_workers <= 1:
        for i, short in enumerate(portfolio, start=1):
            if cancel_token is not None and cancel_token.is_cancelled:
                progress("cancelled", {"remaining": len(portfolio) - i + 1})
                break
            pr = _run(short, i)
            result.per_project.append(pr)
            if pr.synthesis is not None:
                result.all_syntheses.append(pr.synthesis)
    else:
        pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="vibe-proj")
        try:
            futures = {
                pool.submit(_run, short, i): short
                for i, short in enumerate(portfolio, start=1)
            }
            cancelled_logged = False
            for fut in as_completed(futures):
                try:
                    pr = fut.result()
                except Exception as e:  # noqa: BLE001 - one project must not kill the run
                    short = futures[fut]
                    pr = ProjectRunResult(project_short=short, project_folder="",
                                          stats=ProjectExtractionStats(project_short=short),
                                          error=f"{type(e).__name__}: {e}")
                    result.errors.append(f"{short}: {pr.error}")
                result.per_project.append(pr)
                if pr.synthesis is not None:
                    result.all_syntheses.append(pr.synthesis)
                if cancel_token is not None and cancel_token.is_cancelled and not cancelled_logged:
                    cancelled_logged = True
                    progress("cancelled", {
                        "remaining": sum(1 for pp in result.per_project if pp.error == "cancelled"),
                    })
        except KeyboardInterrupt:
            # Second Ctrl-C: drop queued projects and do not wait for running ones.
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            pool.shutdown(wait=True)

    # ----- Global stages (serial) -----

    if cancel_token is not None and cancel_token.is_cancelled:
        # Skip expensive global stages on cancellation. Reports from
        # completed projects are still written by the caller.
        return result

    for short in portfolio:
        current_ids = {sess.session_id for sess in buckets.get(short, [])}
        result.all_session_facts.extend(
            f for f in load_project_session_facts(short) if f.session_id in current_ids
        )

    progress("stage3", {})
    if result.all_session_facts:
        tl_res = build_timeline(
            result.all_session_facts, provider, model=model, force=force,
        )
        result.timeline_cached = tl_res.cached
        progress("stage3_done", {"cached": tl_res.cached})

    progress("stage4", {})
    if result.all_syntheses:
        cp_res = build_cross_project(
            result.all_syntheses, provider, model=model, force=force,
        )
        result.cross_cached = cp_res.cached
        progress("stage4_done", {"cached": cp_res.cached})

    progress("stage5", {})
    if result.all_session_facts:
        pf = build_profile(
            result.all_session_facts, result.all_syntheses, provider, model=model, force=force,
        )
        if pf.error:
            result.errors.append(f"profile: {pf.error}")
        progress("stage5_done", {"cached": pf.cached})

    return result


# ---------------------------------------------------------------------------
# Render step (no LLM, no cost)
# ---------------------------------------------------------------------------


def write_all_reports(
    syntheses: list[ProjectSynthesis],
    session_facts: list[SessionFacts],
) -> dict[str, Path]:
    """
    Render every human-facing artifact: per-project PROJECT.md, global
    timeline/cross/exec-summary markdown, and the HTML dashboard.
    """
    from ..renderers.markdown import write_global_reports, write_project_reports
    from ..renderers.dashboard import write_dashboard
    from .cross_project import load_cross_project
    from .timeline import load_timeline

    outputs: dict[str, Path] = {}
    for syn in syntheses:
        paths = write_project_reports(syn.project_short, syn)
        outputs[f"project:{syn.project_short}"] = paths["project"]

    from .profile import load_profile
    from ..renderers.share import ShareLeakError, write_share

    tl = load_timeline()
    cross = load_cross_project()
    profile = load_profile()

    gpaths = write_global_reports(tl, cross, syntheses, session_facts, profile=profile)
    outputs.update({f"global:{k}": v for k, v in gpaths.items()})

    dashboard_path = write_dashboard(syntheses, tl, cross, session_facts, profile=profile)
    outputs["dashboard"] = dashboard_path

    if profile:
        names = share_name_list(syntheses, session_facts)
        try:
            spaths = write_share(profile, names, person_names=_person_names())
            outputs.update({f"share:{k}": v for k, v in spaths.items()})
        except ShareLeakError as e:
            outputs["share_blocked"] = Path(str(e))

    return outputs


def _person_names() -> list[str]:
    """The user's own names (git user.name, login), removed from shared files."""
    import getpass
    import subprocess
    names: set[str] = set()
    try:
        out = subprocess.run(["git", "config", "--global", "user.name"], capture_output=True, text=True, timeout=5)
        full = out.stdout.strip()
        if full:
            names.add(full)
            # Single parts of 4+ letters only, so short names do not erase common words.
            names.update(part for part in full.split() if len(part) >= 4)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        names.add(getpass.getuser())
    except Exception:  # noqa: BLE001
        pass
    names.add(Path.home().name)
    return sorted(n for n in names if n and len(n) >= 3)


def share_name_list(syntheses, session_facts) -> list[str]:
    """
    Every name that can identify a project, for the share export: short names
    of everything ever analyzed (not just this run's), and the folder names of
    real paths.
    """
    from ..paths import get_extractions_dir
    names = {s.project_short for s in syntheses} | {f.project_short for f in session_facts}
    root = get_extractions_dir()
    if root.exists():
        names |= {p.name for p in root.iterdir() if p.is_dir()}
    for s in syntheses:
        if s.real_path:
            names.add(Path(s.real_path).name)
    return sorted(n for n in names if n)


def load_all_syntheses(portfolio: Iterable[str] | None = None) -> list[ProjectSynthesis]:
    """
    Load every existing per-project synthesis from disk. Used by the
    dashboard-only path (no LLM calls, no re-analysis).
    """
    from ..paths import get_extractions_dir
    root = get_extractions_dir()
    if not root.exists():
        return []
    shorts = list(portfolio) if portfolio is not None else [p.name for p in root.iterdir() if p.is_dir()]
    from .schemas import SCHEMA_VERSION
    out: list[ProjectSynthesis] = []
    for s in shorts:
        syn = load_project_synthesis(s)
        # Skip outputs from older schema versions (e.g. v1 code-skill reports).
        if syn is not None and syn.schema_version == SCHEMA_VERSION:
            out.append(syn)
    return out


def load_all_session_facts(portfolio: Iterable[str] | None = None) -> list[SessionFacts]:
    from ..paths import get_extractions_dir
    root = get_extractions_dir()
    if not root.exists():
        return []
    shorts = list(portfolio) if portfolio is not None else [p.name for p in root.iterdir() if p.is_dir()]
    from .schemas import SCHEMA_VERSION
    out: list[SessionFacts] = []
    for s in shorts:
        out.extend(f for f in load_project_session_facts(s) if f.schema_version == SCHEMA_VERSION)
    return out


__all__ = [
    "CancellationToken",
    "DEFAULT_MAX_WORKERS",
    "PipelineRunResult",
    "ProjectRunResult",
    "select_portfolio",
    "run_one_project",
    "run_full_pipeline",
    "write_all_reports",
    "load_all_syntheses",
    "load_all_session_facts",
]
