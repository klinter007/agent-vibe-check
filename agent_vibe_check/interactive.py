#!/usr/bin/env python3
"""
Vibe Check - Interactive TUI.

Thin shell around the four-stage pipeline (see agent_vibe_check.pipeline).
The pipeline is the source of truth; this module only handles prompts,
progress rendering, and opening output files.
"""

from __future__ import annotations

import os
import random
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich import box

from .extract import (
    DEFAULT_MIN_SESSION_MESSAGES,
    DEFAULT_MIN_TOTAL_MESSAGES,
    iter_matching_sessions,
    project_stats,
    qualifying_projects,
    short_project_name,
)
from .paths import (
    check_cursor_installation,
    get_cursor_projects_dir,
    get_extractions_dir,
)
from .providers import (
    ProviderError,
    build_provider,
    detect_available,
    get_provider_info,
    list_providers,
)
from . import config as vibe_config
from . import sources
from . import schema as schema_check
from .pipeline import orchestrator as orch

console = Console()

COLORS = {
    "title": "bold #D4A574",
    "subtitle": "#A8A8A8",
    "accent": "#7EB8DA",
    "success": "#8FBC8F",
    "warning": "#DAA06D",
    "error": "#CD5C5C",
    "muted": "#6B6B6B",
    "highlight": "#B8860B",
}


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header():
    title = Text()
    title.append("VIBE CHECK\n", style=COLORS["title"])
    title.append(
        "What you built. What you learned. What to fix next.",
        style=COLORS["subtitle"],
    )
    console.print()
    console.print(
        Panel(title, box=box.DOUBLE, padding=(1, 4), border_style=COLORS["accent"])
    )
    console.print()


def print_separator():
    console.print(f"  [dim]{'─' * 55}[/dim]")


def wait_for_key(message: str = "Press Enter to continue..."):
    console.print()
    console.input(f"  [{COLORS['muted']}]{message}[/]")


# ---------------------------------------------------------------------------
# Rolling ticker
# ---------------------------------------------------------------------------

_TICKER_MESSAGES = [
    "Sampling conversations...",
    "Counting your thoughts...",
    "Weighing past decisions...",
    "Scoring substance...",
    "Checking the vibe...",
    "Finding rich conversations...",
    "Ignoring typo fixes...",
    "Measuring session depth...",
    "Judging your patience...",
    "Tallying back-and-forths...",
    "Reading composer data...",
    "Bucketing projects...",
    "Resolving workspace paths...",
    "Scanning bubble blobs...",
    "Separating signal from noise...",
    "Finding the gold...",
    "Forgetting abandoned ideas...",
    "Excavating old projects...",
    "Measuring your verbosity...",
    "Reassembling past reasoning...",
    "Rediscovering forgotten work...",
    "Locating missing context...",
    "Evaluating chat substance...",
    "Ranking your best sessions...",
    "Filtering small talk...",
    "Summoning your past self...",
    "Identifying deep dives...",
    "Timestamping memories...",
    "Counting hard-won lessons...",
    "Grouping by richness...",
]

_SPINNER_FRAMES = ["|", "/", "-", "\\"]


def rolling_status(label: str, worker):
    """Run `worker()` in a thread, show a rolling ticker until it returns."""
    result: dict = {}
    done = threading.Event()

    def _runner():
        try:
            result["value"] = worker()
        except BaseException as e:  # noqa: BLE001
            result["error"] = e
        finally:
            done.set()

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    messages = list(_TICKER_MESSAGES)
    random.shuffle(messages)
    idx = 0
    start = time.monotonic()
    with Live("", console=console, refresh_per_second=6, transient=True) as live:
        while not done.is_set():
            msg = messages[idx // 3 % len(messages)]
            frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
            text = Text()
            text.append(f"  {frame} ", style=COLORS["accent"])
            text.append(label, style=COLORS["muted"])
            text.append("  ")
            text.append(msg, style=COLORS["subtitle"])
            elapsed = time.monotonic() - start
            if elapsed > 2:
                text.append(f"  ({elapsed:0.1f}s)", style=COLORS["muted"])
            live.update(text)
            done.wait(0.15)
            idx += 1
    t.join()
    if "error" in result:
        raise result["error"]
    return result["value"]


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def get_input(prompt: str, default: str = "") -> str:
    if default:
        result = console.input(
            f"  [{COLORS['accent']}]{prompt}[/] "
            f"[{COLORS['muted']}][{default}][/]: "
        ).strip()
        return result if result else default
    return console.input(f"  [{COLORS['accent']}]{prompt}[/]: ").strip()


def select_from_list(items: list, prompt: str = "Select") -> int | None:
    console.print()
    for i, item in enumerate(items, 1):
        console.print(f"    [{COLORS['highlight']}]{i}.[/] {item}")
    console.print(f"    [{COLORS['muted']}]0. Cancel[/]")
    console.print()
    try:
        choice = int(get_input(prompt))
    except ValueError:
        return None
    if choice == 0:
        return None
    if 1 <= choice <= len(items):
        return choice - 1
    return None


# ---------------------------------------------------------------------------
# Provider selection
# ---------------------------------------------------------------------------


def select_provider():
    available = detect_available()

    if len(available) == 1:
        info = get_provider_info(available[0])
        console.print(
            f"  {info.display_name} key: [{COLORS['success']}]Found[/] "
            f"(${info.env_var})"
        )
        try:
            return build_provider(name=info.name)
        except ProviderError as e:
            console.print(f"  [{COLORS['error']}]{e}[/]")
            return None

    if len(available) > 1:
        console.print(
            f"  [{COLORS['subtitle']}]Multiple provider keys detected. Pick one:[/]"
        )
        for i, pname in enumerate(available, start=1):
            info = get_provider_info(pname)
            console.print(f"    [{COLORS['accent']}]{i}[/]. {info.display_name}")
        choice = get_input(f"Choice [1-{len(available)}, default 1]") or "1"
        try:
            idx = int(choice) - 1
            name = available[idx]
        except (ValueError, IndexError):
            name = available[0]
        try:
            return build_provider(name=name)
        except ProviderError as e:
            console.print(f"  [{COLORS['error']}]{e}[/]")
            return None

    console.print(f"  [{COLORS['warning']}]No LLM API key found.[/]")
    console.print()
    providers = list_providers()
    for i, info in enumerate(providers, start=1):
        console.print(
            f"    [{COLORS['accent']}]{i}[/]. {info.display_name} "
            f"[{COLORS['muted']}]({info.env_var})[/] - {info.signup_url}"
        )
    console.print()

    choice = get_input(f"Choose provider [1-{len(providers)}, default 1]") or "1"
    try:
        idx = int(choice) - 1
        info = providers[idx]
    except (ValueError, IndexError):
        info = providers[0]

    key = console.input(f"  [{COLORS['accent']}]Paste your {info.display_name} key (or Enter to skip)[/]: ", password=True).strip()
    if not key:
        return None

    # Offer to persist. Default yes: the whole point is zero-friction next run.
    persist_choice = (
        get_input("Save this key for next time? [Y/n]") or "y"
    ).strip().lower()
    if persist_choice not in ("n", "no"):
        try:
            vibe_config.save_api_key(info.env_var, key)
            console.print(
                f"  [{COLORS['success']}]Saved[/] to "
                f"[dim]{vibe_config._config_path()}[/] (only you can read it)."
            )
        except OSError as e:
            console.print(
                f"  [{COLORS['warning']}]Could not save: {e}. "
                f"Key still active for this session.[/]"
            )

    os.environ[info.env_var] = key
    try:
        return build_provider(name=info.name, api_key=key)
    except ProviderError as e:
        console.print(f"  [{COLORS['error']}]{e}[/]")
        return None


# ---------------------------------------------------------------------------
# Installation check
# ---------------------------------------------------------------------------


def _home_path(p: Path) -> str:
    try:
        return "~/" + str(p.relative_to(Path.home()))
    except ValueError:
        return str(p)


def check_installation() -> bool:
    """Print what each enabled source holds. False if nothing is readable."""
    enabled = sources.enabled_keys()
    if not enabled:
        console.print(
            f"  [{COLORS['error']}]No chat sources enabled. "
            f"Open Settings -> Sources to choose them.[/]"
        )
        return False
    infos = {i.key: i for i in sources.detect() if i.key in enabled}
    for key in enabled:
        info = infos.get(key)
        if info is None:
            continue
        color = COLORS["success"] if info.found else COLORS["warning"]
        console.print(
            f"  {info.label}: [{color}]{info.session_count} sessions[/] "
            f"[{COLORS['muted']}]{_home_path(info.roots[0])}[/]"
        )
    if not any(i.found for i in infos.values()):
        console.print(f"  [{COLORS['error']}]ERROR: no chat history found in the enabled sources.[/]")
        return False
    if "cursor" not in enabled:
        return True
    return _check_cursor_installation()


def _check_cursor_installation() -> bool:
    status = check_cursor_installation()
    installed_color = COLORS["success"] if status["installed"] else COLORS["error"]
    console.print(
        f"  Cursor Installed: "
        f"[{installed_color}]{'Yes' if status['installed'] else 'No'}[/]"
    )
    total = status.get("unique_projects_total") or max(
        status.get("projects_with_history", 0),
        status.get("projects_with_composer_db", 0),
    )
    console.print(
        f"  Projects With Chat History: [{COLORS['accent']}]{total}[/] "
        f"[{COLORS['muted']}]"
        f"(agent-transcripts: {status.get('projects_with_history', 0)}, "
        f"composer DB: {status.get('projects_with_composer_db', 0)})[/]"
    )
    console.print(
        f"  Conversation Data: "
        f"[{COLORS['accent']}]{status['global_db_size_mb']:.0f} MB[/]"
    )

    if not status["installed"]:
        console.print(f"  [{COLORS['warning']}]Cursor not found; other sources still work.[/]")
        return True

    try:
        result = schema_check.check_global(get_cursor_projects_dir(), sample_size=8)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [{COLORS['warning']}]Schema check skipped: {e}[/]")
        return True
    if result.total_lines == 0:
        console.print(f"  [{COLORS['muted']}]Schema check: no transcripts to sample[/]")
    elif result.ok:
        console.print(
            f"  Schema Check: [{COLORS['success']}]OK[/] "
            f"[{COLORS['muted']}]({result.summary})[/]"
        )
    else:
        console.print(f"  Schema Check: [{COLORS['warning']}]DRIFT[/]")
        console.print(f"  [{COLORS['muted']}]{result.summary}[/]")
        console.print(f"  [{COLORS['warning']}]{result.reason}[/]")
    return True


# ---------------------------------------------------------------------------
# Onboarding: which agent tools may we read?
# ---------------------------------------------------------------------------


def _yes(answer: str, default: bool = True) -> bool:
    a = answer.strip().lower()
    if not a:
        return default
    return a in ("y", "yes")


def run_onboarding() -> None:
    """
    Ask which AI coding tools the user works with and get permission per
    source before anything is read. Saves the choice to config.json.
    """
    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]WHICH AI CODING TOOLS DO YOU USE?[/]")
    print_separator()
    console.print()
    console.print(
        f"  [{COLORS['subtitle']}]Vibe Check reads chat history only from tools you allow.\n"
        f"  Files stay on this machine. Only the prompts sent to your chosen LLM leave it.[/]"
    )
    console.print()

    infos = sources.detect()
    for i, info in enumerate(infos, start=1):
        if info.found:
            status = f"[{COLORS['success']}]{info.session_count} transcripts found[/]"
        else:
            status = f"[{COLORS['muted']}]not found[/]"
        console.print(
            f"    [{COLORS['highlight']}]{i}.[/] [{COLORS['accent']}]{info.label}[/] "
            f"[{COLORS['muted']}]({info.vendor})[/]  {status}"
        )
        for root in info.roots:
            console.print(f"       [{COLORS['muted']}]{_home_path(root)}[/]")
        console.print(f"       [{COLORS['muted']}]{info.docs_url}[/]")
    console.print()

    found = [i for i in infos if i.found]
    if not found:
        console.print(
            f"  [{COLORS['warning']}]No chat history found for any supported tool. "
            f"Use one of the tools above, then run Vibe Check again.[/]"
        )
        wait_for_key()   # nothing saved: onboarding runs again next time
        return

    raw = get_input("Which one do you use most? (number)", "1" if len(infos) else "")
    try:
        n = int(raw)
        primary = infos[n - 1].key if 1 <= n <= len(infos) else found[0].key
    except ValueError:
        primary = found[0].key
    vibe_config.set_setting("primary_tool", primary)
    console.print()

    allowed: list[str] = []
    for info in found:
        ans = get_input(
            f"Allow reading {info.label} history ({info.session_count} sessions)? [Y/n]", "y"
        )
        if _yes(ans):
            allowed.append(info.key)
    sources.set_enabled_keys(allowed)
    console.print()
    if allowed:
        labels = ", ".join(i.label for i in infos if i.key in allowed)
        console.print(f"  [{COLORS['success']}]Saved. Reading: {labels}.[/]")
        console.print(f"  [{COLORS['muted']}]Change this any time in Settings -> Sources.[/]")
    else:
        console.print(f"  [{COLORS['warning']}]No source allowed. Nothing will be read.[/]")
    wait_for_key()


# ---------------------------------------------------------------------------
# Progress printer for the pipeline
# ---------------------------------------------------------------------------


def _make_progress_printer():
    """
    Return a callback suitable for `orchestrator.run_full_pipeline`.

    In parallel mode, events from different projects interleave freely.
    Each line is tagged with the project short name so it's obvious
    which line belongs to which in-flight worker.

    We also emit a rolling throughput line every ~30s so long per-project
    Stage 1 runs (e.g. a project with 90+ sessions) don't look stuck.
    """
    lock = threading.Lock()
    state = {
        "started": time.monotonic(),
        "sessions_done": 0,
        "last_tick": time.monotonic(),
    }

    def _print(*args, **kwargs):
        with lock:
            console.print(*args, **kwargs)

    def _maybe_tick_throughput():
        now = time.monotonic()
        if now - state["last_tick"] < 30.0:
            return
        elapsed = now - state["started"]
        rate = state["sessions_done"] / max(elapsed, 1.0) * 60.0
        state["last_tick"] = now
        _print(
            f"    [{COLORS['muted']}]... {state['sessions_done']} sessions "
            f"done in {elapsed/60:.1f} min ({rate:.1f}/min)[/]"
        )

    def _progress(event: str, data: dict):
        if event == "portfolio":
            mw = data.get("max_workers", 1)
            parallel = f" (concurrency: {mw})" if mw > 1 else ""
            _print(
                f"  [{COLORS['subtitle']}]Portfolio: "
                f"{data['count']} projects queued{parallel}[/]"
            )
        elif event == "project_start":
            _print(
                f"  [{COLORS['accent']}][{data['i']}/{data['total']}][/] "
                f"[{COLORS['title']}]{data['project']}[/] "
                f"[{COLORS['muted']}]({data['sessions']} sessions) "
                f"starting...[/]"
            )
        elif event == "stage1":
            # Per-session tick from inside Stage 1. Count toward
            # throughput, print a rolling line at most every 30s.
            state["sessions_done"] += 1
            _maybe_tick_throughput()
        elif event == "stage1_done":
            _print(
                f"    [{COLORS['muted']}][{data['project']}] stage 1: "
                f"{data['ok']} ok, {data['cached']} cached, "
                f"{data['skipped']} skipped, {data['failed']} failed[/]"
            )
        elif event == "stage2_done":
            tag = "cached" if data["cached"] else "fresh"
            mark = COLORS["success"] if data["synthesis"] else COLORS["error"]
            _print(
                f"    [{mark}][{data['project']}] stage 2: "
                f"{'ok' if data['synthesis'] else 'failed'} "
                f"[{COLORS['muted']}]({tag})[/][/]"
            )
        elif event == "cancelled":
            _print(
                f"  [{COLORS['warning']}]Cancellation requested. "
                f"Waiting for in-flight workers to finish...[/]"
            )
        elif event == "stage3":
            _print()
            _print(
                f"  [{COLORS['accent']}]Stage 3: timeline aggregation...[/]"
            )
        elif event == "stage3_done":
            tag = "cached" if data["cached"] else "fresh"
            _print(
                f"    [{COLORS['success']}]timeline done[/] "
                f"[{COLORS['muted']}]({tag})[/]"
            )
        elif event == "stage4":
            _print(
                f"  [{COLORS['accent']}]Stage 4: cross-project matrix...[/]"
            )
        elif event == "stage4_done":
            tag = "cached" if data["cached"] else "fresh"
            _print(
                f"    [{COLORS['success']}]cross-project done[/] "
                f"[{COLORS['muted']}]({tag})[/]"
            )
        elif event == "batch_status":
            _print(f"  [{COLORS['muted']}]{data.get('message', '')}[/]")
        elif event == "stage5":
            _print(f"  [{COLORS['accent']}]Stage 5: your profile and card...[/]")
        elif event == "stage5_done":
            tag = "cached" if data["cached"] else "fresh"
            _print(f"    [{COLORS['success']}]profile done[/] [{COLORS['muted']}]({tag})[/]")

    return _progress


@contextmanager
def _catch_sigint(token):
    """
    Install a SIGINT handler that trips the cancellation token on first
    Ctrl-C and restores Python's default handler on the second, so a
    second Ctrl-C raises KeyboardInterrupt like normal.

    Restores the previous handler on exit regardless of outcome.
    """
    previous = signal.getsignal(signal.SIGINT)
    state = {"count": 0}

    def _handler(signum, frame):
        state["count"] += 1
        if state["count"] == 1:
            token.cancel()
            # Don't print from a signal handler if we can help it --
            # the progress printer will surface the "cancelled" event.
            # But we do want the user to get feedback immediately if no
            # event is coming for a while, so a minimal stderr line is ok.
            print(
                "\n  [Ctrl-C caught] Finishing in-flight work, then stopping. "
                "Ctrl-C again to abort now.",
                file=sys.stderr, flush=True,
            )
            return
        signal.signal(signal.SIGINT, previous)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


# ---------------------------------------------------------------------------
# Option 6: full pipeline
# ---------------------------------------------------------------------------


# Per 1M tokens (input, output), standard API prices. Batch halves both.
PRICES = {
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-haiku-5-5": (0.10, 0.50),
}


def model_display_for(provider) -> str:
    return vibe_config.get_model_for(provider.info.env_var) or provider.info.default_model


def _open_cmd(path) -> str:
    """The command that opens a file on this OS."""
    if sys.platform == "darwin":
        return f'open "{path}"'
    if sys.platform.startswith("win"):
        return f'start "" "{path}"'
    return f'xdg-open "{path}"'


def _print_outputs(outputs: dict) -> None:
    labels = [
        ("dashboard", "Report (private)"),
        ("global:profile", "Profile"),
        ("global:rules", "Rules to paste"),
        ("share:card", "Share card (redacted)"),
        ("share:page", "Share page (redacted)"),
    ]
    console.print()
    console.print(f"  [{COLORS['success']}]Written:[/]")
    for key, label in labels:
        if outputs.get(key):
            console.print(f"    [{COLORS['accent']}]{label}:[/] {outputs[key]}")
    if outputs.get("share_blocked"):
        console.print(f"    [{COLORS['warning']}]Share export blocked: {outputs['share_blocked']}[/]")
    per_project = [k for k in outputs if k.startswith("project:")]
    console.print(
        f"    [{COLORS['muted']}]{len(per_project)} PROJECT.md files under {get_extractions_dir()}[/]"
    )
    if outputs.get("dashboard"):
        console.print()
        console.print(f"  [{COLORS['muted']}]Open the report:[/] {_open_cmd(outputs['dashboard'])}")


def run_full_pipeline():
    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]FULL RUN[/]")
    print_separator()
    console.print()

    if not check_installation():
        wait_for_key()
        return
    console.print()

    provider = select_provider()
    if not provider:
        console.print(
            f"  [{COLORS['warning']}]Need an LLM provider for the pipeline.[/]"
        )
        wait_for_key()
        return

    stats = rolling_status("Measuring project richness...", project_stats)
    keep = qualifying_projects(stats)

    if not keep:
        console.print(
            f"  [{COLORS['warning']}]No projects meet the richness threshold.[/]"
        )
        console.print(
            f"  [{COLORS['muted']}]Need >= {DEFAULT_MIN_SESSION_MESSAGES} msgs "
            f"in one session OR >= {DEFAULT_MIN_TOTAL_MESSAGES} total.[/]"
        )
        wait_for_key()
        return

    # Sort richest-first so the first provider call produces the "best" data
    # if the user Ctrl-C's.
    keep.sort(key=lambda s: -s.total_messages)

    total = len(keep)
    console.print(
        f"  [{COLORS['success']}]{total}[/] projects qualify "
        f"[{COLORS['muted']}](>= {DEFAULT_MIN_SESSION_MESSAGES} msgs in a session "
        f"OR >= {DEFAULT_MIN_TOTAL_MESSAGES} total)[/]"
    )
    console.print()
    console.print(f"  [{COLORS['subtitle']}]Queue (richest first):[/]")
    for i, s in enumerate(keep[:20], start=1):
        console.print(
            f"    [{COLORS['muted']}]{i:>2}.[/] {s.short_name} "
            f"[{COLORS['muted']}]"
            f"({s.sessions} sessions, {s.total_messages:,} msgs, "
            f"longest {s.max_session_messages:,})[/]"
        )
    if total > 20:
        console.print(f"    [{COLORS['muted']}]... and {total - 20} more[/]")
    console.print()

    max_workers = int(vibe_config.get_setting("max_workers", 4))
    max_workers = max(1, min(max_workers, 16))

    # Scope: everything, or a cheap first look at recent work.
    scope = get_input("Scope: [A]ll sessions or [Q]uick (10 most recent per project)?", "A")
    quick_cap = 10 if scope.strip().lower().startswith("q") else None
    total_sessions = sum(min(s.sessions, quick_cap or s.sessions) for s in keep)

    use_batch = False
    if getattr(provider, "supports_batch", False):
        ans = get_input("Use batch mode? 50% cheaper, results usually within an hour [Y/n]", "y")
        use_batch = _yes(ans)

    # Measured on real history: a session renders to ~10k input tokens on
    # average and the review is ~2k output tokens (thinking included).
    price = PRICES.get(model_display_for(provider))
    if price:
        per_session = 10_000 * price[0] / 1e6 + 2_000 * price[1] / 1e6
        if use_batch:
            per_session *= 0.5
        lo, hi = total_sessions * per_session * 0.6, total_sessions * per_session * 1.4
        cost_note = f"Roughly ${lo:.0f}-${hi:.0f}"
    else:
        # No price table for this model: say so instead of guessing.
        cost_note = f"About {total_sessions * 12_000:,} tokens (check your provider's prices)"
    if use_batch:
        time_note = "Stage 1 runs as one batch (usually under an hour, max 24h)"
    else:
        minutes = max(1, int(total_sessions * 20 / max_workers / 60))
        time_note = f"about {minutes}-{minutes * 2} min at concurrency {max_workers}"

    model_override = vibe_config.get_model_for(provider.info.env_var)
    model_display = model_override or provider.info.default_model
    console.print(
        f"  [{COLORS['muted']}]{total_sessions} sessions across {total} projects. "
        f"{time_note}. {cost_note} on "
        f"{provider.info.display_name} [{model_display}] for new sessions; "
        f"already-reviewed sessions are free.\n"
        f"  Ctrl-C once = graceful stop (safe, resumable).[/]"
    )
    console.print()

    answer = get_input(f"Proceed with {total} projects? [y/N]")
    if answer.strip().lower() not in ("y", "yes"):
        console.print(f"  [{COLORS['muted']}]Cancelled.[/]")
        wait_for_key()
        return

    print_separator()
    portfolio = [s.short_name for s in keep]

    cancel_token = orch.CancellationToken()
    started = time.time()
    with _catch_sigint(cancel_token):
        result = orch.run_full_pipeline(
            provider,
            portfolio=portfolio,
            progress=_make_progress_printer(),
            max_workers=max_workers,
            model=model_override,
            cancel_token=cancel_token,
            use_batch=use_batch,
            max_sessions_per_project=quick_cap,
        )
    elapsed = time.time() - started

    print_separator()
    console.print()
    ok = sum(1 for p in result.per_project if p.synthesis is not None)
    failed = sum(1 for p in result.per_project if p.synthesis is None)
    console.print(
        f"  [{COLORS['title']}]PIPELINE COMPLETE[/] "
        f"[{COLORS['muted']}]({elapsed/60:.1f} min)[/]"
    )
    console.print(
        f"  [{COLORS['success']}]{ok} analyzed[/], "
        f"[{COLORS['error']}]{failed} failed[/]"
    )
    if failed:
        console.print(f"  [{COLORS['muted']}]Failures:[/]")
        for p in result.per_project:
            if p.synthesis is None:
                console.print(f"    - {p.project_short}: {p.error}")
    for err in result.errors:
        console.print(f"  [{COLORS['error']}]Problem: {err}[/]")
    if cancel_token.is_cancelled:
        console.print(f"  [{COLORS['warning']}]Stopped early. Run again to continue where it stopped.[/]")

    console.print()
    console.print(f"  [{COLORS['accent']}]Rendering reports + dashboard...[/]")
    try:
        outputs = orch.write_all_reports(
            result.all_syntheses, result.all_session_facts
        )
    except Exception as e:  # noqa: BLE001
        console.print(f"  [{COLORS['error']}]Render failed: {e}[/]")
        wait_for_key()
        return

    _print_outputs(outputs)
    wait_for_key()


# ---------------------------------------------------------------------------
# Option 1: single project
# ---------------------------------------------------------------------------


def run_one_project_interactive():
    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]ANALYZE ONE PROJECT[/]")
    print_separator()
    console.print()

    if not check_installation():
        wait_for_key()
        return
    console.print()

    provider = select_provider()
    if not provider:
        wait_for_key()
        return

    # Build a quick list of candidates: every qualifying project, sorted by size.
    stats = rolling_status("Measuring project richness...", project_stats)
    keep = qualifying_projects(stats)
    keep.sort(key=lambda s: -s.total_messages)
    if not keep:
        console.print(f"  [{COLORS['warning']}]Nothing meets the richness bar.[/]")
        wait_for_key()
        return

    items = [
        f"{s.short_name} ({s.sessions} sessions, {s.total_messages:,} msgs)"
        for s in keep[:30]
    ]
    idx = select_from_list(items, "Pick a project")
    if idx is None:
        return
    target = keep[idx]

    console.print()
    buckets = rolling_status("Loading sessions...", orch._bucket_sessions_by_short)
    sessions = buckets.get(target.short_name, [])
    if not sessions:
        console.print(f"  [{COLORS['warning']}]No sessions found after load.[/]")
        wait_for_key()
        return

    console.print()
    console.print(
        f"  [{COLORS['accent']}]Running Stage 1 + Stage 2 on "
        f"{len(sessions)} sessions...[/]"
    )
    model_override = vibe_config.get_model_for(provider.info.env_var)
    cancel_token = orch.CancellationToken()
    with _catch_sigint(cancel_token):
        result = orch.run_one_project(
            target.short_name, sessions, provider,
            model=model_override,
            progress=_make_progress_printer(),
            cancel_token=cancel_token,
        )
    if result.synthesis is None:
        console.print(f"  [{COLORS['error']}]Failed: {result.error}[/]")
        wait_for_key()
        return

    # Rebuild global reports + dashboard using all available syntheses so
    # this project's update shows up in the existing dashboard.
    console.print()
    console.print(f"  [{COLORS['accent']}]Refreshing global reports...[/]")
    all_syn = orch.load_all_syntheses()
    all_facts = orch.load_all_session_facts()
    try:
        # Refresh the profile numbers so the card includes this project.
        # One small LLM call writes the profile text (cached when unchanged).
        from .pipeline.profile import build_profile
        build_profile(all_facts, all_syn, provider)
        outputs = orch.write_all_reports(all_syn, all_facts)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [{COLORS['warning']}]Global refresh failed: {e}[/]")
        outputs = {}

    project_md = (
        get_extractions_dir() / target.short_name / "PROJECT.md"
    )
    console.print()
    console.print(f"  [{COLORS['success']}]Done.[/]")
    if project_md.exists():
        console.print(f"    [{COLORS['accent']}]Report:[/] {project_md}")
    dashboard = outputs.get("dashboard")
    if dashboard:
        console.print(f"    [{COLORS['accent']}]Dashboard:[/] {dashboard}")
    wait_for_key()


# ---------------------------------------------------------------------------
# Option 5: rebuild dashboard from existing artifacts (no LLM)
# ---------------------------------------------------------------------------


def run_build_dashboard():
    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]REBUILD REPORT AND CARD[/]")
    print_separator()
    console.print()
    console.print(
        f"  [{COLORS['muted']}]No LLM calls. Just re-renders the HTML from "
        f"whatever's already on disk.[/]"
    )
    console.print()

    syntheses = orch.load_all_syntheses()
    facts = orch.load_all_session_facts()
    if not syntheses:
        console.print(
            f"  [{COLORS['warning']}]No per-project syntheses on disk. "
            f"Run option 6 first.[/]"
        )
        wait_for_key()
        return

    console.print(f"  [{COLORS['accent']}]Loading existing artifacts...[/]")
    console.print(
        f"  [{COLORS['muted']}]{len(syntheses)} project syntheses, "
        f"{len(facts)} session facts[/]"
    )

    try:
        outputs = orch.write_all_reports(syntheses, facts)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [{COLORS['error']}]Render failed: {e}[/]")
        wait_for_key()
        return

    _print_outputs(outputs)
    wait_for_key()


# ---------------------------------------------------------------------------
# Option 3: Atlas (unchanged legacy module)
# ---------------------------------------------------------------------------


def run_build_atlas():
    from . import atlas as atlas_mod

    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]BUILD PROJECT ATLAS[/]")
    print_separator()
    console.print()

    provider = select_provider()
    if not provider:
        console.print(
            f"  [{COLORS['warning']}]Need an LLM provider to classify.[/]"
        )
        wait_for_key()
        return

    from .extract import list_projects as _list
    counts = _list()
    total = len(counts)
    at1 = sum(1 for c in counts.values() if c >= 1)
    at3 = sum(1 for c in counts.values() if c >= 3)
    at5 = sum(1 for c in counts.values() if c >= 5)
    console.print(
        f"  [{COLORS['muted']}]Your enabled sources have history for {total} projects "
        f"(>=1: {at1}, >=3: {at3}, >=5: {at5}).[/]"
    )
    raw = get_input("Minimum sessions per project [1]")
    try:
        min_sessions = max(1, int(raw.strip())) if raw.strip() else 1
    except ValueError:
        min_sessions = 1

    console.print(
        f"  [{COLORS['accent']}]Collecting projects "
        f"(min {min_sessions})...[/]"
    )
    candidates = atlas_mod.collect_candidates(min_sessions=min_sessions)
    if not candidates:
        console.print(f"  [{COLORS['warning']}]Nothing qualifies.[/]")
        wait_for_key()
        return
    console.print(f"  Found [{COLORS['success']}]{len(candidates)}[/] projects.")

    console.print(
        f"  [{COLORS['accent']}]Classifying with "
        f"{provider.info.display_name}...[/]"
    )
    try:
        atlas_mod.classify(candidates, provider)
    except Exception as e:  # noqa: BLE001
        console.print(f"  [{COLORS['error']}]Failed: {e}[/]")
        wait_for_key()
        return

    md_path, json_path = atlas_mod.write_atlas(candidates)
    console.print()
    console.print(f"  [{COLORS['success']}]Atlas written:[/]")
    console.print(f"    {md_path}")
    console.print(f"    {json_path}")
    console.print(f"  [{COLORS['muted']}]Open with:[/] {_open_cmd(md_path)}")
    wait_for_key()


# ---------------------------------------------------------------------------
# Option 2: browse existing project reports
# ---------------------------------------------------------------------------


def load_existing_session():
    root = get_extractions_dir()
    if not root.exists():
        console.print(f"  [{COLORS['warning']}]No extractions yet.[/]")
        wait_for_key()
        return

    reports: list[tuple[str, Path]] = []
    for project_dir in sorted(root.iterdir()):
        md = project_dir / "PROJECT.md"
        if md.exists():
            reports.append((project_dir.name, md))

    if not reports:
        console.print(
            f"  [{COLORS['warning']}]No PROJECT.md files found. "
            f"Run option 6 first.[/]"
        )
        wait_for_key()
        return

    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]EXISTING REPORTS[/]")
    print_separator()

    items = [f"{name}" for name, _ in reports]
    idx = select_from_list(items, "Select project")
    if idx is None:
        return
    _, md_path = reports[idx]
    console.print()
    console.print(f"  [{COLORS['muted']}]Open with:[/] {_open_cmd(md_path)}")
    console.print()
    console.print(md_path.read_text(encoding="utf-8"))
    wait_for_key()


# ---------------------------------------------------------------------------
# Settings screen
# ---------------------------------------------------------------------------


def _key_source(info) -> tuple[str, str]:
    """
    Return (status_label, color) describing where this provider's key lives.
    Env var wins over saved config, matching the precedence in config.get_api_key.
    """
    if os.environ.get(info.env_var):
        return ("Set in environment", COLORS["success"])
    if vibe_config.load_api_keys().get(info.env_var):
        return ("Saved in config", COLORS["success"])
    return ("Not set", COLORS["muted"])


def run_settings() -> None:
    """
    Manage API keys, models, and concurrency without grepping shell dotfiles.
    """
    while True:
        clear_screen()
        print_header()
        console.print(f"  [{COLORS['title']}]SETTINGS[/]")
        print_separator()
        console.print()
        console.print(
            f"  [{COLORS['muted']}]Stored at {vibe_config._config_path()} "
            f"(0600). Env vars override saved keys.[/]"
        )
        console.print()

        console.print(f"  [{COLORS['subtitle']}]API KEYS / MODELS[/]")
        providers = list_providers()
        for i, info in enumerate(providers, start=1):
            status, color = _key_source(info)
            saved = vibe_config.load_api_keys().get(info.env_var)
            preview = f" [{COLORS['muted']}]{vibe_config.mask_key(saved)}[/]" if saved else ""
            model_override = vibe_config.get_model_for(info.env_var)
            model_display = model_override or info.default_model
            marker = "*" if model_override else " "
            console.print(
                f"    [{COLORS['highlight']}]{i}.[/] {info.display_name} "
                f"[{COLORS['muted']}]({info.env_var})[/] -- "
                f"[{color}]{status}[/]{preview}"
            )
            console.print(
                f"        [{COLORS['muted']}]model{marker}: {model_display}[/]"
            )
        console.print()

        max_workers = int(vibe_config.get_setting("max_workers", 4))
        console.print(f"  [{COLORS['subtitle']}]PIPELINE[/]")
        console.print(
            f"    [{COLORS['highlight']}]C.[/] Concurrency "
            f"[{COLORS['accent']}]({max_workers} parallel workers)[/]"
        )
        console.print()

        enabled = sources.enabled_keys()
        labels = ", ".join(sources.tool_label(k) for k in enabled) or "none"
        console.print(
            f"    [{COLORS['highlight']}]S.[/] Sources "
            f"[{COLORS['accent']}]({labels})[/]"
        )
        console.print()

        console.print(f"    [{COLORS['muted']}]0. Back[/]")
        console.print()

        raw = get_input("Select").strip().lower()
        if raw in ("", "0"):
            return
        if raw == "s":
            run_onboarding()
            continue
        if raw == "c":
            _edit_concurrency()
            continue
        try:
            choice = int(raw)
        except ValueError:
            continue
        if not (1 <= choice <= len(providers)):
            continue

        info = providers[choice - 1]
        _manage_single_key(info)


def _edit_concurrency() -> None:
    current = int(vibe_config.get_setting("max_workers", 4))
    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]PIPELINE CONCURRENCY[/]")
    print_separator()
    console.print()
    console.print(
        f"  [{COLORS['muted']}]How many projects to analyze in parallel "
        f"during the full pipeline. LLM stages are serial WITHIN a project "
        f"(Stage 1 -> 2); this only parallelizes across projects.\n\n"
        f"  Higher = faster, but more concurrent spend and more load against "
        f"your provider's rate limits. 4 is a safe default for Claude tier 4.\n\n"
        f"  Global Stages 3 and 4 always run single-threaded.[/]"
    )
    console.print()
    console.print(f"  Current: [{COLORS['accent']}]{current}[/]")
    console.print()
    raw = get_input("New value (1-16, Enter to keep)").strip()
    if not raw:
        return
    try:
        val = int(raw)
    except ValueError:
        console.print(f"  [{COLORS['error']}]Not a number.[/]")
        wait_for_key()
        return
    if not (1 <= val <= 16):
        console.print(f"  [{COLORS['error']}]Out of range (1-16).[/]")
        wait_for_key()
        return
    vibe_config.set_setting("max_workers", val)
    console.print(f"  [{COLORS['success']}]Saved.[/] Concurrency = {val}.")
    wait_for_key()


def _manage_single_key(info) -> None:
    """Add / replace / delete / set-model flow for one provider."""
    while True:
        clear_screen()
        print_header()
        console.print(f"  [{COLORS['title']}]{info.display_name}[/]")
        print_separator()
        console.print()

        env_set = bool(os.environ.get(info.env_var))
        saved = vibe_config.load_api_keys().get(info.env_var)
        model_override = vibe_config.get_model_for(info.env_var)

        if env_set:
            console.print(
                f"  [{COLORS['warning']}]Note:[/] [{info.env_var}] is set in your "
                f"shell environment; that value takes precedence."
            )
            console.print()

        if saved:
            console.print(
                f"  Saved key: "
                f"[{COLORS['accent']}]{vibe_config.mask_key(saved)}[/]"
            )
        else:
            console.print(
                f"  [{COLORS['muted']}]No saved key. Get one at "
                f"{info.signup_url}[/]"
            )
        model_display = model_override or info.default_model
        model_tag = "custom" if model_override else "default"
        console.print(
            f"  Model:     [{COLORS['accent']}]{model_display}[/] "
            f"[{COLORS['muted']}]({model_tag})[/]"
        )
        console.print()

        console.print(f"    [{COLORS['highlight']}]1.[/] "
                      f"{'Replace' if saved else 'Add'} API key")
        if saved:
            console.print(f"    [{COLORS['highlight']}]2.[/] Delete API key")
        console.print(f"    [{COLORS['highlight']}]3.[/] Change model")
        if model_override:
            console.print(f"    [{COLORS['highlight']}]4.[/] "
                          f"Reset to default ({info.default_model})")
        console.print(f"    [{COLORS['muted']}]0. Back[/]")
        console.print()
        try:
            choice = int(get_input("Select"))
        except ValueError:
            return
        if choice == 0:
            return
        if choice == 1:
            _prompt_and_save(info)
        elif choice == 2 and saved:
            if vibe_config.delete_api_key(info.env_var):
                console.print(
                    f"  [{COLORS['success']}]Deleted.[/] "
                    f"(Env var still active if set.)"
                )
            wait_for_key()
        elif choice == 3:
            _prompt_and_set_model(info)
        elif choice == 4 and model_override:
            vibe_config.set_model_for(info.env_var, None)
            console.print(
                f"  [{COLORS['success']}]Reset to default:[/] "
                f"{info.default_model}"
            )
            wait_for_key()


def _prompt_and_set_model(info) -> None:
    console.print()
    console.print(
        f"  [{COLORS['muted']}]Examples for {info.display_name}:[/]"
    )
    if info.name == "claude":
        console.print(
            f"  [{COLORS['muted']}]    claude-sonnet-5-5  (default, recommended)\n"
            f"      claude-opus-5-5    (slower, pricier)\n"
            f"      claude-haiku-5-5   (much cheaper, less careful)[/]"
        )
    elif info.name == "gemini":
        console.print(
            f"  [{COLORS['muted']}]    gemini-pro-latest    (default, follows Google's current Pro)\n"
            f"      gemini-flash-latest  (cheaper, faster)[/]"
        )
    console.print()
    val = get_input("New model (Enter to cancel)").strip()
    if not val:
        return
    vibe_config.set_model_for(info.env_var, val)
    console.print(
        f"  [{COLORS['success']}]Saved.[/] "
        f"{info.display_name} model = {val}"
    )
    wait_for_key()


def _prompt_and_save(info) -> None:
    key = console.input(f"  [{COLORS['accent']}]Paste your {info.display_name} key (or Enter to cancel)[/]: ", password=True).strip()
    if not key:
        return
    try:
        vibe_config.save_api_key(info.env_var, key)
        console.print(
            f"  [{COLORS['success']}]Saved[/] "
            f"({vibe_config.mask_key(key)}) to "
            f"[dim]{vibe_config._config_path()}[/]."
        )
    except OSError as e:
        console.print(f"  [{COLORS['error']}]Could not save: {e}[/]")
    wait_for_key()


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------


def main_menu() -> str | None:
    clear_screen()
    print_header()
    console.print(f"  [{COLORS['title']}]MAIN MENU[/]")
    print_separator()
    console.print()
    console.print(f"    [{COLORS['highlight']}]1.[/] [{COLORS['accent']}]Analyze one project[/]")
    console.print(f"       [{COLORS['muted']}]Stage 1 + 2 on a single project (refreshes globals)[/]")
    console.print()
    console.print(f"    [{COLORS['highlight']}]2.[/] [{COLORS['accent']}]Browse existing reports[/]")
    console.print(f"       [{COLORS['muted']}]View PROJECT.md files from previous runs[/]")
    console.print()
    console.print(f"    [{COLORS['highlight']}]3.[/] [{COLORS['accent']}]Build Project Atlas[/]")
    console.print(f"       [{COLORS['muted']}]One-page searchable index of all projects[/]")
    console.print()
    console.print(f"    [{COLORS['highlight']}]4.[/] [{COLORS['accent']}]Rebuild report and card[/]")
    console.print(f"       [{COLORS['muted']}]Re-render the report, card and share export (no LLM)[/]")
    console.print()
    console.print(f"    [{COLORS['highlight']}]5.[/] [{COLORS['accent']}]Full pipeline[/]")
    console.print(f"       [{COLORS['muted']}]Review every session, then build your profile, card and rules[/]")
    console.print()
    console.print(f"    [{COLORS['highlight']}]6.[/] [{COLORS['accent']}]Settings / API keys[/]")
    console.print(f"       [{COLORS['muted']}]API keys, models, and which chat sources to read[/]")
    console.print()
    console.print(f"    [{COLORS['muted']}]0. Exit[/]")
    console.print()
    try:
        choice = int(get_input("Select"))
    except ValueError:
        return None
    return {0: "exit", 1: "one", 2: "load", 3: "atlas",
            4: "dashboard", 5: "full", 6: "settings"}.get(choice)


def main():
    try:
        _main()
    except (KeyboardInterrupt, EOFError):
        console.print()
        console.print(f"  [{COLORS['muted']}]Stopped. Everything finished so far is saved; run again to continue.[/]")


def _main():
    if not sources.onboarding_done():
        run_onboarding()
    while True:
        choice = main_menu()
        if choice == "exit":
            clear_screen()
            print_header()
            console.print(
                f"  [{COLORS['success']}]Thanks for using Vibe Check.[/]"
            )
            console.print()
            return
        if choice is None:
            continue
        if choice == "one":
            run_one_project_interactive()
        elif choice == "load":
            load_existing_session()
        elif choice == "atlas":
            run_build_atlas()
        elif choice == "dashboard":
            run_build_dashboard()
        elif choice == "full":
            run_full_pipeline()
        elif choice == "settings":
            run_settings()


if __name__ == "__main__":
    main()
