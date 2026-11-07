"""
Schema drift detector.

Proactively samples a handful of transcripts and reports what fraction of
lines our parser accepts. Cursor's on-disk format has shifted at least once
already (JSON array -> JSONL, removal of top-level `thinking`), so this guard
gives early warning instead of producing silently-degraded analyses.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from .parser import iter_project_transcripts, parse_line


# Minimum acceptance rate before we consider the parser healthy.
# Below this: warn the user, recommend opening an issue.
MIN_PARSE_RATE = 0.7


@dataclass
class SchemaCheckResult:
    sampled_files: int
    total_lines: int
    parsed_lines: int
    rejected_lines: int
    parse_rate: float
    ok: bool
    reason: str = ""

    @property
    def summary(self) -> str:
        pct = self.parse_rate * 100
        return (
            f"parsed {self.parsed_lines}/{self.total_lines} lines "
            f"({pct:.1f}%) across {self.sampled_files} sample files"
        )


def check_project(project_dir: Path, sample_size: int = 5, seed: int | None = None) -> SchemaCheckResult:
    """Sample up to `sample_size` transcripts from a project and measure parse rate."""
    files = list(iter_project_transcripts(project_dir))
    if not files:
        return SchemaCheckResult(
            sampled_files=0,
            total_lines=0,
            parsed_lines=0,
            rejected_lines=0,
            parse_rate=0.0,
            ok=False,
            reason="No transcript files found",
        )

    rng = random.Random(seed)
    sample = rng.sample(files, min(sample_size, len(files)))

    total = 0
    parsed = 0
    for path in sample:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    total += 1
                    if parse_line(line) is not None:
                        parsed += 1
        except OSError:
            continue

    rate = parsed / total if total else 0.0
    ok = rate >= MIN_PARSE_RATE
    reason = (
        ""
        if ok
        else (
            f"Parse rate {rate:.1%} below threshold {MIN_PARSE_RATE:.0%}. "
            "Cursor's transcript format may have changed - parser needs an update."
        )
    )

    return SchemaCheckResult(
        sampled_files=len(sample),
        total_lines=total,
        parsed_lines=parsed,
        rejected_lines=total - parsed,
        parse_rate=rate,
        ok=ok,
        reason=reason,
    )


def check_global(cursor_projects_dir: Path, sample_size: int = 10) -> SchemaCheckResult:
    """Pick transcripts from across all projects to cross-check format stability."""
    all_files: list[Path] = []
    if not cursor_projects_dir.is_dir():
        return SchemaCheckResult(
            sampled_files=0,
            total_lines=0,
            parsed_lines=0,
            rejected_lines=0,
            parse_rate=0.0,
            ok=False,
            reason=f"Cursor projects dir not found: {cursor_projects_dir}",
        )

    for project_dir in cursor_projects_dir.iterdir():
        if project_dir.is_dir():
            all_files.extend(iter_project_transcripts(project_dir))

    if not all_files:
        return SchemaCheckResult(
            sampled_files=0,
            total_lines=0,
            parsed_lines=0,
            rejected_lines=0,
            parse_rate=0.0,
            ok=False,
            reason="No transcripts found anywhere under projects dir",
        )

    sample = random.sample(all_files, min(sample_size, len(all_files)))
    total = 0
    parsed = 0
    for path in sample:
        try:
            with path.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    total += 1
                    if parse_line(line) is not None:
                        parsed += 1
        except OSError:
            continue

    rate = parsed / total if total else 0.0
    ok = rate >= MIN_PARSE_RATE
    return SchemaCheckResult(
        sampled_files=len(sample),
        total_lines=total,
        parsed_lines=parsed,
        rejected_lines=total - parsed,
        parse_rate=rate,
        ok=ok,
        reason="" if ok else (
            f"Global parse rate {rate:.1%} below threshold. Format drift likely."
        ),
    )
