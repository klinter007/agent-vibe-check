"""Smoke test: parse real transcripts on disk with the new parser and print stats."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_vibe_check.extract import short_project_name
from agent_vibe_check.parser import (
    iter_all_projects,
    parse_project,
    parse_transcript,
)


def main() -> int:
    projects_dir = Path.home() / ".cursor" / "projects"
    if not projects_dir.is_dir():
        print(f"ERROR: {projects_dir} does not exist")
        return 1

    print(f"Scanning: {projects_dir}")
    print()

    projects = list(iter_all_projects(projects_dir))
    print(f"Projects with transcripts: {len(projects)}")
    print()

    totals = {
        "transcripts_attempted": 0,
        "transcripts_parsed": 0,
        "transcripts_failed": 0,
        "sessions_total": 0,
        "subagent_sessions": 0,
        "messages_total": 0,
        "tool_calls_total": 0,
        "user_queries": 0,
    }
    tool_name_counter: Counter[str] = Counter()
    top_projects: list[tuple[str, int, int]] = []

    for project_dir in projects:
        sessions = parse_project(project_dir)
        msg_count = sum(s.message_count for s in sessions)
        tc_count = sum(len(s.tool_calls) for s in sessions)
        top_projects.append((project_dir.name, len(sessions), msg_count))

        totals["sessions_total"] += len(sessions)
        totals["subagent_sessions"] += sum(1 for s in sessions if s.is_subagent)
        totals["messages_total"] += msg_count
        totals["tool_calls_total"] += tc_count
        for s in sessions:
            totals["user_queries"] += sum(
                1 for m in s.user_messages if m.clean_user_text
            )
            for tc in s.tool_calls:
                tool_name_counter[tc.name] += 1

    print("=== Totals ===")
    for k, v in totals.items():
        print(f"  {k:25s} {v:>6}")
    print()

    print("=== Top 10 projects by session count ===")
    top_projects.sort(key=lambda x: x[1], reverse=True)
    for name, n_sessions, n_msgs in top_projects[:10]:
        short = short_project_name(name)[:60]
        print(f"  {n_sessions:>4} sessions · {n_msgs:>5} msgs · {short}")
    print()

    print("=== Top 10 tool_use calls ===")
    for name, count in tool_name_counter.most_common(10):
        print(f"  {count:>5}  {name}")
    print()

    # Sanity: grab one session, show the first user query and first assistant reasoning block
    print("=== Sanity sample ===")
    for project_dir in projects[:5]:
        for path in (project_dir / "agent-transcripts").rglob("*.jsonl"):
            if path.parent.name == "subagents":
                continue
            s = parse_transcript(path)
            if s and s.user_messages and s.assistant_messages:
                print(f"  project: {project_dir.name[:60]}")
                print(f"  session: {s.session_id[:8]}...")
                print(f"  first user: {s.first_user_query[:120]!r}")
                ra = s.assistant_messages[0]
                if ra.text_blocks:
                    print(f"  first assistant text: {ra.text_blocks[0][:120]!r}")
                if ra.tool_calls:
                    print(f"  first tool call: {ra.tool_calls[0].name}")
                return 0
    print("  (no suitable sample session found)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
