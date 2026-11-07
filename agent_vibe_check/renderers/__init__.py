"""
Markdown renderers for pipeline outputs.

These consume the schemas in `agent_vibe_check.pipeline.schemas` and produce
human-readable markdown. Kept separate from the pipeline so we can regenerate
reports without re-running the LLM.
"""

from __future__ import annotations
