"""
Four-stage analysis pipeline.

  Stage 1  session_extractor   raw messages -> session facts  (LLM, cached per session)
  Stage 2  project_synthesizer session facts -> project doc    (LLM, cached per project)
  Stage 3  timeline            session facts -> timeline doc   (local + 1 LLM narrative)
  Stage 4  cross_project       project docs  -> comparison     (local + 1 LLM anomaly pass)

Each stage writes its output atomically and can be resumed independently.
Upstream data ingestion (parser.py, composer_db.py, extract.py) is unchanged.
"""

from __future__ import annotations
