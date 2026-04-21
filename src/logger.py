# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Append-only JSON logging for RAG pipeline runs (``logs/experiment_logs.json``).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG_PATH = _PROJECT_ROOT / "logs" / "experiment_logs.json"


def write_log(entry: dict[str, Any], log_path: str | Path | None = None) -> None:
    """
    Append ``entry`` to a JSON array in ``experiment_logs.json`` (create file if missing).

    Expected keys (pipeline supplies all):
      - timestamp, query, expanded_query, retrieved_chunks, prompt_template_used,
        prompt_sent_to_llm, llm_response
    """
    path = Path(log_path) if log_path else DEFAULT_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    existing: list[Any] = []
    if path.is_file():
        raw = path.read_text(encoding="utf-8").strip()
        if raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    existing = data
            except json.JSONDecodeError:
                existing = []

    existing.append(entry)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def log_feedback(
    feedback: str,
    query: str,
    top_chunk: str,
    log_path: str | Path | None = None,
) -> None:
    """
    Append a UI feedback row (thumbs up/down) to ``experiment_logs.json``.

    ``feedback`` should be ``\"positive\"`` or ``\"negative\"``.
    """
    write_log(
        {
            "timestamp": utc_timestamp(),
            "feedback": feedback,
            "query": query,
            "top_chunk": top_chunk,
        },
        log_path=log_path,
    )
