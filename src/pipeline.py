# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
End-to-end RAG pipeline: expand → retrieve → prompt → Groq LLM, with structured logging.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedder import Embedder
from src.llm_handler import generate_response
from src.logger import utc_timestamp, write_log
from src.prompt_builder import build_prompt
from src.retriever import _load_or_build_index, expand_query, retrieve


def _chunks_for_log(retrieved: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize retrieval rows for ``write_log`` (``score`` + ``source``)."""
    out: list[dict[str, Any]] = []
    for c in retrieved:
        meta = c.get("metadata") or {}
        src = meta.get("source", "unknown")
        out.append(
            {
                "rank": c.get("rank"),
                "text": c.get("text", ""),
                "score": c.get("similarity_score"),
                "source": src,
            }
        )
    return out


class RAGPipeline:
    """Loads FAISS index + chunk metadata once; runs retrieval and generation."""

    def __init__(self, index_dir: str | Path | None = None, embedder: Embedder | None = None) -> None:
        self._root = _PROJECT_ROOT
        self.index_dir = Path(index_dir) if index_dir else self._root / "data" / "faiss_index"
        self._embedder = embedder or Embedder()
        self._index, self._chunks_metadata = _load_or_build_index(self._embedder, self.index_dir)

    def run_pipeline(
        self,
        query: str,
        template: int = 1,
        k: int = 5,
        *,
        log_to_file: bool = True,
        verbose: bool = True,
    ) -> dict[str, Any]:
        """
        Run expand → retrieve → prompt → LLM; optionally log one JSON record.

        Returns a dict with ``query``, ``expanded_query``, ``retrieved_chunks`` (raw retrieval),
        ``prompt_template_used``, ``prompt_sent_to_llm``, ``llm_response``.

        Set ``verbose=False`` to suppress stage prints (e.g. adversarial test harnesses).
        """
        query = query.strip()

        if verbose and hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass

        # --- Stage 1 ---
        expanded = expand_query(query)
        if verbose:
            print("\n=== STAGE 1: Query Expansion ===\n")
            print(f"Original:  {query}")
            print(f"Expanded:  {expanded}\n")

        # --- Stage 2 ---
        retrieved = retrieve(
            query,
            self._index,
            self._chunks_metadata,
            self._embedder,
            k=k,
        )
        if verbose:
            print("=== STAGE 2: Retrieval ===\n")
            print(f"Retrieved {len(retrieved)} chunk(s) (top-{k}, after distance filter).\n")
            for row in retrieved:
                sc = row.get("similarity_score")
                meta = row.get("metadata") or {}
                print(
                    f"  rank={row.get('rank')}  score={sc:.4f}  source={meta.get('source')}  "
                    f"text[:80]={repr((row.get('text') or '')[:80])}..."
                )
            print()

        # --- Stage 3 ---
        prompt = build_prompt(query, retrieved, template=int(template))
        if verbose:
            print("=== STAGE 3: Prompt Construction ===\n")
            print(f"Template: {template}  |  Prompt length: {len(prompt)} chars\n")
            print(prompt[:1200] + ("...\n" if len(prompt) > 1200 else "\n"))

        # --- Stage 4 ---
        llm_response = generate_response(prompt)
        if verbose:
            print("=== STAGE 4: LLM Response ===\n")
            print(llm_response)
            print()

        result: dict[str, Any] = {
            "query": query,
            "expanded_query": expanded,
            "retrieved_chunks": retrieved,
            "prompt_template_used": int(template),
            "prompt_sent_to_llm": prompt,
            "llm_response": llm_response,
        }

        if log_to_file:
            entry = {
                "timestamp": utc_timestamp(),
                "query": query,
                "expanded_query": expanded,
                "retrieved_chunks": _chunks_for_log(retrieved),
                "prompt_template_used": int(template),
                "prompt_sent_to_llm": prompt,
                "llm_response": llm_response,
            }
            write_log(entry)

        return result


def load_pipeline(index_dir: str | Path | None = None, embedder: Embedder | None = None) -> RAGPipeline:
    """Convenience: construct pipeline (loads index from disk)."""
    return RAGPipeline(index_dir=index_dir, embedder=embedder)


if __name__ == "__main__":
    # Avoid UnicodeEncodeError on Windows consoles when prompts contain symbols (e.g. ≤).
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    # From project root:  python src/pipeline.py
    queries = [
        "Who won the most votes in Accra Central constituency?",
        "What is Ghana's projected GDP growth in the 2025 budget?",
        "What is the meaning of life?",
    ]
    pipe = load_pipeline()
    for i, q in enumerate(queries, start=1):
        print("\n" + "#" * 80)
        print(f"RUN {i}/{len(queries)}")
        print("#" * 80)
        # Use strict template (1) for all; query 3 tests hallucination control.
        pipe.run_pipeline(q, template=1, k=5)
