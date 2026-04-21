# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Prompt construction for RAG: strict vs flexible templates with numbered retrieval chunks.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

MAX_CONTEXT_CHARS = 3000

# Template 1 - Strict (hallucination control)
TEMPLATE_STRICT = """You are an AI assistant for Academic City University.
Answer the question using ONLY the context provided below.
If the answer is not in the context, say exactly:
"I don't have enough information to answer that."
Do not make up any information.

Context:
{context}

Question: {query}

Answer:
"""

# Template 2 - Flexible (reasoning allowed)
TEMPLATE_FLEXIBLE = """You are an AI assistant for Academic City University.
Use the context below as your primary source.
If you reason beyond the context, clearly start that
sentence with "Based on general knowledge:".

Context:
{context}

Question: {query}

Answer:
"""


def _format_numbered_chunks(chunks: list[dict[str, Any]]) -> str:
    """Join chunks as ``[Chunk n]`` blocks (1-based), best chunks first."""
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        body = (ch.get("text") or "").strip()
        parts.append(f"[Chunk {i}]\n{body}")
    return "\n\n".join(parts)


def _truncate_context(
    retrieved_chunks: list[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build numbered context; ensure total length <= ``max_chars``.

    ``similarity_score`` is squared L2 distance (lower = better match). When truncating,
    **drop the worst chunks first** (highest ``similarity_score``), then renumber.
    If a single chunk still exceeds ``max_chars``, its text is truncated with ``...``.
    """
    if not retrieved_chunks:
        return "", []

    # Work on a mutable list; prefer better matches (lower distance) first in the prompt.
    remaining = sorted(retrieved_chunks, key=lambda c: float(c.get("similarity_score", 0.0)))

    while remaining:
        context = _format_numbered_chunks(remaining)
        if len(context) <= max_chars:
            return context, remaining
        if len(remaining) == 1:
            ch = dict(remaining[0])
            prefix = "[Chunk 1]\n"
            max_body = max(0, max_chars - len(prefix))
            ch["text"] = _truncate_single_chunk_text((ch.get("text") or "").strip(), max_body)
            fixed = [ch]
            return _format_numbered_chunks(fixed), fixed
        worst = max(remaining, key=lambda c: float(c.get("similarity_score", 0.0)))
        remaining.remove(worst)

    return "", []


def _truncate_single_chunk_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def build_prompt(
    query: str,
    retrieved_chunks: list[dict[str, Any]],
    template: int = 1,
) -> str:
    """
    Format template **1** (strict) or **2** (flexible) with ``query`` and numbered ``context``.

    Context is built from ``retrieved_chunks``, truncated to **3000** characters by dropping
    worst-matching chunks first (highest ``similarity_score``).
    """
    query = query.strip()
    context, _used = _truncate_context(retrieved_chunks, MAX_CONTEXT_CHARS)
    tpl = TEMPLATE_STRICT if int(template) == 1 else TEMPLATE_FLEXIBLE
    return tpl.format(context=context, query=query)


def compare_prompt_templates(query: str, retrieved_chunks: list[dict[str, Any]]) -> None:
    """
    Build both templates for the same query and chunks, print them, and append to experiment log.

    Prints under clear section headers (terminal width rarely allows true two-column prompts).
    """
    strict_p = build_prompt(query, retrieved_chunks, template=1)
    flex_p = build_prompt(query, retrieved_chunks, template=2)

    width = 100
    print("\n" + "=" * width)
    print("TEMPLATE 1 - STRICT (left)  |  TEMPLATE 2 - FLEXIBLE (right)")
    print("=" * width)

    lines1 = strict_p.splitlines()
    lines2 = flex_p.splitlines()
    n = max(len(lines1), len(lines2))
    col_w = 48
    for i in range(n):
        a = lines1[i] if i < len(lines1) else ""
        b = lines2[i] if i < len(lines2) else ""
        if len(a) > col_w:
            a = a[: col_w - 1] + "…"
        if len(b) > col_w:
            b = b[: col_w - 1] + "…"
        print(f"{a:<{col_w}} | {b:<{col_w}}")

    print("=" * width)
    print("\n--- Full Template 1 (Strict) ---\n")
    print(strict_p)
    print("\n--- Full Template 2 (Flexible) ---\n")
    print(flex_p)
    print()

    _append_prompt_comparison_log(query, strict_p, flex_p)


def _append_prompt_comparison_log(query: str, strict_prompt: str, flexible_prompt: str) -> None:
    path = _PROJECT_ROOT / "logs" / "experiment_logs.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment": "prompt_template_comparison",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "template_1_strict": strict_prompt,
        "template_2_flexible": flexible_prompt,
    }

    existing: list[Any] = []
    if path.is_file():
        try:
            raw = path.read_text(encoding="utf-8").strip()
            if raw:
                existing = json.loads(raw)
        except json.JSONDecodeError:
            existing = []
        if not isinstance(existing, list):
            existing = []

    existing.append(record)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Appended prompt comparison to: {path.resolve()}\n")


def _load_index_and_retrieve(
    query: str,
    k: int = 8,
    *,
    embedder: Any = None,
) -> list[dict[str, Any]]:
    from src.embedder import Embedder
    from src.retriever import retrieve
    from src.vector_store import load_index

    index_dir = _PROJECT_ROOT / "data" / "faiss_index"
    idx, meta = load_index(index_dir)
    emb = embedder or Embedder()
    return retrieve(query, idx, meta, emb, k=k)


if __name__ == "__main__":
    # Run from ``rag_chatbot`` root:  python src/prompt_builder.py
    from src.embedder import Embedder
    from src.retriever import retrieve
    from src.vector_store import load_index

    queries = [
        "Who won the presidential election?",
        "What is the education budget allocation?",
    ]
    shared_embedder = Embedder()
    idx, meta = load_index(_PROJECT_ROOT / "data" / "faiss_index")
    for q in queries:
        print(f"\n{'#' * 80}\nQuery: {q}\n{'#' * 80}")
        chunks = retrieve(q, idx, meta, shared_embedder, k=8)
        if not chunks:
            print("No retrieved chunks (build index with src/retriever.py or vector_store first).")
            continue
        compare_prompt_templates(q, chunks)
