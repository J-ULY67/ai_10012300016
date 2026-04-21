# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Text chunking for Ghana election + budget RAG.

Two strategies are implemented for Part A comparison:
  1) Fixed-size windows with overlap (fast, predictable).
  2) Sentence-aware grouping (preserves linguistic boundaries).

Hyperparameters (500 characters, 50 overlap for fixed strategy):
  - 500 characters is chosen because Ghana election rows are short (~100–200 chars) and budget
    PDF prose often arrives in multi-sentence paragraphs. ~500 chars fits roughly 3–6 English
    sentences—enough semantic context for MiniLM-class embeddings (256–512 token budgets) without
    stuffing entire pages into one vector. It also stays below common transformer truncation limits
    while remaining larger than a single short fact, which improves retrieval specificity versus
    tiny 100-char fragments.

  - 50 characters overlap (10% of the window) is chosen to reduce boundary effects: a fact that
    spans a fixed cut (e.g., a number at the end of one chunk and its label at the start of the
    next) still appears wholly in at least one window. 50 chars is small enough to avoid near-
    duplicate chunks dominating the index, but large enough to bridge typical word/phrase splits
    in budget tables and narrative text.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Callable

# Fixed strategy defaults (see module docstring for domain justification).
FIXED_CHUNK_SIZE = 500
FIXED_OVERLAP = 50


def _merge_base_metadata(
    base: dict[str, Any],
    *,
    chunk_index: int,
    strategy: str,
    page: int | None,
) -> dict[str, Any]:
    meta = dict(base)
    meta["chunk_index"] = chunk_index
    meta["strategy"] = strategy
    meta["page"] = page
    return meta


def chunk_text_fixed(
    text: str,
    base_metadata: dict[str, Any],
    *,
    chunk_size: int = FIXED_CHUNK_SIZE,
    overlap: int = FIXED_OVERLAP,
    page: int | None = None,
    page_for_offset: Callable[[int], int | None] | None = None,
) -> list[dict[str, Any]]:
    """
    Sliding character windows: ``chunk_size`` with ``overlap`` between consecutive windows.

    If ``page_for_offset`` is provided, ``metadata["page"]`` is set from the start offset of each
    chunk (for concatenated PDF text). Otherwise ``page`` is used for all chunks (e.g. per-page
    chunking).
    """
    text = text.strip()
    if not text:
        return []

    step = max(1, chunk_size - overlap)
    out: list[dict[str, Any]] = []
    chunk_index = 0
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        piece = text[start:end].strip()
        if piece:
            resolved_page: int | None
            if page_for_offset is not None:
                resolved_page = page_for_offset(start)
            else:
                resolved_page = page

            out.append(
                {
                    "text": piece,
                    "metadata": _merge_base_metadata(
                        base_metadata,
                        chunk_index=chunk_index,
                        strategy="fixed",
                        page=resolved_page,
                    ),
                }
            )
            chunk_index += 1
        if end >= len(text):
            break
        start += step

    return out


def _split_sentences(text: str) -> list[str]:
    """Simple sentence split (English prose + budget lines); not a legal-grade tokenizer."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence end followed by space/newline; keep delimiter via lookahead not needed.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def chunk_text_sentence(
    text: str,
    base_metadata: dict[str, Any],
    *,
    target_chars: int = FIXED_CHUNK_SIZE,
    page: int | None = None,
    page_for_offset: Callable[[int], int | None] | None = None,
) -> list[dict[str, Any]]:
    """
    Greedily pack whole sentences until adding the next would exceed ``target_chars``, then flush.

    Chunks may slightly exceed ``target_chars`` if a single sentence is longer (kept unsplit).
    """
    text = text.strip()
    if not text:
        return []

    sentences = _split_sentences(text)
    if not sentences:
        return []

    out: list[dict[str, Any]] = []
    chunk_index = 0
    buf: list[str] = []
    buf_len = 0
    # Global char offset in original ``text`` for page mapping (best-effort).
    search_from = 0

    def flush() -> None:
        nonlocal chunk_index, buf, buf_len, search_from
        if not buf:
            return
        piece = " ".join(buf).strip()
        if not piece:
            buf = []
            buf_len = 0
            return

        start_offset = text.find(piece[: min(80, len(piece))], search_from)
        if start_offset < 0:
            start_offset = text.find(piece, 0)
        if start_offset < 0:
            start_offset = 0
        search_from = start_offset + max(1, len(piece) // 2)

        if page_for_offset is not None:
            resolved_page = page_for_offset(start_offset)
        else:
            resolved_page = page

        out.append(
            {
                "text": piece,
                "metadata": _merge_base_metadata(
                    base_metadata,
                    chunk_index=chunk_index,
                    strategy="sentence",
                    page=resolved_page,
                ),
            }
        )
        chunk_index += 1
        buf = []
        buf_len = 0

    for sent in sentences:
        add_len = len(sent) if not buf else len(sent) + 1
        if buf and buf_len + add_len > target_chars:
            flush()
        buf.append(sent)
        buf_len += add_len

    flush()
    return out


def _stats_for_chunks(chunks: list[dict[str, Any]]) -> tuple[int, float, int, int]:
    lengths = [len(c["text"]) for c in chunks]
    if not lengths:
        return 0, 0.0, 0, 0
    return (
        len(lengths),
        float(statistics.mean(lengths)),
        min(lengths),
        max(lengths),
    )


def compare_chunking_strategies(sample_text: str) -> None:
    """
    Part A: run both strategies on the same sample and print comparative statistics.

    Prints chunk count, average chunk size, min and max chunk sizes (characters).
    """
    base = {"source": "demo"}
    fixed = chunk_text_fixed(sample_text, base, page=None)
    sent = chunk_text_sentence(sample_text, base, page=None)

    fc, favg, fmin, fmax = _stats_for_chunks(fixed)
    sc, savg, smin, smax = _stats_for_chunks(sent)

    print("=== Chunking comparison (Part A) ===")
    print(f"Input length: {len(sample_text)} characters\n")

    print("Strategy 1 - fixed (500 / 50 overlap)")
    print(f"  Chunks: {fc}")
    print(f"  Avg size: {favg:.1f} chars")
    print(f"  Min / max size: {fmin} / {fmax} chars\n")

    print("Strategy 2 - sentence-aware (~500 target)")
    print(f"  Chunks: {sc}")
    print(f"  Avg size: {savg:.1f} chars")
    print(f"  Min / max size: {smin} / {smax} chars")


if __name__ == "__main__":
    # Example: from project root (rag_chatbot):
    #   python -m src.chunker
    sample = (
        "The 2025 budget prioritizes fiscal consolidation. "
        "Capital expenditure is aligned with growth targets. "
        "Revenue measures include VAT adjustments. "
        "Debt sustainability remains a key objective for the medium term."
    ) * 8
    compare_chunking_strategies(sample)
