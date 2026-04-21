# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Top-k retrieval over a FAISS ``IndexFlatL2`` index with optional distance filtering.

``similarity_score`` stores the **squared L2 distance** returned by FAISS (lower = closer).
For L2-normalized embeddings this is standard; values above the threshold are treated as
low-similarity / irrelevant for filtering.

Feedback loop re-ranking (novelty)
--------------------------------------------------------------------------------
Standard RAG is **stateless**: the same query usually yields the same vector order. This
module adds a **lightweight learning loop**: thumbs-up / thumbs-down rows in
``experiment_logs.json`` (chunk text + ``positive`` / ``negative``) are mapped to retrieved
passages and used to **nudge** effective squared-L2 scores—promoting liked chunks and
demoting disliked ones—then **re-sorting** before the LLM sees context.

Why it is novel for a student RAG stack:
- Most tutorials stop at fixed FAISS top-k; few connect **explicit user preference** back into
  retrieval without a hosted reranker API or full learning-to-rank training loop.
- It stays **transparent**: fixed ±0.15 adjustments on known scores are easy to explain and tune.

How retrieval improves over time:
- Feedback accumulates per chunk snippet. Future queries that surface overlapping text can
  **surface or bury** passages based on past judgements—adding domain supervision on top of
  static embeddings.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.embedder import Embedder

# Default: drop hits whose squared L2 distance exceeds this (higher = farther in embedding space).
DEFAULT_MAX_L2_SQUARED = 1.5

# Nudge applied to squared L2 distance when feedback matches a retrieved chunk (lower distance = better).
RERANK_BONUS_POSITIVE = 0.15
RERANK_PENALTY_NEGATIVE = 0.15


def _normalize_chunk_text(text: str) -> str:
    return " ".join((text or "").split())


def load_feedback_history(log_path: str | Path | None = None) -> dict[str, str]:
    """
    Read ``logs/experiment_logs.json`` and build a map **chunk text snippet → feedback**.

    Only entries containing a ``"feedback"`` field (``"positive"`` / ``"negative"``) and a
    ``top_chunk`` string are kept. Later entries overwrite earlier ones for the same
    normalized chunk key.
    """
    path = Path(log_path) if log_path else _PROJECT_ROOT / "logs" / "experiment_logs.json"
    if not path.is_file():
        return {}

    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return {}

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(data, list):
        return {}

    out: dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if "feedback" not in entry:
            continue
        fb = entry.get("feedback")
        chunk = entry.get("top_chunk")
        if chunk is None or fb not in ("positive", "negative"):
            continue
        key = _normalize_chunk_text(str(chunk))
        if key:
            out[key] = str(fb)
    return out


def _feedback_delta_for_chunk(chunk_text: str, history: dict[str, str]) -> float:
    """
    Return additive delta to **squared L2 distance**:
    thumbs-up → subtract ``RERANK_BONUS_POSITIVE`` (better rank); thumbs-down → add penalty.
    """
    if not history:
        return 0.0

    n = _normalize_chunk_text(chunk_text)
    if not n:
        return 0.0

    for key, fb in history.items():
        kn = _normalize_chunk_text(key)
        if kn and n == kn:
            return -RERANK_BONUS_POSITIVE if fb == "positive" else RERANK_PENALTY_NEGATIVE

    # Longest plausible substring overlap between logged snippet and full chunk text.
    best_fb: str | None = None
    best_prio = -1
    min_key = 24

    for key, fb in history.items():
        kn = _normalize_chunk_text(key)
        if not kn:
            continue
        if kn in n and len(kn) >= min(min_key, max(8, len(n) // 8)):
            prio = len(kn)
            if prio > best_prio:
                best_prio = prio
                best_fb = fb
        elif n in kn and len(n) >= min(min_key, max(8, len(kn) // 8)):
            prio = len(n)
            if prio > best_prio:
                best_prio = prio
                best_fb = fb

    if best_fb == "positive":
        return -RERANK_BONUS_POSITIVE
    if best_fb == "negative":
        return RERANK_PENALTY_NEGATIVE
    return 0.0


def expand_query(query: str) -> str:
    """
    Append 2–3 domain terms per matched keyword to steer the embedding toward useful vocabulary.

    Rules (case-insensitive substring match):
    - "election" -> votes constituency party
    - "budget" -> expenditure revenue fiscal
    - "gdp" -> economic growth output
    - "president" -> candidate winner votes
    """
    q = query.strip()
    if not q:
        return q

    low = q.lower()
    extras: list[str] = []

    if "election" in low:
        extras.append("votes constituency party")
    if "budget" in low:
        extras.append("expenditure revenue fiscal")
    if "gdp" in low:
        extras.append("economic growth output")
    if "president" in low:
        extras.append("candidate winner votes")

    if not extras:
        return q
    return f"{q} {' '.join(extras)}"


def retrieve(
    query: str,
    index: faiss.Index,
    chunks_metadata: list[dict[str, Any]],
    embedder: Embedder | None = None,
    k: int = 5,
    *,
    max_l2_squared: float | None = DEFAULT_MAX_L2_SQUARED,
    apply_query_expansion: bool = True,
    rerank: bool = False,
    feedback_log_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """
    Embed the (optionally expanded) query, search FAISS, return ranked hits.

    Each hit includes ``similarity_score`` (raw **squared L2 distance** from FAISS).

    If ``rerank=True``, loads feedback history from ``experiment_logs.json`` and adjusts an
    ``effective_score`` (same units as L2²): thumbs-up chunks get a lower effective distance,
    thumbs-down a higher one; results are **re-sorted** by ``effective_score``. Fields
    ``feedback_delta`` and ``effective_score`` are added; when ``rerank=False``,
    ``effective_score`` equals ``similarity_score`` and ``feedback_delta`` is ``0.0``.

    If ``max_l2_squared`` is not ``None``, results with ``similarity_score > max_l2_squared`` are
    dropped before re-ranking. Extra neighbors are retrieved so that up to ``k`` results may remain after filtering.
    """
    if index.ntotal == 0:
        return []
    if len(chunks_metadata) != index.ntotal:
        raise ValueError(f"chunks_metadata length {len(chunks_metadata)} != index.ntotal {index.ntotal}")

    embedder = embedder or Embedder()
    text_for_embed = expand_query(query) if apply_query_expansion else query.strip()
    q_vec = embedder.embed_query(text_for_embed)
    q_np = np.asarray(q_vec, dtype=np.float32).reshape(1, -1)

    # Request enough neighbors to survive filtering.
    pool = min(index.ntotal, max(k * 20, k + 50))

    distances, indices = index.search(q_np, pool)
    dist_row = distances[0]
    idx_row = indices[0]

    out: list[dict[str, Any]] = []
    rank = 0
    for dist, idx in zip(dist_row.tolist(), idx_row.tolist()):
        if idx < 0:
            continue
        if max_l2_squared is not None and dist > max_l2_squared:
            continue
        if idx >= len(chunks_metadata):
            continue
        rank += 1
        meta = chunks_metadata[idx]
        out.append(
            {
                "rank": rank,
                "text": meta["text"],
                "metadata": dict(meta.get("metadata", {})),
                "similarity_score": float(dist),
            }
        )
        if len(out) >= k:
            break

    if not rerank:
        for row in out:
            row["feedback_delta"] = 0.0
            row["effective_score"] = row["similarity_score"]
        return _renumber_ranks(out)

    history = load_feedback_history(feedback_log_path)
    for row in out:
        delta = _feedback_delta_for_chunk(row["text"], history)
        raw = float(row["similarity_score"])
        eff = max(0.0, raw + delta)
        row["feedback_delta"] = float(delta)
        row["effective_score"] = float(eff)

    out.sort(key=lambda r: r["effective_score"])
    return _renumber_ranks(out)


def _renumber_ranks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def demonstrate_reranking(
    query: str,
    index: faiss.Index,
    chunks_metadata: list[dict[str, Any]],
    embedder: Embedder | None = None,
    *,
    k: int = 5,
    log_path: str | Path | None = None,
) -> None:
    """
    Inserts a synthetic **positive** feedback row for one retrieved chunk, then prints
    ranking with ``rerank=False`` vs ``rerank=True`` (evidence that feedback changes order).

    By default writes ``logs/rerank_demo_feedback.json`` (isolated from the main experiment log).
    Pass ``log_path`` pointing at ``logs/experiment_logs.json`` to exercise the same file
    ``load_feedback_history`` uses in production.
    """
    # Use a dedicated JSON file so the demo does not mix with unrelated experiment_logs rows.
    path = (
        Path(log_path)
        if log_path
        else _PROJECT_ROOT / "logs" / "rerank_demo_feedback.json"
    )
    embedder = embedder or Embedder()

    base = retrieve(
        query,
        index,
        chunks_metadata,
        embedder,
        k=k,
        rerank=False,
    )
    if len(base) < 2:
        print("Need at least 2 retrieved chunks to demonstrate re-ranking. Got:", len(base))
        return

    # Nudge the 2nd hit so a rank swap vs the 1st is often visible after re-sorting.
    snippet = (base[1].get("text") or "")[:400]

    fake = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "feedback": "positive",
        "query": f"[demo_rerank] {query}",
        "top_chunk": snippet,
        "note": "synthetic feedback for demonstrate_reranking()",
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([fake], ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote synthetic positive feedback to {path.resolve()}")
    print(f"Target chunk (2nd hit) preview: {snippet[:120]}...\n")

    without = retrieve(
        query,
        index,
        chunks_metadata,
        embedder,
        k=k,
        rerank=False,
    )
    with_rerank = retrieve(
        query,
        index,
        chunks_metadata,
        embedder,
        k=k,
        rerank=True,
        feedback_log_path=path,
    )

    print("=== BEFORE (rerank=False) — order by raw FAISS L2² ===")
    for r in without:
        print(f"  rank {r['rank']}: score={r['similarity_score']:.4f}  {r['text'][:80]}...")

    print("\n=== AFTER (rerank=True) — order by effective_score after feedback nudges ===")
    for r in with_rerank:
        print(
            f"  rank {r['rank']}: raw={r['similarity_score']:.4f}  "
            f"delta={r['feedback_delta']:+.2f}  eff={r['effective_score']:.4f}  "
            f"{r['text'][:80]}..."
        )

    print(
        "\nIf feedback matched, the favoured chunk should move up relative to neighbours "
        "(lower effective_score is better).\n"
    )


def demonstrate_failure_case(
    index: faiss.Index,
    chunks_metadata: list[dict[str, Any]],
    embedder: Embedder | None = None,
    *,
    log_path: str | Path | None = None,
    k: int = 5,
) -> None:
    """
    Show how a vague query returns weak matches without a distance cutoff, and cleaner results with it.

    Appends one JSON record to ``logs/experiment_logs.json``.
    """
    vague = "What happened?"
    embedder = embedder or Embedder()

    print("\n=== Failure case: vague query (no distance filter) ===\n")
    bad = retrieve(vague, index, chunks_metadata, embedder, k=k, max_l2_squared=None)
    for r in bad:
        print(
            f"rank={r['rank']}  score={r['similarity_score']:.4f}  "
            f"meta={r['metadata']}  text={r['text'][:120]}..."
        )
    if not bad:
        print("(no results)")

    print("\n=== Same query WITH distance filter (max squared L2 = {}) ===\n".format(DEFAULT_MAX_L2_SQUARED))
    good = retrieve(vague, index, chunks_metadata, embedder, k=k, max_l2_squared=DEFAULT_MAX_L2_SQUARED)
    for r in good:
        print(
            f"rank={r['rank']}  score={r['similarity_score']:.4f}  "
            f"meta={r['metadata']}  text={r['text'][:120]}..."
        )
    if not good:
        print("(no results passed the filter - vague queries may retrieve nothing useful)")

    print(
        "\nEffect: filtering removes high-distance (low-similarity) chunks so the top list "
        "is less arbitrary for underspecified questions.\n"
    )

    base = Path(__file__).resolve().parent.parent
    path = Path(log_path) if log_path else base / "logs" / "experiment_logs.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "experiment": "retriever_failure_case",
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "query": vague,
        "without_distance_filter": bad,
        "with_distance_filter": good,
        "max_l2_squared_threshold": DEFAULT_MAX_L2_SQUARED,
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
    print(f"Logged failure-case comparison to: {path.resolve()}")


def _load_or_build_index(embedder: Embedder) -> tuple[faiss.Index, list[dict[str, Any]]]:
    from src.data_loader import load_budget_pdf, load_election_csv
    from src.vector_store import build_index, load_index, save_index

    root = _PROJECT_ROOT
    index_dir = root / "data" / "faiss_index"
    bin_path = index_dir / "faiss_index.bin"

    if bin_path.is_file():
        idx, meta = load_index(index_dir)
        # Rebuild if this is the small random demo from ``vector_store.py`` self-test or near-empty.
        stale_demo = (
            idx.ntotal < 20
            or (meta and meta[0].get("text", "").startswith("Sample chunk "))
        )
        if not stale_demo:
            return idx, meta
        print("Replacing stale demo index with full election + budget embeddings...")

    data_dir = root / "data"
    election = load_election_csv(data_dir / "Ghana_Election_Result.csv")
    budget_pages = load_budget_pdf(data_dir / "2025-Budget-Statement-and-Economic-Policy_v4.pdf")

    chunks: list[dict[str, Any]] = list(election)
    for pg in budget_pages:
        chunks.append(
            {
                "text": pg["text"],
                "metadata": {"source": "budget", "page_number": pg["page_number"]},
            }
        )

    print(f"Building embeddings for {len(chunks)} chunks (first-time index)...")
    embedded = embedder.embed_chunks(chunks)
    index, meta = build_index(embedded)
    save_index(index, meta, index_dir)
    print(f"Saved index to {index_dir}")
    return index, meta


def _print_results(title: str, results: list[dict[str, Any]]) -> None:
    print(f"\n--- {title} ---\n")
    if not results:
        print("(no results)\n")
        return
    for r in results:
        print(f"rank={r['rank']}  similarity_score={r['similarity_score']:.4f}")
        print(f"  text: {r['text'][:200]}{'...' if len(r['text']) > 200 else ''}")
        print(f"  metadata: {r['metadata']}\n")


if __name__ == "__main__":
    # Run from ``rag_chatbot`` root:
    #   python src/retriever.py
    #   python src/retriever.py --demo-rerank
    emb = Embedder()
    faiss_index, chunks_meta = _load_or_build_index(emb)

    if "--demo-rerank" in sys.argv:
        demonstrate_reranking(
            "Who won the presidential election in Ghana?",
            faiss_index,
            chunks_meta,
            emb,
            k=5,
        )
    else:
        queries = [
            "Who won the most votes in Accra Central?",
            "What is Ghana's projected revenue in the 2025 budget?",
            "What happened?",
        ]

        for q in queries:
            expanded = expand_query(q)
            print(f"\nQuery: {q}")
            print(f"Expanded (used for embedding): {expanded}")
            results = retrieve(q, faiss_index, chunks_meta, emb, k=5)
            _print_results(f"Top 5 (with filter, k=5): {q[:50]}...", results)

        demonstrate_failure_case(faiss_index, chunks_meta, emb, k=5)
