# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
FAISS vector storage with parallel chunk metadata (manual RAG; no hosted vector DB).

Why FAISS ``IndexFlatL2`` instead of Chroma or Pinecone for this project?
--------------------------------------------------------------------------------
- **Corpus size**: Ghana election + budget chunks are on the order of hundreds to a few thousand
  vectors. A **flat** index (exact search over all stored vectors) is fast enough on a laptop and
  is the simplest correct baseline for coursework—no approximate-ANN tuning required.
- **No extra services**: **Chroma** and **Pinecone** add processes, accounts, or cloud APIs. This
  assignment targets a **self-contained** Python pipeline: FAISS runs in-process with no API keys
  and no network dependency after install.
- **Teaching clarity**: ``IndexFlatL2`` maps directly to “store matrix of embeddings + Euclidean
  distance to query”—easy to explain next to retrieval metrics. Hosted DBs hide indexing details
  behind SDKs.
- **Upgrade path**: If the corpus grew to millions of vectors, you would switch to FAISS IVF/HNSW
  or a managed service; that is unnecessary at this scale.

``IndexFlatL2`` uses **squared** L2 distance (lower is more similar). Query vectors should use the
same embedding model and preprocessing as the indexed vectors.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

INDEX_FILENAME = "faiss_index.bin"
METADATA_FILENAME = "chunks_metadata.pkl"


def _stack_embeddings(embedded_chunks: list[dict[str, Any]]) -> tuple[np.ndarray, int]:
    if not embedded_chunks:
        raise ValueError("embedded_chunks must be non-empty to infer embedding dimension")
    vectors: list[np.ndarray] = []
    dim: int | None = None
    for i, ch in enumerate(embedded_chunks):
        emb = ch.get("embedding")
        if emb is None:
            raise ValueError(f"Chunk {i} has no 'embedding' field")
        v = np.asarray(emb, dtype=np.float32).reshape(-1)
        if dim is None:
            dim = v.shape[0]
        elif v.shape[0] != dim:
            raise ValueError(f"Chunk {i} embedding dim {v.shape[0]} != {dim}")
        vectors.append(v)
    mat = np.stack(vectors, axis=0).astype(np.float32, copy=False)
    return mat, dim


def build_index(
    embedded_chunks: list[dict[str, Any]],
) -> tuple[faiss.IndexFlatL2, list[dict[str, Any]]]:
    """
    Build a FAISS ``IndexFlatL2`` and a parallel list of ``{text, metadata}`` (no embeddings stored).

    ``embedded_chunks`` must follow the embedder output shape: each item has ``text``, ``metadata``,
    and ``embedding`` (1-D float array).
    """
    mat, dim = _stack_embeddings(embedded_chunks)
    index = faiss.IndexFlatL2(dim)
    index.add(mat)

    chunks_metadata: list[dict[str, Any]] = []
    for ch in embedded_chunks:
        chunks_metadata.append(
            {
                "text": ch["text"],
                "metadata": dict(ch.get("metadata", {})),
            }
        )

    if index.ntotal != len(chunks_metadata):
        raise RuntimeError("FAISS ntotal does not match metadata length")

    return index, chunks_metadata


def save_index(
    index: faiss.Index,
    chunks_metadata: list[dict[str, Any]],
    path: str | Path = "data/faiss_index",
) -> None:
    """
    Persist the FAISS index to ``faiss_index.bin`` and metadata to ``chunks_metadata.pkl``
    under ``path`` (directory created if needed).
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)

    index_path = dir_path / INDEX_FILENAME
    meta_path = dir_path / METADATA_FILENAME

    faiss.write_index(index, str(index_path))
    with meta_path.open("wb") as f:
        pickle.dump(chunks_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_index(path: str | Path = "data/faiss_index") -> tuple[faiss.Index, list[dict[str, Any]]]:
    """Load ``(index, chunks_metadata)`` from disk."""
    dir_path = Path(path)
    index_path = dir_path / INDEX_FILENAME
    meta_path = dir_path / METADATA_FILENAME

    if not index_path.is_file():
        raise FileNotFoundError(f"Missing FAISS index: {index_path.resolve()}")
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing metadata pickle: {meta_path.resolve()}")

    index = faiss.read_index(str(index_path))
    with meta_path.open("rb") as f:
        chunks_metadata = pickle.load(f)

    if index.ntotal != len(chunks_metadata):
        raise ValueError(
            f"Index ntotal ({index.ntotal}) != len(chunks_metadata) ({len(chunks_metadata)})"
        )

    return index, chunks_metadata


if __name__ == "__main__":
    # Demo: build from fake embedded chunks (384-dim like all-MiniLM-L6-v2), save, reload, verify count.
    #
    # From project root:
    #   python src/vector_store.py
    #
    # With real embedder output:
    #   from src.embedder import Embedder
    #   from src.vector_store import build_index, save_index, load_index
    #   embedded = Embedder().embed_chunks(your_chunks)
    #   index, meta = build_index(embedded)
    #   save_index(index, meta, "data/faiss_index")
    #   index2, meta2 = load_index("data/faiss_index")
    rng = np.random.default_rng(0)
    dim = 384
    n = 5
    embedded_chunks: list[dict[str, Any]] = []
    for i in range(n):
        v = rng.standard_normal(dim, dtype=np.float32)
        v /= np.linalg.norm(v) + 1e-12
        embedded_chunks.append(
            {
                "text": f"Sample chunk {i}",
                "metadata": {"source": "demo", "chunk_index": i},
                "embedding": v,
            }
        )

    index, chunks_metadata = build_index(embedded_chunks)
    print("Built index - ntotal:", index.ntotal, "| metadata rows:", len(chunks_metadata))

    out_dir = Path(__file__).resolve().parent.parent / "data" / "faiss_index"
    save_index(index, chunks_metadata, out_dir)
    print("Saved to:", out_dir)

    index2, meta2 = load_index(out_dir)
    print("Loaded index - ntotal:", index2.ntotal, "| metadata rows:", len(meta2))
    print("Chunk counts match:", index2.ntotal == len(meta2) == n)
