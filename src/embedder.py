# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Embedding pipeline for RAG chunks and user queries.

Why sentence-transformers instead of OpenAI embeddings?
--------------------------------------------------------------------------------
- **No API key / no per-token cost**: Suitable for a coursework setting and repeatable
  experiments without cloud billing or quota limits.
- **Offline-capable**: Once the model weights are cached, embeddings work without network
  access—useful for demos, grading, and reproducibility.
- **Speed vs. this corpus**: Election + budget chunks are in the hundreds to low thousands;
  MiniLM-class encoders run comfortably on CPU or modest GPUs, so end-to-end indexing stays
  practical for the assignment timeline.
- **Standard cosine / inner-product retrieval**: With L2-normalized vectors, similarity search
  matches typical FAISS setups used in teaching RAG without vendor lock-in.

OpenAI embeddings remain a strong production option when you need maximum quality and already
have API infrastructure; here local sentence-transformers trade a small accuracy gap for
simplicity and control.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Hugging Face / sentence-transformers model id (384-dimensional embeddings).
MODEL_NAME = "all-MiniLM-L6-v2"


class Embedder:
    """Wraps ``all-MiniLM-L6-v2`` for chunk and query embeddings (L2-normalized)."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        *,
        batch_size: int = 32,
        device: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError("Install sentence-transformers: pip install sentence-transformers") from e

        kwargs: dict[str, Any] = {}
        if device:
            kwargs["device"] = device
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name, **kwargs)
        get_dim = getattr(self._model, "get_embedding_dimension", None)
        self.dimension = (
            int(get_dim()) if callable(get_dim) else int(self._model.get_sentence_embedding_dimension())
        )

    def embed_chunks(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Embed each chunk's ``text`` and return the same items with ``embedding`` added.

        Input format (from ``chunker``): ``{"text": str, "metadata": dict}``.

        Output format: ``{"text", "metadata", "embedding"}`` where ``embedding`` is a 1-D
        ``numpy.ndarray`` (float32, L2-normalized).

        Prints progress every 100 chunks (and a final line if the total is not a multiple of 100).
        """
        if not chunks:
            return []

        n = len(chunks)
        out: list[dict[str, Any]] = []

        for start in range(0, n, 100):
            end = min(start + 100, n)
            batch = chunks[start:end]
            texts = [c["text"] for c in batch]
            vectors = self._model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            arr = np.asarray(vectors, dtype=np.float32)
            for i, ch in enumerate(batch):
                row = dict(ch)
                row["embedding"] = arr[i]
                out.append(row)
            print(f"Embedded {end} / {n} chunks")

        return out

    def embed_query(self, query_text: str) -> np.ndarray:
        """Return a single L2-normalized query vector (shape ``(dimension,)``)."""
        q = self._model.encode(
            [query_text],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        v = np.asarray(q[0], dtype=np.float32)
        return v


def embed_chunks(chunks: list[dict[str, Any]], embedder: Embedder | None = None) -> list[dict[str, Any]]:
    """
    Module-level helper: embed chunks using a shared or new :class:`Embedder`.

    If ``embedder`` is omitted, a new model instance is created (loads weights once per call).
    Prefer passing one ``Embedder`` for batch pipelines.
    """
    e = embedder or Embedder()
    return e.embed_chunks(chunks)


def embed_query(query_text: str, embedder: Embedder | None = None) -> np.ndarray:
    """Module-level helper for a single query embedding."""
    e = embedder or Embedder()
    return e.embed_query(query_text)


if __name__ == "__main__":
    # Test: embed 5 sample chunks and print vector shapes.
    #
    # From project root (rag_chatbot):
    #   python src/embedder.py
    #
    # Or:
    #   python -m src.embedder
    sample_chunks = [
        {
            "text": "In Ashanti Region, Nana Akufo Addo of NPP received 1795824 votes.",
            "metadata": {"source": "election", "chunk_index": 0, "strategy": "fixed", "page": None},
        },
        {
            "text": "The 2025 budget prioritizes fiscal consolidation and growth.",
            "metadata": {"source": "budget", "chunk_index": 0, "strategy": "sentence", "page": 3},
        },
        {
            "text": "Revenue measures include adjustments to VAT and excise frameworks.",
            "metadata": {"source": "budget", "chunk_index": 1, "strategy": "sentence", "page": 12},
        },
        {
            "text": "Debt sustainability remains central to medium-term macroeconomic policy.",
            "metadata": {"source": "budget", "chunk_index": 2, "strategy": "sentence", "page": 15},
        },
        {
            "text": "Capital expenditure is aligned with infrastructure and human development goals.",
            "metadata": {"source": "budget", "chunk_index": 3, "strategy": "sentence", "page": 20},
        },
    ]

    emb = Embedder()
    embedded = emb.embed_chunks(sample_chunks)

    print("Model:", MODEL_NAME)
    print("Embedding dimension (from model):", emb.dimension)
    for i, item in enumerate(embedded):
        vec = item["embedding"]
        print(f"Chunk {i}: embedding shape {vec.shape}, dtype {vec.dtype}")

    q = emb.embed_query("What was the NPP vote count in Ashanti?")
    print("Query embedding shape:", q.shape)
