# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Load Ghana election CSV rows and Ghana budget PDF pages as structured text items for RAG.

Election CSV: one natural-language sentence per row with metadata.
Budget PDF: one cleaned text block per page (pdfplumber), with page_number for citation.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import pdfplumber
except ImportError as e:  # pragma: no cover
    raise ImportError("pdfplumber is required: pip install pdfplumber") from e


def _normalize_column_name(name: str) -> str:
    """Lowercase, underscores, and stable names for columns like Votes(%)."""
    n = name.strip().lower().replace("%", "pct")
    n = n.replace("(", "_").replace(")", "")
    n = re.sub(r"\s+", "_", n)
    n = re.sub(r"[^a-z0-9_]+", "_", n)
    n = re.sub(r"_+", "_", n).strip("_")
    return n


def _strip_string_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    return out


def load_election_csv(csv_path: str | Path) -> list[dict[str, Any]]:
    """
    Load election results, clean, and emit one readable sentence per row.

    Each item: ``{"text": str, "metadata": {"source": "election", "row_index": int}}``.

    Note: This dataset uses *regions* (new_region), not formal constituencies; the sentence
    template uses the geographic label as [constituency] per assignment wording.
    """
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    df = pd.read_csv(path)
    df.columns = [_normalize_column_name(c) for c in df.columns]
    df = df.dropna(how="any")
    df = _strip_string_columns(df)

    # Prefer explicit constituency if present; else use new_region (this file's geography).
    has_constituency = "constituency" in df.columns
    has_new_region = "new_region" in df.columns
    if not has_constituency and not has_new_region:
        raise ValueError("CSV must include 'constituency' or 'new_region' after normalization.")

    required = {"candidate", "party", "votes"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")

    results: list[dict[str, Any]] = []
    for row_index, row in df.reset_index(drop=True).iterrows():
        constituency = (
            str(row["constituency"]).strip()
            if has_constituency
            else str(row["new_region"]).strip()
        )
        candidate = str(row["candidate"]).strip()
        party = str(row["party"]).strip()
        votes_val = row["votes"]
        if pd.isna(votes_val):
            continue
        try:
            votes_int = int(float(str(votes_val).replace(",", "")))
        except ValueError:
            continue

        text = (
            f"In {constituency}, {candidate} of {party} received {votes_int} votes."
        )
        results.append(
            {
                "text": text,
                "metadata": {"source": "election", "row_index": int(row_index)},
            }
        )

    return results


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _remove_repeated_headers_footers(
    page_lines: list[list[str]],
    min_pages_for_freq: int = 3,
    freq_ratio: float = 0.35,
) -> list[list[str]]:
    """
    Drop short lines that repeat across many pages (typical running headers/footers/page numbers).

    Long lines are kept even if repeated (unlikely to be boilerplate).
    """
    num_pages = len(page_lines)
    if num_pages == 0:
        return page_lines

    line_counts: Counter[str] = Counter()
    for lines in page_lines:
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if len(s) > 120:
                continue
            line_counts[s] += 1

    threshold = max(min_pages_for_freq, int(num_pages * freq_ratio))
    boilerplate = {line for line, c in line_counts.items() if c >= threshold}

    cleaned: list[list[str]] = []
    for lines in page_lines:
        kept: list[str] = []
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s in boilerplate:
                continue
            kept.append(s)
        cleaned.append(kept)
    return cleaned


def load_budget_pdf(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text per page with pdfplumber, normalize whitespace, and strip repeated headers/footers.

    Returns a list of::
        {"text": str, "page_number": int, "source": "budget"}
    """
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path.resolve()}")
    if path.stat().st_size == 0:
        # Placeholder files from scaffolding; replace with the real budget PDF for extraction.
        return []

    page_line_lists: list[list[str]] = []

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            lines = raw.splitlines()
            page_line_lists.append(lines)

    page_line_lists = _remove_repeated_headers_footers(page_line_lists)

    results: list[dict[str, Any]] = []
    for i, lines in enumerate(page_line_lists, start=1):
        page_text = _collapse_whitespace("\n".join(lines))
        if not page_text:
            continue
        results.append(
            {
                "text": page_text,
                "page_number": i,
                "source": "budget",
            }
        )

    return results


def load_all(
    csv_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    *,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Convenience loader using default filenames under ``data/`` next to this package.

    Returns ``{"election": [...], "budget": [...]}``.
    """
    base = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent / "data"
    csv_p = Path(csv_path) if csv_path else base / "Ghana_Election_Result.csv"
    pdf_p = Path(pdf_path) if pdf_path else base / "2025-Budget-Statement-and-Economic-Policy_v4.pdf"

    return {
        "election": load_election_csv(csv_p),
        "budget": load_budget_pdf(pdf_p),
    }


if __name__ == "__main__":
    # Example: from project root, run:
    #   python -m src.data_loader
    # or:
    #   cd rag_chatbot && python src/data_loader.py
    data = load_all()
    print("Election rows:", len(data["election"]))
    print("Budget pages (non-empty):", len(data["budget"]))
    if data["election"]:
        print("Sample:", data["election"][0]["text"][:200], "...")
    if data["budget"]:
        p0 = data["budget"][0]
        print("Budget page", p0["page_number"], "chars:", len(p0["text"]))
