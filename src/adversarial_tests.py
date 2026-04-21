# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Adversarial queries: compare RAG (strict template + retrieval) vs. raw LLM, twice per query.

Hallucination flags use simple heuristics (refusal = safe; accepting a 70%% national
premise without refusal = flagged). Tune for your report as needed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.llm_handler import generate_response
from src.pipeline import load_pipeline

LOG_PATH = _PROJECT_ROOT / "logs" / "adversarial_test_results.json"

# --- Queries (assignment spec) ---
QUERY_AMBIGUOUS = "What major changes were announced in December?"
QUERY_MISLEADING = "Who won the 2020 Ghana presidential election with over 70% of votes?"


def _similarity(a: str, b: str) -> float:
    a, b = (a or "").strip(), (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return float(SequenceMatcher(None, a, b).ratio())


def _is_refusal(text: str) -> bool:
    t = text.lower()
    if "don't have enough information" in t or "do not have enough information" in t:
        return True
    if "i cannot answer" in t and "context" in t:
        return True
    return False


def _hallucination_rag_ambiguous(response: str, retrieved: list[dict[str, Any]]) -> bool:
    """Flag if answer asserts specific December facts not supported by retrieved text."""
    if _is_refusal(response):
        return False
    blob = " ".join((c.get("text") or "") for c in retrieved).lower()
    r = response.lower()
    if "december" in r and "december" not in blob:
        # Claims about December without December in context
        if re.search(r"\b(announced|passed|signed|budget|election)\b", r):
            return True
    return False


def _hallucination_rag_misleading(response: str) -> bool:
    """Trap question: no national presidential result had >70%. Affirming 70% = hallucination."""
    if _is_refusal(response):
        return False
    if re.search(r"7\s*0\.?\d*\s*%", response) or re.search(r"\bseventy\b", response, re.I):
        return True
    return False


def _hallucination_llm_misleading(response: str) -> bool:
    if _is_refusal(response):
        return False
    rl = response.lower()
    # Good behavior: explicitly rejects the false ">70%" premise or gives real vote shares.
    if "not over 70" in rl or "not 70" in rl or ("did not" in rl and "70" in rl):
        return False
    if re.search(r"\b5[0-2]\.\d+\s*%", response) and not re.search(r"7\s*0\s*%", response):
        return False
    if re.search(r"7\s*0\.?\d*\s*%", response) or re.search(r"\bseventy\b", response, re.I):
        return True
    if re.search(r"\b(won|winner|elected|victory)\b", response, re.I):
        return True
    return False


def _hallucination_llm_ambiguous(response: str) -> bool:
    if _is_refusal(response):
        return False
    # Common failure mode: generic world news unrelated to Ghana corpora.
    if re.search(
        r"\b(Meta|Google|SpaceX|OpenAI|NASA|Federal Reserve|European Union|EU's)\b",
        response,
        re.I,
    ):
        return True
    if re.search(r"\b(definitely|clearly|the only)\b", response, re.I):
        return True
    return False


def _evaluate(
    query: str,
    response: str,
    path: str,
    retrieved: list[dict[str, Any]] | None,
) -> bool:
    q = query.strip()
    if q == QUERY_MISLEADING:
        return _hallucination_rag_misleading(response) if path == "rag" else _hallucination_llm_misleading(response)
    if q == QUERY_AMBIGUOUS:
        if path == "rag":
            return _hallucination_rag_ambiguous(response, retrieved or [])
        return _hallucination_llm_ambiguous(response)
    return False


def _consistency_note(r1: str, r2: str, label: str) -> str:
    sim = _similarity(r1, r2)
    qual = "very similar" if sim >= 0.85 else ("somewhat similar" if sim >= 0.55 else "different")
    return f"{label}: ran twice, SequenceMatcher ratio={sim:.2f} ({qual})"


def _print_side_by_side(rag: str, pure: str, width: int = 52) -> None:
    print(f"\n{'RAG (with retrieval)':<{width}} | {'Pure LLM (query only)':<{width}}")
    print("-" * width + "-+-" + "-" * width)
    la = (rag or "").splitlines()
    lb = (pure or "").splitlines()
    n = max(len(la), len(lb))
    for i in range(n):
        a = la[i] if i < len(la) else ""
        b = lb[i] if i < len(lb) else ""
        if len(a) > width - 1:
            a = a[: width - 2] + "…"
        if len(b) > width - 1:
            b = b[: width - 2] + "…"
        print(f"{a:<{width}} | {b:<{width}}")


def run_adversarial_suite() -> list[dict[str, Any]]:
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            pass

    pipe = load_pipeline()
    queries = [QUERY_AMBIGUOUS, QUERY_MISLEADING]
    all_records: list[dict[str, Any]] = []

    for q in queries:
        print("\n" + "=" * 110)
        print(f"ADVERSARIAL QUERY: {q}")
        print("=" * 110)

        rag_runs: list[str] = []
        llm_runs: list[str] = []
        retrieved: list[dict[str, Any]] = []

        for run in (1, 2):
            print(f"\n--- Run {run}/2 ---\n")
            out = pipe.run_pipeline(q, template=1, k=5, log_to_file=False, verbose=False)
            rag_res = out["llm_response"]
            retrieved = out["retrieved_chunks"]
            pure_res = generate_response(q)
            rag_runs.append(rag_res)
            llm_runs.append(pure_res)

            print(f"[Run {run}] Side-by-side (truncated lines):")
            _print_side_by_side(rag_res, pure_res)
            if run == 1:
                print("\n[Run 1] Full RAG response:\n")
                print(rag_res)
                print("\n[Run 1] Full pure LLM response:\n")
                print(pure_res)
                print()

        rag_h1 = _evaluate(q, rag_runs[0], "rag", retrieved)
        rag_h2 = _evaluate(q, rag_runs[1], "rag", retrieved)
        llm_h1 = _evaluate(q, llm_runs[0], "llm", None)
        llm_h2 = _evaluate(q, llm_runs[1], "llm", None)

        rag_hall = rag_h1 or rag_h2
        llm_hall = llm_h1 or llm_h2

        note = (
            _consistency_note(rag_runs[0], rag_runs[1], "RAG")
            + "; "
            + _consistency_note(llm_runs[0], llm_runs[1], "Pure LLM")
        )

        print("\n--- Evaluation (automated heuristics; verify in your report) ---")
        print(f"Did RAG hallucinate? (either run flagged): {rag_hall}")
        print(f"Did pure LLM hallucinate? (either run flagged): {llm_hall}")
        print(f"Consistency: {note}\n")

        record = {
            "query": q,
            "rag_response": rag_runs[0],
            "pure_llm_response": llm_runs[0],
            "rag_response_run_2": rag_runs[1],
            "pure_llm_response_run_2": llm_runs[1],
            "hallucination_detected_rag": rag_hall,
            "hallucination_detected_llm": llm_hall,
            "consistency_note": note,
            "hallucination_run_detail": {
                "rag_run1": rag_h1,
                "rag_run2": rag_h2,
                "llm_run1": llm_h1,
                "llm_run2": llm_h2,
            },
        }
        all_records.append(record)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "results": all_records,
    }
    LOG_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {LOG_PATH.resolve()}\n")

    _print_summary_table(all_records)
    return all_records


def _print_summary_table(records: list[dict[str, Any]]) -> None:
    print("\n" + "#" * 110)
    print("SUMMARY — Adversarial comparison (heuristic hallucination flags)")
    print("#" * 110)
    hdr = f"{'Query (short)':<45} | {'RAG halluc?':<12} | {'LLM halluc?':<12} | {'Consistency (from note)':<30}"
    print(hdr)
    print("-" * len(hdr))
    for r in records:
        qshort = (r["query"][:42] + "...") if len(r["query"]) > 45 else r["query"]
        note = r.get("consistency_note", "")[:28] + "..." if len(r.get("consistency_note", "")) > 30 else r.get(
            "consistency_note", ""
        )
        print(
            f"{qshort:<45} | {str(r['hallucination_detected_rag']):<12} | "
            f"{str(r['hallucination_detected_llm']):<12} | {note:<30}"
        )
    print()


if __name__ == "__main__":
    run_adversarial_suite()
