# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
LLM calls via Groq (Llama 3.3 70B).

API keys must not be committed. Set ``GROQ_API_KEY`` in the environment or in a ``.env`` file
(see ``.gitignore``). Optional: ``python-dotenv`` loads ``.env`` automatically if installed.
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from groq import Groq
except ImportError as e:  # pragma: no cover
    raise ImportError("Install groq: pip install groq") from e

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[misc, assignment]

_MODEL = "llama-3.3-70b-versatile"

_ROOT = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(_ROOT / ".env")


def _api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "Set GROQ_API_KEY in your environment or in a .env file in the project root "
            "(never commit real keys to git)."
        )
    return key


_client: Groq | None = None


def _client_singleton() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=_api_key())
    return _client


def generate_response(prompt: str) -> str:
    """Send ``prompt`` as a single user message; return assistant text or an error string."""
    client = _client_singleton()
    try:
        response = client.chat.completions.create(
            model=_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.3,
        )
        choice = response.choices[0].message
        content = choice.content
        if content is None:
            return "Error generating response: empty content from model"
        return content
    except Exception as e:
        return f"Error generating response: {str(e)}"


if __name__ == "__main__":
    # Test: set GROQ_API_KEY first, then:
    #   python src/llm_handler.py
    out = generate_response("Say hello in one sentence")
    print(out)
