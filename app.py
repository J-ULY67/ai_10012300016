# Student Name: Nana Kwaku Owusu-Ansah
# Index Number: 10012300016
# Course: CS4241 - Introduction to Artificial Intelligence
# Lecturer: Godwin N. Danso

"""
Streamlit chat UI for the Academic City RAG assistant.

Run::

    streamlit run app.py
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any

import streamlit as st

BASE = Path(__file__).resolve().parent
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from src.logger import log_feedback
from src.pipeline import load_pipeline

STUDENT_NAME = "Nana Kwaku Owusu-Ansah"
STUDENT_INDEX = "10012300016"


@st.cache_resource(show_spinner="Loading FAISS index and embedding model…")
def get_pipeline():
    return load_pipeline()


def _top_chunk_text(retrieved: list[dict[str, Any]]) -> str:
    if not retrieved:
        return ""
    return (retrieved[0].get("text") or "").strip()


def _text_to_html(text: str) -> str:
    return html.escape(text or "").replace("\n", "<br/>")


def _inject_styles() -> None:
    st.markdown(
        r"""
        <style>
            /* Theme-aware main area: follow Streamlit light/dark */
            @media (prefers-color-scheme: light) {
                div.block-container {
                    background-color: #ffffff !important;
                }
            }
            @media (prefers-color-scheme: dark) {
                div.block-container {
                    background-color: transparent !important;
                }
            }
            div.block-container {
                padding-bottom: 120px !important;
                max-width: 880px !important;
            }

            /* Sender labels — theme text (not pure black/white) */
            .bubble-role-label {
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0.02em;
                margin-bottom: 6px;
                color: #1a1a1a;
            }
            @media (prefers-color-scheme: dark) {
                .bubble-role-label {
                    color: #f0f0f0;
                }
            }
            .bubble-role-label-user { text-align: right; }
            .bubble-role-label-assistant { text-align: left; }

            .bubble-user-wrap {
                display: flex;
                flex-direction: column;
                align-items: flex-end;
                margin: 12px 0 20px 0;
            }
            .bubble-assistant-wrap {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                margin: 12px 0 8px 0;
            }

            /* User bubble — navy; text stays white for contrast on both themes */
            .user-bubble {
                background: #1e3a5f;
                color: #ffffff;
                padding: 12px 16px;
                border-radius: 18px 18px 4px 18px;
                margin: 8px 0;
                max-width: 75%;
                margin-left: auto;
                font-size: 15px;
                line-height: 1.5;
                text-align: left;
            }
            @media (prefers-color-scheme: dark) {
                .user-bubble {
                    background: #2a4a7f;
                    color: #ffffff;
                }
            }

            /* Assistant bubble — light/dark surfaces */
            .assistant-bubble {
                background: #f0f0f0;
                color: #1a1a1a;
                padding: 12px 16px;
                border-radius: 18px 18px 18px 4px;
                margin: 8px 0;
                max-width: 75%;
                font-size: 15px;
                line-height: 1.5;
                text-align: left;
                border: 1px solid #e0e0e0;
            }
            @media (prefers-color-scheme: dark) {
                .assistant-bubble {
                    background: #2d2d2d;
                    color: #f0f0f0;
                    border-color: #404040;
                }
            }

            /* Streamlit in-app theme toggle (may differ from OS prefers-color-scheme) */
            [data-theme="dark"] .bubble-role-label {
                color: #f0f0f0 !important;
            }
            [data-theme="dark"] .user-bubble {
                background: #2a4a7f !important;
                color: #ffffff !important;
            }
            [data-theme="dark"] .assistant-bubble {
                background: #2d2d2d !important;
                color: #f0f0f0 !important;
                border-color: #404040 !important;
            }
            [data-theme="dark"] div.block-container {
                background-color: transparent !important;
            }

            /* Do not force expander summary to a fixed gray — let theme handle it */
            div[data-testid="stExpander"] details > summary {
                font-size: 13px !important;
            }

            /* Input: visible in light and dark */
            .stTextInput input,
            div[data-testid="stTextInput"] input {
                border: 1.5px solid #666666 !important;
                font-size: 15px !important;
                border-radius: 12px !important;
                min-height: 48px !important;
                padding-left: 14px !important;
            }
            @media (prefers-color-scheme: dark) {
                .stTextInput input,
                div[data-testid="stTextInput"] input {
                    border: 1.5px solid #999999 !important;
                    color: #f0f0f0 !important;
                    background: #1e1e1e !important;
                }
            }
            [data-theme="dark"] .stTextInput input,
            [data-theme="dark"] div[data-testid="stTextInput"] input {
                border: 1.5px solid #999999 !important;
                color: #f0f0f0 !important;
                background: #1e1e1e !important;
            }
            div[data-testid="stTextInput"] input:focus {
                outline: none !important;
            }

            div[data-testid="stButton"] > button[kind="primary"] {
                border-radius: 12px !important;
                font-weight: 600 !important;
                min-height: 48px !important;
                padding-left: 22px !important;
                padding-right: 22px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_user_message(content: str) -> None:
    body = _text_to_html(content)
    st.markdown(
        f"""
        <div class="bubble-user-wrap">
            <div class="bubble-role-label bubble-role-label-user">You</div>
            <div class="user-bubble">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_assistant_message(content: str, detail: dict[str, Any] | None) -> None:
    body = _text_to_html(content)
    st.markdown(
        f"""
        <div class="bubble-assistant-wrap">
            <div class="bubble-role-label bubble-role-label-assistant">Academic City RAG</div>
            <div class="assistant-bubble">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not detail:
        return

    with st.expander("🔍 View sources", expanded=False):
        st.markdown("Retrieval details for markers — expand sections below.")
        chunks = detail.get("retrieved_chunks") or []
        if not chunks:
            st.info("No chunks passed the distance filter.")
        else:
            for row in chunks:
                rank = row.get("rank")
                meta = row.get("metadata") or {}
                src = meta.get("source", "unknown")
                score = row.get("similarity_score")
                score_s = f"{float(score):.6f}" if score is not None else "n/a"
                snippet = (row.get("text") or "")[:100]
                if len((row.get("text") or "")) > 100:
                    snippet += "…"
                st.markdown(f"**Source:** `{src}`  \n**Score:** `{score_s}`  \n**Rank:** {rank}")
                st.write("**Preview (100 chars):**")
                st.write(snippet)
                st.divider()
        st.markdown("**Full prompt sent to LLM**")
        st.code(detail.get("prompt", "") or "", language="text")


def main() -> None:
    st.set_page_config(
        page_title="Academic City RAG Assistant",
        page_icon="🎓",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    _inject_styles()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_feedback_query" not in st.session_state:
        st.session_state.last_feedback_query = ""
    if "last_feedback_chunks" not in st.session_state:
        st.session_state.last_feedback_chunks = []
    if "input_turn" not in st.session_state:
        st.session_state.input_turn = 0

    with st.sidebar:
        st.markdown("## Academic City RAG Assistant")
        st.markdown("---")

        top_k = st.slider("Top-k retrieval", min_value=1, max_value=10, value=5, step=1)
        template_mode = st.radio(
            "Prompt template",
            ("Strict", "Flexible"),
            index=0,
            horizontal=True,
            help="Strict: answer only from context. Flexible: may add clearly labeled general knowledge.",
        )
        template_num = 1 if template_mode == "Strict" else 2

        st.divider()
        st.markdown("**Feedback (last answer)**")
        fc1, fc2 = st.columns(2)
        with fc1:
            if st.button("👍", key="fb_positive", use_container_width=True):
                q = st.session_state.last_feedback_query
                tc = _top_chunk_text(st.session_state.last_feedback_chunks)
                if q:
                    log_feedback("positive", q, tc)
                    st.toast("Saved positive feedback.", icon="✅")
                else:
                    st.warning("Ask a question first.")
        with fc2:
            if st.button("👎", key="fb_negative", use_container_width=True):
                q = st.session_state.last_feedback_query
                tc = _top_chunk_text(st.session_state.last_feedback_chunks)
                if q:
                    log_feedback("negative", q, tc)
                    st.toast("Saved negative feedback.", icon="✅")
                else:
                    st.warning("Ask a question first.")

        st.divider()
        st.caption(STUDENT_NAME)
        st.caption(f"Index {STUDENT_INDEX}")

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            _render_user_message(msg.get("content", ""))
        else:
            _render_assistant_message(msg.get("content", ""), msg.get("detail"))

    st.write("")

    icol, bcol = st.columns([5, 1])
    with icol:
        user_text = st.text_input(
            "message",
            key=f"chat_field_{st.session_state.input_turn}",
            placeholder="Ask about Ghana elections or the 2025 budget...",
            label_visibility="collapsed",
        )
    with bcol:
        send = st.button("Send", type="primary", use_container_width=True)

    if send and user_text and user_text.strip():
        prompt = user_text.strip()
        st.session_state.messages.append({"role": "user", "content": prompt})

        pipe = get_pipeline()
        with st.spinner("Thinking…"):
            result = pipe.run_pipeline(
                prompt,
                template=template_num,
                k=top_k,
                log_to_file=False,
                verbose=False,
            )

        retrieved = result.get("retrieved_chunks") or []
        st.session_state.last_feedback_query = prompt
        st.session_state.last_feedback_chunks = retrieved

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("llm_response") or "",
                "detail": {
                    "expanded_query": result.get("expanded_query", ""),
                    "retrieved_chunks": retrieved,
                    "prompt": result.get("prompt_sent_to_llm") or "",
                },
            }
        )
        st.session_state.input_turn += 1
        st.rerun()


if __name__ == "__main__":
    main()
