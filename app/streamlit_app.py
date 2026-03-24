"""
streamlit_app.py – RAG chat interface.

Run with:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Optional

# Allow imports from project root when launched with `streamlit run app/...`
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st

from src.config import load_config, list_corpora
from src.rag_answer import AnswerResult, retrieve_and_answer, save_feedback


# ---------------------------------------------------------------------------
# Page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Framework",
    page_icon="🔍",
    layout="wide",
)


# ---------------------------------------------------------------------------
# FAISS index cache
# Keyed by (index_path, meta_path, mtime) so a rebuild auto-invalidates.
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_index_cached(index_path: str, meta_path: str, mtime: float):
    """Load FAISS index + chunk metadata once per (path, modification time)."""
    import faiss, json
    index  = faiss.read_index(index_path)
    with open(meta_path) as f:
        chunks = json.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_CONFIDENCE_LABEL = {
    "high":   ":green[🟢 High confidence]",
    "medium": ":orange[🟡 Medium confidence]",
    "low":    ":red[🔴 Low confidence]",
}


def _render_citations_inline(text: str) -> str:
    """Replace [ref_N] with bold **[N]** for Markdown display."""
    return re.sub(r"\[ref_(\d+)\]", r"**[\1]**", text)


def _chunk_location(chunk: dict) -> str:
    """Build 'Title › Section, p.N' location string from chunk metadata."""
    meta    = chunk.get("metadata") or {}
    title   = meta.get("title")   or "unknown"
    section = meta.get("section")
    page    = meta.get("page")
    loc     = title
    if section:
        loc += f" › {section}"
    if page:
        loc += f", p.{page}"
    return loc


def _render_result(result: AnswerResult, chunks: List[dict]) -> None:
    """
    Render a full AnswerResult inside the current st.chat_message block.
    Shows: answer text, confidence badge, citations panel, all-chunks expander.
    """
    # ---- Cannot-answer path ----
    if result["cannot_answer"] or not result["answer"].strip():
        st.warning(
            "The retrieved context doesn't contain sufficient information "
            "to answer this question."
        )
        if chunks:
            with st.expander(f"Retrieved chunks ({len(chunks)}) — none were sufficient"):
                _render_chunks_list(chunks, cited_ids=set())
        return

    # ---- Answer with inline citations ----
    rendered = _render_citations_inline(result["answer"])
    st.markdown(rendered)

    # ---- Confidence badge ----
    badge = _CONFIDENCE_LABEL.get(result["confidence"], result["confidence"])
    st.caption(badge)

    # ---- Cited sources panel ----
    cited_ids     = set(result["citations"])
    cited_chunks  = [c for c in chunks if c["chunk_id"] in cited_ids]

    if cited_chunks:
        with st.expander(f"Cited sources ({len(cited_chunks)})"):
            _render_chunks_list(cited_chunks, cited_ids=cited_ids, numbered=True)

    # ---- All retrieved chunks ----
    if chunks:
        label = f"All retrieved chunks ({len(chunks)})"
        with st.expander(label):
            _render_chunks_list(chunks, cited_ids=cited_ids, numbered=True, show_cited_marker=True)


def _render_chunks_list(
    chunks:            List[dict],
    cited_ids:         set,
    numbered:          bool = True,
    show_cited_marker: bool = False,
) -> None:
    """Render a list of chunk cards with location, score, and text preview."""
    for i, chunk in enumerate(chunks, 1):
        score      = chunk.get("score", 0.0)
        is_cited   = chunk["chunk_id"] in cited_ids
        loc        = _chunk_location(chunk)

        prefix = f"**[{i}]**" if numbered else "—"
        marker = " ✦ *cited*" if (show_cited_marker and is_cited) else ""

        st.markdown(f"{prefix}{marker} `{loc}` — score: `{score:.3f}`")
        preview = chunk["text"]
        if len(preview) > 400:
            preview = preview[:400] + "…"
        st.text(preview)

        if i < len(chunks):
            st.divider()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Corpus")

    corpora = list_corpora()
    if not corpora:
        st.error("No *.yaml files found in configs/")
        st.stop()

    # Reset conversation when corpus changes
    if "active_corpus" not in st.session_state:
        st.session_state.active_corpus = corpora[0]

    selected = st.selectbox("Select corpus", corpora,
                            index=corpora.index(st.session_state.active_corpus)
                            if st.session_state.active_corpus in corpora else 0)

    if selected != st.session_state.active_corpus:
        st.session_state.active_corpus  = selected
        st.session_state.messages       = []
        st.session_state.last_result    = None
        st.session_state.last_chunks    = []
        st.session_state.last_query     = ""
        st.session_state.feedback_done  = True

    try:
        cfg = load_config(f"configs/{selected}.yaml")
    except Exception as exc:
        st.error(f"Failed to load config: {exc}")
        st.stop()

    # ---- Top-k slider ----
    st.divider()
    k_slider = st.slider(
        "Top-k chunks",
        min_value=1,
        max_value=20,
        value=cfg.retrieve.top_k,
        help="Number of chunks retrieved per query. "
             "Higher = more context, more cost.",
    )

    # ---- Index status ----
    st.divider()
    st.header("Index status")

    index_exists  = cfg.index_path.exists()
    chunks_exist  = cfg.chunks_path.exists()
    docs_exist    = (cfg.processed_dir / "docs.jsonl").exists()

    st.write(f"{'✅' if docs_exist   else '❌'}  docs.jsonl")
    st.write(f"{'✅' if chunks_exist else '❌'}  chunks.jsonl")
    st.write(f"{'✅' if index_exists else '❌'}  faiss.index")

    if index_exists:
        mtime    = cfg.index_path.stat().st_mtime
        idx, _cm = _load_index_cached(
            str(cfg.index_path), str(cfg.meta_path), mtime
        )
        st.caption(f"{idx.ntotal:,} vectors · dim {idx.d}")
    else:
        st.warning(
            "Index not built yet. Run:\n\n"
            "```bash\npython -m src.embed_index --corpus "
            f"{cfg.corpus}\n```"
        )

    # ---- Config details ----
    st.divider()
    with st.expander("Config details"):
        st.json({
            "corpus":     cfg.corpus,
            "chunk_size": cfg.chunk.chunk_size,
            "top_k":      cfg.retrieve.top_k,
            "model":      cfg.answer.model,
            "embed":      cfg.embed.model,
        })


# ---------------------------------------------------------------------------
# Page title
# ---------------------------------------------------------------------------

st.title("RAG Framework")
st.caption(
    f"Corpus: **{cfg.corpus}** · "
    f"Model: **{cfg.answer.model}** · "
    f"k = **{k_slider}**"
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "messages":      [],
        "last_result":   None,
        "last_chunks":   [],
        "last_query":    "",
        "feedback_done": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

_init_state()


# ---------------------------------------------------------------------------
# Chat history (re-rendered on every Streamlit rerun)
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            stored_result = msg.get("result")
            stored_chunks = msg.get("chunks") or []
            if stored_result:
                _render_result(stored_result, stored_chunks)
            else:
                st.markdown(msg.get("content", ""))


# ---------------------------------------------------------------------------
# Chat input + query handler
# ---------------------------------------------------------------------------

query = st.chat_input(
    "Ask a question about the corpus…",
    disabled=not index_exists,
)

if query:
    # User message
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Assistant message
    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating…"):
            try:
                result, chunks = retrieve_and_answer(query, cfg, k=k_slider)
            except Exception as exc:
                st.error(f"Error: {exc}")
                st.stop()

        _render_result(result, chunks)

    # Update state
    st.session_state.messages.append({
        "role":    "assistant",
        "content": result["answer"],
        "result":  result,
        "chunks":  chunks,
    })
    st.session_state.last_result   = result
    st.session_state.last_chunks   = chunks
    st.session_state.last_query    = query
    st.session_state.feedback_done = False


# ---------------------------------------------------------------------------
# Feedback widget  (shown below last answer until submitted)
# ---------------------------------------------------------------------------

if (
    st.session_state.last_result is not None
    and not st.session_state.feedback_done
    and index_exists
):
    st.divider()
    st.subheader("Rate this answer")

    col_left, col_right = st.columns(2)

    with col_left:
        correctness = st.radio(
            "Correctness",
            options=["✅ Correct", "🤷 Partial", "❌ Wrong"],
            horizontal=True,
            index=0,
        )

    with col_right:
        usefulness_opt = st.radio(
            "Usefulness",
            options=["👍 Useful", "👎 Not useful"],
            horizontal=True,
            index=0,
        )

    comment = st.text_area("Optional comment", placeholder="What was wrong or missing?", height=80)

    if st.button("Submit feedback", type="primary"):
        correctness_map = {
            "✅ Correct": "correct",
            "🤷 Partial": "partial",
            "❌ Wrong":   "wrong",
        }
        save_feedback(
            query       = st.session_state.last_query,
            result      = st.session_state.last_result,
            chunks      = st.session_state.last_chunks,
            correctness = correctness_map.get(correctness, "partial"),
            usefulness  = usefulness_opt.startswith("👍"),
            comment     = comment.strip(),
            cfg         = cfg,
        )
        st.session_state.feedback_done = True
        st.toast("Feedback saved — thank you!", icon="✅")
        st.rerun()
