"""
retrieve.py – embed queries and retrieve top-k chunks from the FAISS index.

Public API
----------
  retrieve(query, cfg, k, filter_doc_id, filter_section_contains)
      → List[dict]          single query

  retrieve_batch(queries, cfg, k)
      → List[List[dict]]    all queries in one OpenAI call + one FAISS call

  format_context(chunks)
      → str                 context block for the LLM prompt

Retrieved chunk dicts are the raw chunk dicts from chunk_meta.json with one
extra field added: "score" (cosine similarity, float 0–1).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from src.config import Config
from src.embed_index import load_index, embed_texts


# ---------------------------------------------------------------------------
# Core search — operates on pre-loaded index + pre-embedded query vectors
# ---------------------------------------------------------------------------

def _search(
    index,
    chunks:   List[dict],
    q_vecs:   np.ndarray,       # (N, D) L2-normalised
    k:        int,
    threshold: float,
    filter_doc_id:           Optional[str],
    filter_section_contains: Optional[str],
) -> List[List[dict]]:
    """
    Run FAISS search for N queries, apply post-retrieval filters, return results.

    When filters are active, over-fetches (k × 4) from FAISS so there is
    enough headroom to still return k results after filtering.

    Returns List[List[dict]] — one result list per query, each sorted by
    descending score.
    """
    has_filter = bool(filter_doc_id or filter_section_contains)
    fetch_k    = min(k * 4 if has_filter else k, index.ntotal)

    scores_batch, indices_batch = index.search(q_vecs, fetch_k)

    all_results: List[List[dict]] = []

    for scores, indices in zip(scores_batch, indices_batch):
        results: List[dict] = []

        for score, idx in zip(scores, indices):
            if idx == -1:                        # FAISS sentinel for "no result"
                continue
            if float(score) < threshold:
                continue

            chunk = dict(chunks[idx])            # shallow copy so we can mutate
            chunk["score"] = float(score)

            # ---- post-retrieval filters ----
            if filter_doc_id and chunk.get("doc_id") != filter_doc_id:
                continue

            if filter_section_contains:
                section = (chunk.get("metadata") or {}).get("section") or ""
                if filter_section_contains.lower() not in section.lower():
                    continue
            # --------------------------------

            results.append(chunk)
            if len(results) >= k:
                break

        all_results.append(results)

    return all_results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve(
    query:                   str,
    cfg:                     Config,
    k:                       Optional[int] = None,
    filter_doc_id:           Optional[str] = None,
    filter_section_contains: Optional[str] = None,
) -> List[dict]:
    """
    Embed a single query and return the top-k most similar chunks.

    Args:
        query:                   Natural-language question.
        cfg:                     Loaded Config object.
        k:                       Number of results. Defaults to cfg.retrieve.top_k.
        filter_doc_id:           Only return chunks from this doc_id.
        filter_section_contains: Only return chunks whose section contains this string.

    Returns:
        List of chunk dicts (with added "score" field), sorted by descending score.
    """
    index, chunks = load_index(cfg)
    k_eff  = min(k or cfg.retrieve.top_k, index.ntotal)
    q_vec  = embed_texts([query], cfg)   # (1, D)
    return _search(
        index, chunks, q_vec, k_eff,
        cfg.retrieve.score_threshold,
        filter_doc_id,
        filter_section_contains,
    )[0]


def retrieve_batch(
    queries: List[str],
    cfg:     Config,
    k:       Optional[int] = None,
) -> List[List[dict]]:
    """
    Embed all queries in a single OpenAI API call, then search FAISS once.

    This is the function the eval harness uses to avoid N separate API calls
    and N separate disk reads of the FAISS index.

    Args:
        queries: List of question strings.
        cfg:     Loaded Config object.
        k:       Results per query. Defaults to cfg.retrieve.top_k.

    Returns:
        List[List[dict]] — one result list per query, parallel to `queries`.
    """
    if not queries:
        return []

    index, chunks = load_index(cfg)
    k_eff  = min(k or cfg.retrieve.top_k, index.ntotal)
    q_vecs = embed_texts(queries, cfg)   # (N, D) — one API call for all queries
    return _search(
        index, chunks, q_vecs, k_eff,
        cfg.retrieve.score_threshold,
        None, None,
    )


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context(chunks: List[dict]) -> str:
    """
    Build the context block that gets inserted into the LLM prompt.

    Each chunk is prefixed with its chunk_id in brackets — this is the exact
    string the LLM should use for inline citations, e.g. [abc123::000004].

    Location string format:  Title > Section, p.N
    """
    parts = []
    for c in chunks:
        meta    = c.get("metadata") or {}
        title   = meta.get("title")   or "unknown"
        section = meta.get("section")
        page    = meta.get("page")
        score   = c.get("score", 0.0)

        location = title
        if section:
            location += f" > {section}"
        if page:
            location += f", p.{page}"

        header = f"[{c['chunk_id']}] {location} (score: {score:.3f})"
        parts.append(f"{header}\n{c['text']}")

    return "\n\n---\n\n".join(parts)
