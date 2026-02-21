"""
rag_answer.py – grounded answer generation with inline citations via GPT-4o mini.

Public API
----------
  rag_answer(query, chunks, cfg)  →  AnswerResult
      Pure generation step. Takes pre-retrieved chunks, returns structured result.

  answer(query, cfg, k)           →  (AnswerResult, List[dict])
      Convenience wrapper: retrieve then answer.

  save_feedback(...)              →  None
      Append a feedback row to data/processed/<corpus>/feedback.csv.

AnswerResult schema
-------------------
  answer           – answer text with inline [ref_N] citations
  citations        – list of chunk_ids cited (resolved from ref_N)
  confidence       – "high" | "medium" | "low"
  supported_claims – [{"claim": str, "citation": chunk_id}, ...]
  cannot_answer    – True if context was insufficient
  raw_response     – raw LLM text (for debugging)
"""

from __future__ import annotations

import csv
import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, TypedDict

from openai import OpenAI

from src.config import Config
from src.retrieve import retrieve


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

class SupportedClaim(TypedDict):
    claim:    str
    citation: str   # full chunk_id (resolved from ref_N)


class AnswerResult(TypedDict):
    answer:           str
    citations:        List[str]            # full chunk_ids
    confidence:       str                  # "high" | "medium" | "low"
    supported_claims: List[SupportedClaim]
    cannot_answer:    bool
    raw_response:     str                  # raw LLM output, kept for debugging


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

# Fixed citation + JSON rules appended to every system prompt.
# The user-facing persona lives in cfg.answer.system_prompt (YAML).
_CITATION_RULES = """
When answering:
1. Use ONLY information present in the provided context passages.
2. Cite every factual claim by placing the passage reference ID immediately \
after the claim in brackets, e.g. "VWAP is computed as price × volume [ref_1]."
3. Use the EXACT reference IDs shown in the context headers — do not invent IDs.
4. A single claim may carry multiple citations: [ref_1][ref_2].
5. If the context does not contain sufficient information to answer, set \
"cannot_answer" to true and leave "answer" as an empty string.
6. Rate your confidence: "high" if the context directly and fully addresses \
the question, "medium" if partially, "low" if only weakly.

Return ONLY valid JSON — no markdown fences, no prose before or after the object. \
The response must start with { and end with }. Use this exact schema:
{
  "answer": "...",
  "citations": ["ref_1", "ref_2"],
  "confidence": "high",
  "supported_claims": [
    {"claim": "exact phrase from answer", "citation": "ref_1"}
  ],
  "cannot_answer": false
}"""

# Injected as an extra user turn on the retry to force stricter compliance.
_STRICT_RETRY_MSG = (
    "Your previous response was not valid JSON. "
    "Return ONLY the JSON object — no markdown, no explanation. "
    "Start your response with { and end with }."
)


def _build_numbered_context(chunks: List[dict]) -> tuple[str, Dict[str, str]]:
    """
    Assign short reference IDs (ref_1, ref_2, …) to each chunk and build the
    context string that goes into the prompt.

    Returns:
        context_str – formatted context block
        ref_map     – {"ref_1": chunk_id, "ref_2": chunk_id, …}

    Using short numeric references instead of full 70-char chunk_ids makes the
    LLM's citation task reliable and prevents hallucinated IDs.
    """
    ref_map: Dict[str, str] = {}
    parts: List[str] = []

    for i, c in enumerate(chunks, 1):
        ref  = f"ref_{i}"
        ref_map[ref] = c["chunk_id"]

        meta     = c.get("metadata") or {}
        title    = meta.get("title")   or "unknown"
        section  = meta.get("section")
        page     = meta.get("page")
        score    = c.get("score", 0.0)

        location = title
        if section:
            location += f" > {section}"
        if page:
            location += f", p.{page}"

        header = f"[{ref}] {location} (score: {score:.3f})"
        parts.append(f"{header}\n{c['text']}")

    return "\n\n---\n\n".join(parts), ref_map


# ---------------------------------------------------------------------------
# OpenAI API call
# ---------------------------------------------------------------------------

def _get_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def _call_openai(
    client:   OpenAI,
    messages: list,
    cfg:      Config,
) -> str:
    system = cfg.answer.system_prompt.rstrip() + "\n" + _CITATION_RULES
    full_messages = [{"role": "system", "content": system}] + messages
    resp = client.chat.completions.create(
        model       = cfg.answer.model,
        max_tokens  = cfg.answer.max_tokens,
        temperature = cfg.answer.temperature,
        messages    = full_messages,
    )
    return resp.choices[0].message.content


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str, ref_map: Dict[str, str]) -> Optional[AnswerResult]:
    """
    Parse the raw LLM text into an AnswerResult.

    Handles two common failure modes:
      - Markdown code fences wrapping the JSON
      - Leading/trailing prose around the JSON object

    Resolves ref_N references → full chunk_ids.
    Returns None if the text cannot be parsed as valid JSON.
    """
    text = raw.strip()

    # Strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        # Find the outermost { ... } block
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    def resolve(ref: str) -> str:
        return ref_map.get(ref, ref)   # unknown refs passed through unchanged

    citations = [resolve(r) for r in data.get("citations", []) if isinstance(r, str)]

    supported_claims: List[SupportedClaim] = []
    for sc in data.get("supported_claims", []):
        if isinstance(sc, dict):
            supported_claims.append({
                "claim":    str(sc.get("claim",    "")),
                "citation": resolve(str(sc.get("citation", ""))),
            })

    confidence = data.get("confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return AnswerResult(
        answer           = str(data.get("answer", "")),
        citations        = citations,
        confidence       = confidence,
        supported_claims = supported_claims,
        cannot_answer    = bool(data.get("cannot_answer", False)),
        raw_response     = "",   # filled by caller
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Tracks parse failures across calls in the current process (informational).
_parse_failures = 0


def rag_answer(query: str, chunks: List[dict], cfg: Config) -> AnswerResult:
    """
    Generate a grounded, cited answer from pre-retrieved chunks.

    Prompt rules (appended to cfg.answer.system_prompt):
      - Answer only from provided context
      - Cite every claim as [ref_N]
      - Return structured JSON with answer/citations/confidence/supported_claims
      - Set cannot_answer=true if context is insufficient

    Retries once with a stricter prompt if the first response fails JSON parsing.
    On two consecutive failures, returns a fallback result with the raw text and
    cannot_answer=True so downstream code can handle it gracefully.

    Args:
        query:  The user question.
        chunks: Pre-retrieved chunk dicts (from retrieve.retrieve).
        cfg:    Loaded Config object.

    Returns:
        AnswerResult TypedDict.
    """
    global _parse_failures

    if not chunks:
        return AnswerResult(
            answer           = "",
            citations        = [],
            confidence       = "low",
            supported_claims = [],
            cannot_answer    = True,
            raw_response     = "",
        )

    context, ref_map = _build_numbered_context(chunks)
    user_message     = f"Context:\n\n{context}\n\nQuestion: {query}"
    messages         = [{"role": "user", "content": user_message}]

    client = _get_client()

    # ---- Attempt 1 ----
    raw    = _call_openai(client, messages, cfg)
    result = _parse_response(raw, ref_map)

    if result is not None:
        result["raw_response"] = raw
        return result

    # ---- Attempt 2: stricter retry ----
    _parse_failures += 1
    print(
        f"[rag_answer] JSON parse failed (attempt 1). "
        f"Retrying with strict prompt. (total failures: {_parse_failures})"
    )
    retry_messages = messages + [
        {"role": "assistant", "content": raw},
        {"role": "user",      "content": _STRICT_RETRY_MSG},
    ]
    raw2   = _call_openai(client, retry_messages, cfg)
    result = _parse_response(raw2, ref_map)

    if result is not None:
        result["raw_response"] = raw2
        return result

    # ---- Both attempts failed ----
    _parse_failures += 1
    print(
        f"[rag_answer] JSON parse failed on both attempts. "
        f"Returning fallback. (total failures: {_parse_failures})"
    )
    return AnswerResult(
        answer           = raw2,
        citations        = [],
        confidence       = "low",
        supported_claims = [],
        cannot_answer    = True,
        raw_response     = raw2,
    )


def answer(
    query: str,
    cfg:   Config,
    k:     Optional[int] = None,
) -> tuple[AnswerResult, List[dict]]:
    """
    Convenience wrapper: retrieve chunks then generate a grounded answer.

    Returns:
        (AnswerResult, chunks) — chunks are the retrieved context used for
        the answer, useful for display and feedback logging.
    """
    chunks = retrieve(query, cfg, k=k)
    result = rag_answer(query, chunks, cfg)
    return result, chunks


# ---------------------------------------------------------------------------
# Feedback logging
# ---------------------------------------------------------------------------

def save_feedback(
    query:       str,
    result:      AnswerResult,
    chunks:      List[dict],
    correctness: str,          # "correct" | "partial" | "wrong"
    usefulness:  bool,
    comment:     str,
    cfg:         Config,
) -> None:
    """
    Append one feedback row to data/processed/<corpus>/feedback.csv.

    Columns: timestamp, query, answer, correctness, usefulness, comment,
             top_chunk_ids (JSON), similarity_scores (JSON)
    """
    path         = cfg.feedback_path
    write_header = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)

    top_chunk_ids     = json.dumps([c["chunk_id"] for c in chunks])
    similarity_scores = json.dumps([round(c.get("score", 0.0), 4) for c in chunks])

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp", "query", "answer", "correctness",
                "usefulness", "comment", "top_chunk_ids", "similarity_scores",
            ])
        writer.writerow([
            datetime.now(timezone.utc).isoformat(),
            query,
            result["answer"],
            correctness,
            int(usefulness),
            comment,
            top_chunk_ids,
            similarity_scores,
        ])
