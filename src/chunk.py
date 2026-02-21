"""
chunk.py – split normalized Documents into overlapping chunks and write chunks.jsonl.

Chunk schema
------------
{
    "chunk_id":  "a3f9c2...::000042",   # doc_id + "::" + zero-padded within-doc index
    "doc_id":    "a3f9c2...",           # sha256 from ingest
    "text":      "...",
    "metadata": {
        "title":       "Order Types",        # from doc
        "section":     "IOC Orders",         # nearest heading before this chunk
        "source":      "data/raw/.../f.pdf", # from doc
        "corpus":      "execution",
        "page":        4,                    # approx (PDFs); null otherwise
        "char_start":  1200,                 # byte offset in doc.text
        "char_end":    4700,
        "chunk_index": 42                    # within-doc ordinal (0-based)
    }
}

Splitting strategies
--------------------
  recursive  – langchain RecursiveCharacterTextSplitter (best for mixed content)
  sentence   – merge sentences up to chunk_size, keep overlap sentences
  fixed      – hard character boundaries with overlap stride
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from src.config import Config


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

# Matches markdown headings, ALL-CAPS lines, or short title-case-ish lines.
_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)")

# An ALL-CAPS line: 4–100 chars, at least half of non-space chars are uppercase.
_ALLCAPS_RE = re.compile(r"^[A-Z0-9][A-Z0-9 ,.\-/&]{3,99}$")


def _classify_heading(line: str) -> Optional[str]:
    """
    Return the section name if `line` looks like a heading, else None.
    Three patterns accepted:
      1. Markdown:  ## Section Title
      2. ALL CAPS:  SECTION TITLE (≥ 4 chars, only uppercase letters/digits/punctuation)
      3. Title case: short line, starts with capital, no sentence-ending punct,
                     not dense with lowercase (heuristic to avoid prose sentences)
    """
    line = line.strip()
    if not line or len(line) > 100:
        return None

    # 1. Markdown heading
    m = _MD_HEADING_RE.match(line)
    if m:
        return m.group(1).strip()

    # 2. ALL CAPS
    if _ALLCAPS_RE.match(line) and not line[-1] in ".,:;?!":
        return line.title()

    # 3. Title-case heuristic: starts with capital, no sentence punctuation,
    #    fewer than 4 lowercase words (so it's not a prose sentence).
    if (
        4 < len(line) < 80
        and line[0].isupper()
        and line[-1] not in ".,:;?!"
        and len(re.findall(r"\b[a-z]{4,}\b", line)) < 4
    ):
        return line

    return None


def _build_section_index(text: str) -> List[Tuple[int, str]]:
    """
    Walk `text` line by line and return [(char_offset, section_name), ...].
    `char_offset` is the byte position of the start of the heading line.
    """
    sections: List[Tuple[int, str]] = []
    offset = 0
    for line in text.split("\n"):
        name = _classify_heading(line)
        if name:
            sections.append((offset, name))
        offset += len(line) + 1  # +1 for the consumed '\n'
    return sections


def _section_at(char_start: int, section_index: List[Tuple[int, str]]) -> Optional[str]:
    """Return the last section heading whose offset is ≤ char_start."""
    result: Optional[str] = None
    for offset, name in section_index:
        if offset <= char_start:
            result = name
        else:
            break
    return result


# ---------------------------------------------------------------------------
# Char-offset finder
# ---------------------------------------------------------------------------

def _find_offsets(full_text: str, chunk_texts: List[str]) -> List[Tuple[int, int]]:
    """
    Locate each chunk text inside full_text by searching forward.

    Returns [(char_start, char_end), ...] parallel to chunk_texts.
    If a chunk cannot be found (e.g. langchain whitespace-normalised it),
    falls back to searching the stripped version, then to a positional estimate.
    """
    offsets: List[Tuple[int, int]] = []
    search_from = 0

    for chunk_text in chunk_texts:
        idx = full_text.find(chunk_text, search_from)

        if idx == -1:
            # Langchain may have trimmed leading/trailing whitespace
            stripped = chunk_text.strip()
            idx = full_text.find(stripped, search_from)
            if idx != -1:
                chunk_text = stripped

        if idx == -1:
            # Positional fallback – not exact but keeps the list well-formed
            start = search_from
        else:
            start = idx

        end = start + len(chunk_text)
        offsets.append((start, end))
        search_from = start + 1  # advance past this chunk's start (handles overlap)

    return offsets


# ---------------------------------------------------------------------------
# Page approximation
# ---------------------------------------------------------------------------

def _approx_page(char_start: int, total_chars: int, num_pages: Optional[int]) -> Optional[int]:
    """Estimate which page a character offset falls on (1-based). PDFs only."""
    if not num_pages or total_chars == 0:
        return None
    ratio = char_start / total_chars
    return max(1, min(num_pages, int(ratio * num_pages) + 1))


# ---------------------------------------------------------------------------
# Splitters  (each returns List[str])
# ---------------------------------------------------------------------------

def _split_recursive(text: str, size: int, overlap: int) -> List[str]:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        raise ImportError(
            "Install langchain-text-splitters: pip install langchain-text-splitters"
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def _split_sentence(text: str, size: int, overlap: int) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    for sent in sentences:
        slen = len(sent)
        if buf and buf_len + slen + 1 > size:
            chunks.append(" ".join(buf))
            # Drop sentences from the front until we're within the overlap budget
            while buf and buf_len > overlap:
                removed = buf.pop(0)
                buf_len -= len(removed) + 1
        buf.append(sent)
        buf_len += slen + 1

    if buf:
        chunks.append(" ".join(buf))
    return chunks


def _split_fixed(text: str, size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + size])
        start += size - overlap
    return chunks


_STRATEGIES = {
    "recursive": _split_recursive,
    "sentence":  _split_sentence,
    "fixed":     _split_fixed,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chunk_documents(docs: List[dict], cfg: Config) -> List[dict]:
    """
    Split each Document dict (from ingest.load_documents) into chunks.

    Returns a flat list of chunk dicts with full metadata including
    section, page, char_start, char_end.
    """
    strategy = cfg.chunk.strategy
    splitter_fn = _STRATEGIES.get(strategy)
    if splitter_fn is None:
        raise ValueError(
            f"Unknown chunk strategy: {strategy!r}. "
            f"Choose from {list(_STRATEGIES)}"
        )

    size    = cfg.chunk.chunk_size
    overlap = cfg.chunk.chunk_overlap

    all_chunks: List[dict] = []

    for doc in docs:
        doc_id     = doc["doc_id"]
        text       = doc["text"]
        title      = doc.get("title", "")
        source     = doc.get("source", "")
        doc_meta   = doc.get("metadata", {})
        num_pages  = doc_meta.get("num_pages")
        total_chars = len(text)

        # Build section index once per document
        section_index = _build_section_index(text)

        # Split into raw text pieces
        pieces = splitter_fn(text, size, overlap)

        # Locate each piece in the original text for char offsets
        offsets = _find_offsets(text, pieces)

        for i, (piece, (char_start, char_end)) in enumerate(zip(pieces, offsets)):
            piece = piece.strip()
            if not piece:
                continue

            section = _section_at(char_start, section_index)
            page    = _approx_page(char_start, total_chars, num_pages)

            all_chunks.append({
                "chunk_id": f"{doc_id}::{i:06d}",
                "doc_id":   doc_id,
                "text":     piece,
                "metadata": {
                    "title":       title,
                    "section":     section,
                    "source":      source,
                    "corpus":      doc_meta.get("corpus", ""),
                    "page":        page,
                    "char_start":  char_start,
                    "char_end":    char_end,
                    "chunk_index": i,
                },
            })

        print(
            f"[chunk] {doc.get('metadata', {}).get('file_type', '')} "
            f"'{title[:40]}' → {len(pieces)} chunks"
        )

    print(f"\n[chunk] total: {len(all_chunks)} chunks from {len(docs)} documents")
    return all_chunks


def save_chunks(chunks: List[dict], cfg: Config) -> None:
    """Write chunks to data/processed/<corpus>/chunks.jsonl."""
    out = cfg.chunks_path
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk) + "\n")
    print(f"[chunk] saved {len(chunks)} chunks → {out}")


def load_chunks(cfg: Config) -> List[dict]:
    """Read chunks.jsonl from disk."""
    path = cfg.chunks_path
    if not path.exists():
        raise FileNotFoundError(
            f"chunks.jsonl not found at {path}. Run ingest + chunk first."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
