"""
ingest.py – walk data/raw/<corpus>/ and produce normalized Document objects.

Document schema
---------------
{
    "doc_id":   "sha256hex",           # sha256(corpus:relative_source)
    "title":    "Inferred Title",
    "source":   "data/raw/execution/foo.pdf",   # relative to project root
    "text":     "normalized full text",
    "metadata": {
        "corpus":     "execution",
        "created_at": "2025-02-21T12:00:00",
        "num_chars":  12345,
        "num_pages":  4,               # int for PDFs, null for others
        "file_type":  ".pdf"
    }
}

Pipeline
--------
    docs = load_documents(cfg)       # walk raw dir → list[dict]
    save_docs(docs, cfg)             # write docs.jsonl
    docs = load_docs(cfg)            # read docs.jsonl back
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


from src.config import Config


# ---------------------------------------------------------------------------
# doc_id
# ---------------------------------------------------------------------------

def _make_doc_id(corpus: str, source: str) -> str:
    """Stable SHA-256 ID derived from corpus name + relative source path."""
    return hashlib.sha256(f"{corpus}:{source}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """
    Clean raw extracted text:
      1. Unicode NFC normalisation + common ligature/quote fixes
      2. Collapse horizontal whitespace (tabs, multiple spaces → one space)
      3. Collapse 3+ consecutive newlines → 2 (preserve paragraph breaks)
      4. Strip trailing whitespace from every line
    """
    # 1. Unicode
    text = unicodedata.normalize("NFC", text)
    # Common PDF artefacts
    text = (
        text.replace("\u2019", "'").replace("\u2018", "'")
            .replace("\u201c", '"').replace("\u201d", '"')
            .replace("\u2013", "-").replace("\u2014", "--")
            .replace("\u00a0", " ")          # non-breaking space
            .replace("\u0000", "")           # null bytes
    )

    # 2. Horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)

    # 3. Excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 4. Trailing spaces per line
    text = "\n".join(line.rstrip() for line in text.split("\n"))

    return text.strip()


def _strip_repeated_lines(pages: List[str]) -> str:
    """
    Heuristically remove headers and footers from PDF text.

    Any short line (< 120 chars) that appears verbatim in more than 60 % of
    pages is considered a repeated header/footer and stripped from every page.
    Returns the cleaned full text.
    """
    if len(pages) < 3:
        return "\n".join(pages)

    threshold = max(2, len(pages) * 0.6)

    # Count how many distinct pages each short line appears on
    line_counts: Counter = Counter()
    for page in pages:
        page_lines = {ln.strip() for ln in page.split("\n") if ln.strip()}
        for ln in page_lines:
            if len(ln) < 120:
                line_counts[ln] += 1

    repeated = {ln for ln, cnt in line_counts.items() if cnt >= threshold}

    cleaned_pages = []
    for page in pages:
        filtered = [
            ln for ln in page.split("\n")
            if ln.strip() not in repeated
        ]
        cleaned_pages.append("\n".join(filtered))

    return "\n\n".join(cleaned_pages)


# ---------------------------------------------------------------------------
# Title inference
# ---------------------------------------------------------------------------

def _infer_title(path: Path, text: str, file_type: str) -> str:
    """
    Attempt to extract a human-readable title from document content.
    Fall back to the filename stem formatted as title case.
    """
    if file_type in (".md", ".txt"):
        # First markdown heading
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                return re.sub(r"^#+\s*", "", line).strip() or _stem_title(path)

    elif file_type in (".html", ".htm"):
        # First non-empty line tends to be the page title after BS extraction
        for line in text.split("\n"):
            line = line.strip()
            if line and len(line) < 120:
                return line

    elif file_type == ".pdf":
        # First short, non-sentence-ending line in the first 15 lines
        for line in text.split("\n")[:15]:
            line = line.strip()
            if 4 < len(line) < 100 and not line.endswith("."):
                return line

    return _stem_title(path)


def _stem_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").title()


# ---------------------------------------------------------------------------
# File readers  →  (text, num_pages)
# ---------------------------------------------------------------------------

def _read_pdf(path: Path) -> Tuple[str, int]:
    try:
        import pypdf
    except ImportError:
        raise ImportError("Install pypdf: pip install pypdf")

    reader = pypdf.PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    text = _strip_repeated_lines(pages)
    return text, len(pages)


def _read_html(path: Path) -> Tuple[str, None]:
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    soup = BeautifulSoup(path.read_bytes(), "html.parser")
    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()
    return soup.get_text(separator="\n"), None


def _read_text(path: Path) -> Tuple[str, None]:
    return path.read_text(encoding="utf-8", errors="replace"), None


_READERS = {
    ".pdf":  _read_pdf,
    ".txt":  _read_text,
    ".md":   _read_text,
    ".htm":  _read_html,
    ".html": _read_html,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_documents(cfg: Config) -> List[dict]:
    """
    Walk cfg.raw_dir, parse every allowed file, and return a list of
    normalized document dicts.

    Logs every skipped / failed file explicitly — never silently drops one.
    """
    raw_dir = cfg.raw_dir
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"Raw data directory not found: {raw_dir}\n"
            f"Add documents to data/raw/{cfg.corpus}/ before ingesting."
        )

    allowed = set(cfg.ingest.file_types)
    docs: List[dict] = []
    skipped = 0

    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext not in allowed:
            print(f"[ingest] SKIP {path.name}  (extension {ext!r} not in allowed list)")
            skipped += 1
            continue

        reader = _READERS.get(ext)
        if reader is None:
            print(f"[ingest] SKIP {path.name}  (no reader for {ext!r})")
            skipped += 1
            continue

        try:
            raw_text, num_pages = reader(path)
        except Exception as exc:
            print(f"[ingest] ERROR {path.name}: {exc}")
            skipped += 1
            continue

        text = _normalize_text(raw_text)
        if not text:
            print(f"[ingest] SKIP {path.name}  (empty after normalisation)")
            skipped += 1
            continue

        # Use path relative to project root for stable, portable source strings
        try:
            source = str(path.relative_to(Path(".")))
        except ValueError:
            source = str(path)

        doc_id = _make_doc_id(cfg.corpus, source)
        title = _infer_title(path, text, ext)

        doc = {
            "doc_id": doc_id,
            "title": title,
            "source": source,
            "text": text,
            "metadata": {
                "corpus": cfg.corpus,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "num_chars": len(text),
                "num_pages": num_pages,
                "file_type": ext,
            },
        }
        docs.append(doc)
        pages_str = f", {num_pages} pages" if num_pages else ""
        print(f"[ingest] OK   {path.name}  ({len(text):,} chars{pages_str})  id={doc_id[:8]}…")

    print(f"\n[ingest] loaded {len(docs)} documents, skipped {skipped}")
    return docs


def save_docs(docs: List[dict], cfg: Config) -> None:
    """Write docs to data/processed/<corpus>/docs.jsonl."""
    out = cfg.processed_dir / "docs.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"[ingest] saved {len(docs)} docs → {out}")


def load_docs(cfg: Config) -> List[dict]:
    """Read docs.jsonl from disk."""
    path = cfg.processed_dir / "docs.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"docs.jsonl not found at {path}. Run load_documents + save_docs first."
        )
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
