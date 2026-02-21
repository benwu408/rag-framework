"""
embed_index.py – embed chunks with OpenAI and build/save/load a FAISS index.

Artifacts written to data/processed/<corpus>/:
  faiss.index     – FAISS IndexFlatIP (L2-normalised vectors → cosine similarity)
  chunk_meta.json – list of chunk dicts, index position i corresponds to FAISS row i

Run as a script:
  python -m src.embed_index --corpus execution
  python -m src.embed_index --config configs/sec.yaml
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import List

import numpy as np

from src.config import Config, load_config
from src.chunk import load_chunks


# ---------------------------------------------------------------------------
# Exponential backoff constants
# ---------------------------------------------------------------------------

_MAX_RETRIES = 6
_BASE_DELAY  = 1.0   # seconds; doubles each retry
_JITTER      = 0.1   # ± 10 % of current delay


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Install openai: pip install openai")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    return OpenAI(api_key=api_key)


def _embed_batch(client, batch: List[str], model: str) -> List[List[float]]:
    """
    Call the OpenAI embeddings API for one batch.
    Retries up to _MAX_RETRIES times with exponential backoff + jitter on
    RateLimitError or transient APIError.
    """
    try:
        from openai import RateLimitError, APIError
    except ImportError:
        raise ImportError("Install openai: pip install openai")

    delay = _BASE_DELAY
    for attempt in range(_MAX_RETRIES):
        try:
            resp = client.embeddings.create(input=batch, model=model)
            return [d.embedding for d in resp.data]

        except RateLimitError:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = delay * (1 + random.uniform(-_JITTER, _JITTER))
            print(
                f"[embed] rate limit – waiting {wait:.1f}s "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(wait)
            delay *= 2

        except APIError as exc:
            if attempt == _MAX_RETRIES - 1:
                raise
            wait = delay * (1 + random.uniform(-_JITTER, _JITTER))
            print(
                f"[embed] API error ({exc}) – waiting {wait:.1f}s "
                f"(attempt {attempt + 1}/{_MAX_RETRIES})"
            )
            time.sleep(wait)
            delay *= 2

    # Unreachable, but satisfies type checkers
    raise RuntimeError("embed_batch: exceeded max retries")


def embed_texts(texts: List[str], cfg: Config) -> np.ndarray:
    """
    Embed all texts in batches of cfg.embed.batch_size.
    Returns an (N, D) float32 array of L2-normalised vectors.
    """
    if not texts:
        raise ValueError("embed_texts: received empty text list")

    client     = _get_openai_client()
    model      = cfg.embed.model
    batch_size = cfg.embed.batch_size

    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs  = _embed_batch(client, batch, model)
        all_vecs.extend(vecs)
        print(f"[embed] {min(i + batch_size, len(texts))}/{len(texts)} embedded")

    arr   = np.array(all_vecs, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr   = arr / np.where(norms == 0, 1.0, norms)   # L2-normalise → cosine via IP
    return arr


# ---------------------------------------------------------------------------
# FAISS index
# ---------------------------------------------------------------------------

def build_index(embeddings: np.ndarray) -> "faiss.Index":
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu")
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_index(index: "faiss.Index", chunks: List[dict], cfg: Config) -> None:
    """
    Persist the FAISS index and parallel chunk metadata to disk.
    chunk_meta.json[i] corresponds to FAISS row i.
    Prints file sizes after writing.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu")

    out_dir = cfg.processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(cfg.index_path))
    with open(cfg.meta_path, "w") as f:
        json.dump(chunks, f, indent=2)

    index_kb = cfg.index_path.stat().st_size / 1024
    meta_kb  = cfg.meta_path.stat().st_size  / 1024
    print(
        f"[embed_index] saved:"
        f"  faiss.index {index_kb:,.0f} KB"
        f"  chunk_meta.json {meta_kb:,.0f} KB"
        f"  ({(index_kb + meta_kb) / 1024:.2f} MB total)"
    )


def load_index(cfg: Config) -> tuple["faiss.Index", List[dict]]:
    """
    Load FAISS index and chunk metadata from disk.
    Returns (index, chunks) where chunks[i] matches FAISS row i.
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss: pip install faiss-cpu")
    if not cfg.index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {cfg.index_path}. Run embed_index first."
        )
    if not cfg.meta_path.exists():
        raise FileNotFoundError(
            f"Metadata not found: {cfg.meta_path}. Run embed_index first."
        )

    index  = faiss.read_index(str(cfg.index_path))
    with open(cfg.meta_path) as f:
        chunks = json.load(f)
    return index, chunks


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_embed_index(cfg: Config) -> None:
    """Load chunks → embed → build index → save. Prints a summary on completion."""
    chunks = load_chunks(cfg)
    if not chunks:
        raise ValueError(f"No chunks found for corpus '{cfg.corpus}'. Run ingest + chunk first.")

    # Count distinct source documents
    doc_ids = {c["doc_id"] for c in chunks}

    texts = [c["text"] for c in chunks]
    print(
        f"[embed_index] corpus   : {cfg.corpus}\n"
        f"[embed_index] documents: {len(doc_ids)}\n"
        f"[embed_index] chunks   : {len(chunks)}\n"
        f"[embed_index] model    : {cfg.embed.model}"
    )

    embeddings = embed_texts(texts, cfg)
    index      = build_index(embeddings)
    save_index(index, chunks, cfg)

    print(f"[embed_index] done – index contains {index.ntotal} vectors (dim={embeddings.shape[1]})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embed chunks and build FAISS index.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--corpus",
        help="Corpus name – loads configs/<corpus>.yaml",
    )
    group.add_argument(
        "--config",
        help="Path to a config YAML file",
    )
    args = parser.parse_args()

    if args.corpus:
        cfg = load_config(f"configs/{args.corpus}.yaml")
    else:
        cfg = load_config(args.config)

    run_embed_index(cfg)
