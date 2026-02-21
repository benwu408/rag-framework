"""
build_pipeline.py – run the full ingestion pipeline for a corpus in one command.

Steps (run in order unless skipped):
  1. Ingest  – parse raw documents        → data/processed/<corpus>/docs.jsonl
  2. Chunk   – split docs into chunks     → data/processed/<corpus>/chunks.jsonl
  3. Embed   – embed + build FAISS index  → data/processed/<corpus>/faiss.index
                                             data/processed/<corpus>/chunk_meta.json

Usage:
  python scripts/build_pipeline.py --corpus execution
  python scripts/build_pipeline.py --config configs/sec.yaml

Partial reruns (skip completed steps):
  python scripts/build_pipeline.py --corpus execution --skip-ingest
  python scripts/build_pipeline.py --corpus execution --skip-ingest --skip-chunk
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config


# ---------------------------------------------------------------------------
# Step runners
# ---------------------------------------------------------------------------

def run_ingest(cfg: Config) -> list:
    from src.ingest import load_documents, save_docs
    print("\n" + "─" * 60)
    print("STEP 1 — INGEST")
    print("─" * 60)
    docs = load_documents(cfg)
    save_docs(docs, cfg)
    return docs


def run_chunk(cfg: Config, docs: list | None = None) -> list:
    from src.ingest import load_docs
    from src.chunk import chunk_documents, save_chunks
    print("\n" + "─" * 60)
    print("STEP 2 — CHUNK")
    print("─" * 60)
    if docs is None:
        docs = load_docs(cfg)
    chunks = chunk_documents(docs, cfg)
    save_chunks(chunks, cfg)
    return chunks


def run_embed(cfg: Config) -> None:
    from src.embed_index import run_embed_index
    print("\n" + "─" * 60)
    print("STEP 3 — EMBED + INDEX")
    print("─" * 60)
    run_embed_index(cfg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full RAG pipeline for a corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", help="Corpus name — loads configs/<corpus>.yaml")
    group.add_argument("--config", help="Path to a config YAML file")

    parser.add_argument("--skip-ingest", action="store_true",
                        help="Load docs.jsonl from disk instead of re-parsing raw files")
    parser.add_argument("--skip-chunk",  action="store_true",
                        help="Load chunks.jsonl from disk instead of re-chunking")
    parser.add_argument("--skip-embed",  action="store_true",
                        help="Skip embedding (index already built)")

    args = parser.parse_args()

    cfg = (
        load_config(f"configs/{args.corpus}.yaml")
        if args.corpus
        else load_config(args.config)
    )

    print(f"\n{'═' * 60}")
    print(f"  RAG PIPELINE  —  corpus: {cfg.corpus}")
    print(f"{'═' * 60}")
    print(f"  Raw dir      : {cfg.raw_dir}")
    print(f"  Processed dir: {cfg.processed_dir}")
    print(f"  Chunk size   : {cfg.chunk.chunk_size} chars  (overlap {cfg.chunk.chunk_overlap})")
    print(f"  Embed model  : {cfg.embed.model}")

    t0 = time.time()
    docs = None

    if not args.skip_ingest:
        docs = run_ingest(cfg)
    else:
        print("\nSTEP 1 — INGEST  [skipped]")

    if not args.skip_chunk:
        run_chunk(cfg, docs)
    else:
        print("STEP 2 — CHUNK   [skipped]")

    if not args.skip_embed:
        run_embed(cfg)
    else:
        print("STEP 3 — EMBED   [skipped]")

    elapsed = time.time() - t0
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Index : {cfg.index_path}")
    print(f"  Chunks: {cfg.chunks_path}")
    print(f"{'═' * 60}\n")


if __name__ == "__main__":
    main()
