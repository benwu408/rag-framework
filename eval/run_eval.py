"""
run_eval.py – offline evaluation of the RAG pipeline.

Metrics computed per query:
  hit_rate    – 1 if any relevant chunk appears in top-k results
  mrr         – mean reciprocal rank of the first relevant chunk
  faithfulness – simple lexical overlap between answer and context (proxy)

Usage:
    python eval/run_eval.py --config configs/execution.yaml
    python eval/run_eval.py --config configs/sec.yaml --output eval/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Optional

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.retrieve import retrieve
from src.rag_answer import answer


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def hit_rate(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    if not relevant_ids:
        return 1.0   # no ground truth → assume pass
    return float(bool(set(retrieved_ids) & set(relevant_ids)))


def mrr(retrieved_ids: List[int], relevant_ids: List[int]) -> float:
    if not relevant_ids:
        return 1.0
    for rank, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def faithfulness_proxy(answer_text: str, context: str) -> float:
    """
    Rough lexical faithfulness: fraction of answer tokens found in context.
    A proper faithfulness metric requires an LLM judge.
    """
    ans_tokens = set(answer_text.lower().split())
    ctx_tokens = set(context.lower().split())
    if not ans_tokens:
        return 0.0
    overlap = ans_tokens & ctx_tokens
    return len(overlap) / len(ans_tokens)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_eval(cfg: Config, queries_path: Path, output_path: Optional[Path]) -> None:
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")

    rows = []
    with open(queries_path, newline="") as f:
        reader = csv.DictReader(f)
        queries = list(reader)

    print(f"[eval] evaluating {len(queries)} queries against corpus '{cfg.corpus}'")

    all_hit, all_mrr, all_faith = [], [], []

    for row in queries:
        qid = row.get("query_id", "?")
        query = row.get("query", "").strip()
        if not query:
            continue

        # Parse ground-truth chunk ids (optional)
        raw_ids = row.get("relevant_chunk_ids", "")
        relevant_ids = [int(x) for x in raw_ids.split(",") if x.strip().isdigit()]

        # Retrieve
        chunks = retrieve(query, cfg)
        retrieved_ids = [c["chunk_id"] for c in chunks]
        context = " ".join(c["text"] for c in chunks)

        # Answer
        try:
            ans_text = answer(query, cfg)
        except Exception as e:
            ans_text = f"ERROR: {e}"

        # Metrics
        hr = hit_rate(retrieved_ids, relevant_ids)
        rr = mrr(retrieved_ids, relevant_ids)
        faith = faithfulness_proxy(ans_text, context)

        all_hit.append(hr)
        all_mrr.append(rr)
        all_faith.append(faith)

        result = {
            "query_id": qid,
            "query": query,
            "hit_rate": round(hr, 4),
            "mrr": round(rr, 4),
            "faithfulness": round(faith, 4),
            "answer": ans_text[:200],
            "retrieved_chunk_ids": retrieved_ids,
        }
        rows.append(result)
        print(f"  [{qid}] hit={hr:.2f} mrr={rr:.2f} faith={faith:.2f} | {query[:60]}")

    # Aggregate
    n = len(rows)
    print("\n--- Aggregate Metrics ---")
    print(f"  Queries evaluated : {n}")
    print(f"  Hit Rate (avg)    : {sum(all_hit)/n:.4f}" if n else "")
    print(f"  MRR (avg)         : {sum(all_mrr)/n:.4f}" if n else "")
    print(f"  Faithfulness (avg): {sum(all_faith)/n:.4f}" if n else "")

    # Save
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="") as f:
            fieldnames = ["query_id", "query", "hit_rate", "mrr", "faithfulness", "answer", "retrieved_chunk_ids"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                r["retrieved_chunk_ids"] = json.dumps(r["retrieved_chunk_ids"])
                writer.writerow(r)
        print(f"\n[eval] results saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation.")
    parser.add_argument("--config", required=True, help="Path to corpus config YAML")
    parser.add_argument("--queries", default=None, help="Override queries CSV path")
    parser.add_argument("--output", default=None, help="Save results to this CSV path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    queries_path = Path(args.queries) if args.queries else Path(cfg.eval.queries_file)
    output_path = Path(args.output) if args.output else None

    run_eval(cfg, queries_path, output_path)


if __name__ == "__main__":
    main()
