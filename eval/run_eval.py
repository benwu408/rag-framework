"""
run_eval.py – offline evaluation of the RAG pipeline.

Pass 1 — Retrieval eval
  Hit@k, Precision@k, Recall@k, MRR  evaluated at k = 1, 3, 5

Pass 2 — Generation eval
  Citation coverage     – fraction of answer sentences containing a [ref_N] citation
  Citation validity     – fraction of cited chunk_ids that were in the retrieved set
  Hallucination rate    – fraction of supported_claims with no valid citation
  Cannot-answer acc.    – accuracy on queries where the corpus contains no answer

Statistical analysis
  95 % bootstrap CI on Precision@5 and Recall@5
  Metric breakdown by difficulty (easy / medium / hard)
  Pearson correlation between top-1 similarity score and answer correctness

Output
  Console table  +  eval/results/<corpus>_<timestamp>.json

Usage
  python eval/run_eval.py --corpus execution
  python eval/run_eval.py --config configs/sec.yaml
  python eval/run_eval.py --corpus execution --queries eval/queries/execution.csv \\
                          --output-dir eval/results/ --no-generate

Queries CSV columns (see eval/queries/execution.csv for template)
  query_id                  unique row ID
  query                     natural-language question
  expected_doc_id           sha256 of the document that should be retrieved
                            (get this from data/processed/<corpus>/docs.jsonl)
                            leave empty for "cannot-answer" queries
  expected_section_contains optional substring that should match chunk.metadata.section
  expected_answer_contains  optional substring used as a correctness proxy
  difficulty                easy | medium | hard
  notes                     free text
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, Config
from src.embed_index import load_index
from src.retrieve import retrieve_batch
from src.rag_answer import rag_answer, AnswerResult


# ---------------------------------------------------------------------------
# Per-query retrieval metrics
# ---------------------------------------------------------------------------

def _retrieval_at_k(
    top_chunks:    List[dict],
    expected_doc:  str,
    total_relevant: int,
    k:             int,
) -> Dict[str, float]:
    """
    Compute Hit, Precision, Recall for the first k items in top_chunks.
    A chunk is relevant if chunk["doc_id"] == expected_doc.
    """
    subset   = top_chunks[:k]
    n_rel_k  = sum(1 for c in subset if c.get("doc_id") == expected_doc)
    hit      = float(n_rel_k > 0)
    prec     = n_rel_k / k
    recall   = (n_rel_k / total_relevant) if total_relevant > 0 else 0.0
    return {"hit": hit, "precision": prec, "recall": recall}


def _mrr(top_chunks: List[dict], expected_doc: str) -> float:
    """Mean Reciprocal Rank — rank of the first relevant chunk."""
    for rank, chunk in enumerate(top_chunks, 1):
        if chunk.get("doc_id") == expected_doc:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Per-query generation metrics
# ---------------------------------------------------------------------------

def _citation_coverage(answer_text: str) -> float:
    """
    Fraction of non-empty sentences in the answer that contain at least one
    [ref_N] citation marker.
    """
    if not answer_text.strip():
        return 0.0
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r"\[ref_\d+\]", s))
    return cited / len(sentences)


def _citation_validity(citations: List[str], retrieved_ids: set) -> float:
    """
    Fraction of cited chunk_ids that actually appeared in the retrieved set.
    1.0 if no citations (vacuously valid).
    """
    if not citations:
        return 1.0
    valid = sum(1 for cid in citations if cid in retrieved_ids)
    return valid / len(citations)


def _hallucination_rate(supported_claims: List[dict], retrieved_ids: set) -> float:
    """
    Fraction of supported_claims whose citation is not in the retrieved set.
    These are claims that reference a source the model was not given — proxy
    for hallucination.
    """
    if not supported_claims:
        return 0.0
    flagged = sum(
        1 for sc in supported_claims
        if sc.get("citation", "") not in retrieved_ids
    )
    return flagged / len(supported_claims)


def _answer_correctness(answer_text: str, expected_contains: str) -> Optional[int]:
    """
    1 if expected_contains (case-insensitive) appears in answer_text, 0 if not,
    None if expected_contains is empty (ground truth unavailable).
    """
    if not expected_contains.strip():
        return None
    return int(expected_contains.lower() in answer_text.lower())


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values:  List[float],
    n_boot:  int   = 1000,
    ci:      float = 0.95,
    seed:    int   = 42,
) -> Dict[str, float]:
    """
    Estimate mean and 95 % CI via bootstrap resampling.
    Returns {"mean": …, "lower": …, "upper": …}.
    """
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    rng  = np.random.default_rng(seed)
    arr  = np.array(values, dtype=float)
    boot = np.array([
        np.mean(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_boot)
    ])
    alpha = (1.0 - ci) / 2.0
    return {
        "mean":  round(float(np.mean(arr)),                           4),
        "lower": round(float(np.percentile(boot, alpha * 100)),       4),
        "upper": round(float(np.percentile(boot, (1 - alpha) * 100)), 4),
    }


def _pearson(x: List[float], y: List[float]) -> Dict:
    """
    Pearson correlation between x and y. Requires scipy.
    Returns {"r": …, "p_value": …} or a note if scipy is missing.
    """
    pairs = [(xi, yi) for xi, yi in zip(x, y) if xi is not None and yi is not None]
    if len(pairs) < 3:
        return {"r": None, "p_value": None, "note": "too few samples"}
    xs, ys = zip(*pairs)
    try:
        from scipy import stats
        r, p = stats.pearsonr(xs, ys)
        return {"r": round(float(r), 4), "p_value": round(float(p), 4)}
    except ImportError:
        return {"r": None, "p_value": None, "note": "scipy not installed"}


def _difficulty_breakdown(rows: List[dict], k_vals: List[int]) -> Dict:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("difficulty") or "unknown"].append(row)

    result = {}
    for diff in sorted(grouped):
        grp = grouped[diff]
        entry: Dict = {"n": len(grp)}
        for k in k_vals:
            hits  = [r[f"hit_at_{k}"]      for r in grp if r.get(f"hit_at_{k}")      is not None]
            precs = [r[f"precision_at_{k}"] for r in grp if r.get(f"precision_at_{k}") is not None]
            recs  = [r[f"recall_at_{k}"]    for r in grp if r.get(f"recall_at_{k}")    is not None]
            if hits:
                entry[f"hit_at_{k}"]      = round(sum(hits)  / len(hits),  4)
                entry[f"precision_at_{k}"]= round(sum(precs) / len(precs), 4)
                entry[f"recall_at_{k}"]   = round(sum(recs)  / len(recs),  4)
        mrr_vals = [r["mrr"] for r in grp if r.get("mrr") is not None]
        entry["mrr"] = round(sum(mrr_vals) / len(mrr_vals), 4) if mrr_vals else 0.0
        result[diff] = entry
    return result


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

_W = 68

def _hline(char: str = "─") -> None:
    print(char * _W)

def _section(title: str) -> None:
    pad = (_W - len(title) - 2) // 2
    print(f"{'─' * pad} {title} {'─' * (_W - pad - len(title) - 2)}")

def _row(label: str, *vals) -> None:
    parts = "  ".join(f"{v}" for v in vals)
    print(f"  {label:<22}{parts}")


def _print_table(agg: dict) -> None:
    n = agg["n_queries"]
    print()
    _hline("═")
    print(f"  EVALUATION RESULTS  —  corpus: {agg['corpus']}  ({n} queries)")
    _hline("═")

    # ── Retrieval ──────────────────────────────────────────────────────────
    _section("RETRIEVAL")
    print(f"  {'':22}{'k=1':>8}  {'k=3':>8}  {'k=5':>8}")
    _hline()
    for metric, label in [("hit_rate", "Hit@k"), ("precision", "Precision@k"), ("recall", "Recall@k")]:
        vals = [
            f"{agg['retrieval'][f'k={k}'][metric]:.4f}"
            for k in [1, 3, 5]
        ]
        _row(label, *[f"{v:>8}" for v in vals])
    _row("MRR (over k=5)", f"{'':>8}  {'':>8}  {agg['retrieval']['k=5']['mrr']:>8.4f}")
    _hline()

    ci5 = agg["retrieval"]["k=5"].get("ci", {})
    if ci5:
        p  = ci5.get("precision", {})
        rc = ci5.get("recall", {})
        print(f"  95% CI (bootstrap, k=5)")
        print(f"    Precision@5: {p.get('mean', 0):.4f}  [{p.get('lower', 0):.4f} – {p.get('upper', 0):.4f}]")
        print(f"    Recall@5:    {rc.get('mean', 0):.4f}  [{rc.get('lower', 0):.4f} – {rc.get('upper', 0):.4f}]")
        _hline()

    # ── By difficulty ──────────────────────────────────────────────────────
    by_diff = agg.get("by_difficulty", {})
    if by_diff:
        _section("BY DIFFICULTY  (Hit@5  Prec@5  Rec@5   MRR)")
        for diff, m in by_diff.items():
            print(
                f"  {diff:<10} n={m['n']:<4} "
                f"Hit={m.get('hit_at_5', 0):.3f}  "
                f"P={m.get('precision_at_5', 0):.3f}  "
                f"R={m.get('recall_at_5', 0):.3f}  "
                f"MRR={m.get('mrr', 0):.3f}"
            )
        _hline()

    # ── Generation ─────────────────────────────────────────────────────────
    gen = agg.get("generation", {})
    if gen:
        _section("GENERATION")
        for label, key in [
            ("Citation Coverage",    "citation_coverage"),
            ("Citation Validity",    "citation_validity"),
            ("Hallucination Rate",   "hallucination_rate"),
            ("Cannot-Answer Acc.",   "cannot_answer_accuracy"),
            ("Answer Correctness",   "answer_correctness"),
        ]:
            v = gen.get(key)
            print(f"  {label:<26}  {v:.4f}" if v is not None else f"  {label:<26}  n/a")
        _hline()

    # ── Latency ────────────────────────────────────────────────────────────
    lat = agg.get("latency", {})
    if lat:
        _section("LATENCY")
        _row("Retrieval (batch)",     f"{lat.get('retrieval_batch_s', 0):.2f} s")
        _row("Retrieval (per query)", f"{lat.get('retrieval_per_query_ms', 0):.1f} ms")
        if "generation_mean_ms" in lat:
            _row("Generation mean",   f"{lat['generation_mean_ms']:.1f} ms")
            _row("Generation p50",    f"{lat['generation_p50_ms']:.1f} ms")
            _row("Generation p95",    f"{lat['generation_p95_ms']:.1f} ms")
            _row("Generation p99",    f"{lat['generation_p99_ms']:.1f} ms")
            _row("Total mean",        f"{lat['total_mean_ms']:.1f} ms")
        _hline()

    # ── Correlation ────────────────────────────────────────────────────────
    corr = agg.get("correlation", {}).get("top1_score_vs_correctness", {})
    if corr.get("r") is not None:
        _section("CORRELATION")
        print(
            f"  top-1 score vs answer correctness: "
            f"r={corr['r']:.4f}  p={corr['p_value']:.4f}"
        )
        _hline()

    print()


# ---------------------------------------------------------------------------
# Persist results
# ---------------------------------------------------------------------------

def _save_results(
    rows:       List[dict],
    agg:        dict,
    output_dir: Path,
    corpus:     str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"{corpus}_{ts}.json"
    with open(path, "w") as f:
        json.dump({"aggregate": agg, "per_query": rows}, f, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def run_eval(
    cfg:         Config,
    queries_path: Path,
    output_dir:  Optional[Path] = None,
    generate:    bool = True,
    k_vals:      List[int] = None,
) -> dict:
    """
    Run the full two-pass evaluation.

    Args:
        cfg:          Loaded Config.
        queries_path: Path to the labeled queries CSV.
        output_dir:   Directory to write timestamped JSON results.
        generate:     Whether to run the generation pass (slower; makes Claude calls).
        k_vals:       k values to evaluate at. Defaults to [1, 3, 5].

    Returns:
        Aggregate metrics dict (also saved to disk if output_dir is set).
    """
    if k_vals is None:
        k_vals = [1, 3, 5]
    max_k = max(k_vals)

    # ── Load queries ──────────────────────────────────────────────────────
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_path}")
    with open(queries_path, newline="") as f:
        raw_queries = [r for r in csv.DictReader(f) if r.get("query", "").strip()]

    if not raw_queries:
        raise ValueError(f"No valid queries in {queries_path}")

    print(f"\n[eval] corpus         : {cfg.corpus}")
    print(f"[eval] queries        : {len(raw_queries)}")
    print(f"[eval] k values       : {k_vals}")
    print(f"[eval] generation pass: {'yes' if generate else 'no'}\n")

    # ── Load index once + count chunks per doc ────────────────────────────
    print("[eval] loading index…")
    index, chunk_meta = load_index(cfg)
    doc_chunk_counts: Dict[str, int] = defaultdict(int)
    for c in chunk_meta:
        doc_chunk_counts[c.get("doc_id", "")] += 1
    print(f"[eval] index: {index.ntotal:,} vectors  |  {len(doc_chunk_counts):,} documents\n")

    # ── Retrieval pass: one batched OpenAI call ───────────────────────────
    query_texts = [r["query"] for r in raw_queries]
    print(f"[eval] embedding {len(query_texts)} queries…")
    retrieval_t0 = time.time()
    all_results = retrieve_batch(query_texts, cfg, k=max_k)
    retrieval_batch_s = time.time() - retrieval_t0
    retrieval_per_query_ms = (retrieval_batch_s / len(query_texts)) * 1000
    print(f"[eval] retrieval done in {retrieval_batch_s:.2f}s "
          f"({retrieval_per_query_ms:.1f}ms per query)\n")

    # ── Evaluate each query ───────────────────────────────────────────────
    rows: List[dict] = []

    for q_row, top_chunks in zip(raw_queries, all_results):
        qid              = q_row.get("query_id", "")
        query            = q_row["query"].strip()
        expected_doc     = q_row.get("expected_doc_id", "").strip()
        expected_section = q_row.get("expected_section_contains", "").strip()
        expected_ans     = q_row.get("expected_answer_contains", "").strip()
        difficulty       = q_row.get("difficulty", "unknown").strip() or "unknown"
        is_cannot_query  = not expected_doc    # empty doc_id ⇒ no answer exists

        total_relevant   = doc_chunk_counts.get(expected_doc, 0)
        retrieved_ids    = {c["chunk_id"] for c in top_chunks}
        top1_score       = top_chunks[0]["score"] if top_chunks else 0.0

        row: dict = {
            "query_id":                qid,
            "query":                   query,
            "difficulty":              difficulty,
            "expected_doc_id":         expected_doc,
            "expected_section_contains": expected_section,
            "expected_answer_contains":  expected_ans,
            "is_cannot_answer_query":  is_cannot_query,
            "top1_score":              round(top1_score, 4),
            "retrieved_chunk_ids":     list(retrieved_ids),
        }

        # Retrieval metrics at each k
        if not is_cannot_query and expected_doc:
            for k in k_vals:
                m = _retrieval_at_k(top_chunks, expected_doc, total_relevant, k)
                row[f"hit_at_{k}"]       = m["hit"]
                row[f"precision_at_{k}"] = round(m["precision"], 4)
                row[f"recall_at_{k}"]    = round(m["recall"], 4)
            row["mrr"] = round(_mrr(top_chunks, expected_doc), 4)
        else:
            for k in k_vals:
                row[f"hit_at_{k}"]       = None
                row[f"precision_at_{k}"] = None
                row[f"recall_at_{k}"]    = None
            row["mrr"] = None

        # Generation pass
        row["answer"]              = ""
        row["cannot_answer"]       = None
        row["citation_coverage"]   = None
        row["citation_validity"]   = None
        row["hallucination_rate"]  = None
        row["answer_correctness"]  = None
        row["cannot_answer_correct"] = None
        row["generation_latency_ms"] = None
        row["total_latency_ms"]    = None

        if generate:
            try:
                gen_t0 = time.time()
                result: AnswerResult = rag_answer(query, top_chunks, cfg)
                gen_latency_ms = (time.time() - gen_t0) * 1000
            except Exception as exc:
                print(f"  [eval] ERROR generating answer for {qid!r}: {exc}")
                rows.append(row)
                continue

            row["generation_latency_ms"] = round(gen_latency_ms, 1)
            row["total_latency_ms"] = round(retrieval_per_query_ms + gen_latency_ms, 1)
            row["answer"]      = result["answer"][:300]
            row["cannot_answer"] = result["cannot_answer"]

            row["citation_coverage"] = round(
                _citation_coverage(result["answer"]), 4)
            row["citation_validity"] = round(
                _citation_validity(result["citations"], retrieved_ids), 4)
            row["hallucination_rate"] = round(
                _hallucination_rate(result["supported_claims"], retrieved_ids), 4)
            row["answer_correctness"] = _answer_correctness(
                result["answer"], expected_ans)
            row["cannot_answer_correct"] = int(
                result["cannot_answer"] == is_cannot_query)

        rows.append(row)

        # Progress line
        ret_str = (
            f"hit@5={row.get('hit_at_5', 'n/a')}  "
            f"p@5={row.get('precision_at_5', 'n/a')}"
        ) if not is_cannot_query else "cannot-answer query"
        gen_str = (
            f"  hall={row.get('hallucination_rate', 'n/a')}"
            f"  gen={row.get('generation_latency_ms', 'n/a')}ms"
        ) if generate else ""
        print(f"  [{qid or '?':>4}]  {ret_str}{gen_str}  | {query[:55]}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    def _mean(vals):
        clean = [v for v in vals if v is not None]
        return round(sum(clean) / len(clean), 4) if clean else None

    retrieval_agg: dict = {}
    for k in k_vals:
        entry = {
            "hit_rate":  _mean([r.get(f"hit_at_{k}")       for r in rows]),
            "precision": _mean([r.get(f"precision_at_{k}") for r in rows]),
            "recall":    _mean([r.get(f"recall_at_{k}")     for r in rows]),
        }
        if k == max(k_vals):
            entry["mrr"] = _mean([r.get("mrr") for r in rows])
        retrieval_agg[f"k={k}"] = entry

    # Bootstrap CI on the highest k
    prec_vals = [r[f"precision_at_{max_k}"] for r in rows if r.get(f"precision_at_{max_k}") is not None]
    rec_vals  = [r[f"recall_at_{max_k}"]    for r in rows if r.get(f"recall_at_{max_k}")    is not None]
    retrieval_agg[f"k={max_k}"]["ci"] = {
        "precision": _bootstrap_ci(prec_vals),
        "recall":    _bootstrap_ci(rec_vals),
    }

    # Cannot-answer queries for generation metrics
    cannot_rows = [r for r in rows if r["is_cannot_answer_query"]]
    ca_acc = _mean([r.get("cannot_answer_correct") for r in cannot_rows]) if cannot_rows else None

    generation_agg = {
        "citation_coverage":    _mean([r.get("citation_coverage")   for r in rows]),
        "citation_validity":    _mean([r.get("citation_validity")    for r in rows]),
        "hallucination_rate":   _mean([r.get("hallucination_rate")  for r in rows]),
        "cannot_answer_accuracy": ca_acc,
        "answer_correctness":   _mean([
            r.get("answer_correctness") for r in rows
            if r.get("answer_correctness") is not None
        ]),
    } if generate else {}

    # Correlation: top-1 score vs correctness
    corr = _pearson(
        [r["top1_score"]         for r in rows],
        [r.get("answer_correctness") for r in rows],
    ) if generate else {}

    # Latency stats
    gen_latencies = [r["generation_latency_ms"] for r in rows
                     if r.get("generation_latency_ms") is not None]
    latency_agg: dict = {
        "retrieval_batch_s":    round(retrieval_batch_s, 2),
        "retrieval_per_query_ms": round(retrieval_per_query_ms, 1),
    }
    if gen_latencies:
        gen_arr = np.array(gen_latencies, dtype=float)
        latency_agg.update({
            "generation_mean_ms": round(float(np.mean(gen_arr)), 1),
            "generation_p50_ms":  round(float(np.percentile(gen_arr, 50)), 1),
            "generation_p95_ms":  round(float(np.percentile(gen_arr, 95)), 1),
            "generation_p99_ms":  round(float(np.percentile(gen_arr, 99)), 1),
            "total_mean_ms":      round(float(retrieval_per_query_ms + np.mean(gen_arr)), 1),
        })

    agg = {
        "corpus":       cfg.corpus,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "n_queries":    len(rows),
        "retrieval":    retrieval_agg,
        "generation":   generation_agg,
        "latency":      latency_agg,
        "by_difficulty": _difficulty_breakdown(rows, k_vals),
        "correlation":  {"top1_score_vs_correctness": corr},
    }

    _print_table(agg)

    if output_dir:
        saved = _save_results(rows, agg, output_dir, cfg.corpus)
        print(f"[eval] results saved → {saved}\n")

    return agg


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run two-pass RAG evaluation (retrieval + generation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--corpus", help="Corpus name — loads configs/<corpus>.yaml")
    group.add_argument("--config", help="Path to a config YAML")

    parser.add_argument("--queries",    default=None,
                        help="Override queries CSV path (default: from config)")
    parser.add_argument("--output-dir", default="eval/results",
                        help="Directory for timestamped JSON output (default: eval/results)")
    parser.add_argument("--no-generate", action="store_true",
                        help="Skip the generation pass (retrieval metrics only, no Claude calls)")
    parser.add_argument("--k-vals", default="1,3,5",
                        help="Comma-separated k values to evaluate (default: 1,3,5)")
    parser.add_argument("--no-save", action="store_true",
                        help="Print results to console only, do not write JSON file")

    args = parser.parse_args()

    cfg = load_config(f"configs/{args.corpus}.yaml") if args.corpus else load_config(args.config)
    queries_path = Path(args.queries) if args.queries else Path(cfg.eval.queries_file)
    output_dir   = None if args.no_save else Path(args.output_dir)
    k_vals       = [int(k.strip()) for k in args.k_vals.split(",")]

    run_eval(
        cfg          = cfg,
        queries_path = queries_path,
        output_dir   = output_dir,
        generate     = not args.no_generate,
        k_vals       = k_vals,
    )


if __name__ == "__main__":
    main()
