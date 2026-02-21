# RAG Framework

A corpus-agnostic Retrieval-Augmented Generation framework — embed any document collection, retrieve grounded answers with inline citations, and evaluate retrieval quality with real metrics.

**Stack:** Python · OpenAI Embeddings · FAISS · Claude (Anthropic) · Streamlit · pypdf · BeautifulSoup

---

## Architecture

```
  Raw Documents                         Query
       │                                  │
       ▼                                  ▼
 ┌─────────────┐                  ┌──────────────┐
 │  ingest.py  │                  │ retrieve.py  │
 │ PDF/MD/HTML │                  │  embed query │
 └──────┬──────┘                  └──────┬───────┘
        │  docs.jsonl                    │  top-k chunks
        ▼                                │
 ┌─────────────┐                         │
 │   chunk.py  │                         ▼
 │section-aware│            ┌────────────────────────┐
 │  splitter   │            │     rag_answer.py      │
 └──────┬──────┘            │  Claude + citations    │
        │  chunks.jsonl     │  structured JSON out   │
        ▼                   └────────────┬───────────┘
 ┌──────────────────┐                    │  AnswerResult
 │ embed_index.py   │                    ▼
 │ OpenAI + FAISS   │       ┌────────────────────────┐
 └──────────────────┘       │   streamlit_app.py     │
   faiss.index              │   or  run_eval.py      │
   chunk_meta.json          └────────────────────────┘
```

Each corpus is fully isolated under `data/raw/<corpus>/` and `data/processed/<corpus>/`. Switching corpora requires no code changes — only a different `--corpus` flag.

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API keys

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Add your documents

```bash
mkdir -p data/raw/execution
cp /path/to/your/docs/*.pdf  data/raw/execution/
cp /path/to/your/docs/*.md   data/raw/execution/
```

Supported formats: `.pdf`, `.txt`, `.md`, `.html`, `.htm`

### 4. Build the index

```bash
python scripts/build_pipeline.py --corpus execution
```

This runs three steps:
- **Ingest** — parse and normalize documents → `docs.jsonl`
- **Chunk** — split with section detection → `chunks.jsonl`
- **Embed** — OpenAI embeddings + FAISS index → `faiss.index`

### 5. Launch the app

```bash
streamlit run app/streamlit_app.py
```

---

## Pipeline Commands

```bash
# Full pipeline
python scripts/build_pipeline.py --corpus execution

# Partial reruns (e.g. re-embed without re-ingesting)
python scripts/build_pipeline.py --corpus execution --skip-ingest --skip-chunk

# Retrieval-only evaluation (no Claude calls)
python eval/run_eval.py --corpus execution --no-generate

# Full evaluation
python eval/run_eval.py --corpus execution --output-dir eval/results/

# Streamlit app
streamlit run app/streamlit_app.py
```

---

## Configuration

Each corpus is controlled by a YAML file in `configs/`.

```yaml
# configs/execution.yaml
corpus: execution

chunk:
  strategy: recursive       # recursive | sentence | fixed
  chunk_size: 512
  chunk_overlap: 64

embed:
  model: text-embedding-3-small
  batch_size: 64

retrieve:
  top_k: 5
  score_threshold: 0.0

answer:
  model: claude-sonnet-4-6
  temperature: 0.2
  max_tokens: 1024
  system_prompt: |
    You are a helpful assistant. Answer questions using only the provided context.
```

To add a new corpus: copy `configs/execution.yaml`, change `corpus:`, drop documents into `data/raw/<corpus>/`.

---

## Evaluation

The eval harness computes two passes over a labeled query set:

**Pass 1 — Retrieval** (no LLM calls):

| Metric | Description |
|--------|-------------|
| Hit@k  | 1 if the expected document appears in top-k |
| Precision@k | Fraction of top-k chunks from the relevant document |
| Recall@k | Fraction of relevant chunks retrieved in top-k |
| MRR | Mean Reciprocal Rank of first relevant chunk |

Evaluated at k = 1, 3, 5 with 95% bootstrap confidence intervals.

**Pass 2 — Generation** (Claude calls):

| Metric | Description |
|--------|-------------|
| Citation Coverage | Fraction of answer sentences containing a citation |
| Citation Validity | Fraction of cited sources that were in the retrieved set |
| Hallucination Rate | Fraction of claims with no valid source citation |
| Cannot-Answer Acc. | Accuracy on queries where the corpus has no answer |

### Metric targets

| Metric | Target |
|--------|--------|
| Hit@5 | > 0.80 |
| Precision@5 | > 0.65 |
| Recall@5 | > 0.75 |
| Citation Coverage | > 0.90 |
| Hallucination Rate | < 0.10 |

### Writing labeled queries

After building the index, inspect `data/processed/<corpus>/docs.jsonl` to find `doc_id` values, then fill in `eval/queries/execution.csv`:

```
query_id, query, expected_doc_id, expected_section_contains, expected_answer_contains, difficulty, notes
```

Aim for 75–100 queries. Mix easy lookups, policy-specific questions, and edge cases with no answer (leave `expected_doc_id` empty — these test the "cannot answer" path).

---

## Streamlit App

```
streamlit run app/streamlit_app.py
```

Features:
- **Corpus selector** — switch corpora from the sidebar
- **Top-k slider** — adjust retrieval depth at query time
- **Cited sources panel** — shows only chunks referenced in the answer
- **Confidence badge** — 🟢 High / 🟡 Medium / 🔴 Low
- **Feedback widget** — Correct / Partial / Wrong + comment, logged to `feedback.csv`

---

## Project Structure

```
rag-framework/
├── configs/
│   ├── execution.yaml         # tuning knobs per corpus
│   └── sec.yaml
├── data/
│   ├── raw/<corpus>/          # drop source documents here
│   └── processed/<corpus>/
│       ├── docs.jsonl         # normalized documents
│       ├── chunks.jsonl       # section-aware chunks
│       ├── faiss.index        # vector index
│       ├── chunk_meta.json    # FAISS row → chunk metadata
│       └── feedback.csv       # human feedback log
├── src/
│   ├── config.py              # Pydantic config + list_corpora()
│   ├── ingest.py              # PDF/MD/HTML → normalized docs
│   ├── chunk.py               # section detection, char offsets
│   ├── embed_index.py         # OpenAI embeddings + FAISS
│   ├── retrieve.py            # retrieve() + retrieve_batch()
│   └── rag_answer.py          # Claude + structured JSON output
├── scripts/
│   └── build_pipeline.py      # ingest → chunk → embed in one command
├── app/
│   └── streamlit_app.py       # chat UI with feedback loop
├── eval/
│   ├── queries/
│   │   └── execution.csv      # labeled query set
│   └── run_eval.py            # Hit@k, Precision@k, Recall@k, hallucination rate
└── requirements.txt
```

---

## Key Design Decisions

**Stable doc IDs:** Every document gets a `doc_id = sha256(corpus:relative_path)`. Chunk IDs are `doc_id::000042`. This makes the eval ground truth portable — `expected_doc_id` in the queries CSV points to the same document regardless of index rebuilds.

**Section-aware chunking:** The chunker scans for markdown headings and ALL-CAPS/title-case lines to build a section index, then stamps each chunk with `section`, `page` (approximate for PDFs), `char_start`, and `char_end`. Citations in the UI render as "Order Types › IOC Orders, p.4" rather than just a filename.

**Numbered context references:** Rather than asking Claude to cite 64-character hex strings, the prompt assigns `[ref_1]`, `[ref_2]` labels to chunks. After generation, references are resolved back to full chunk IDs. This makes citation reliable and keeps the prompt readable.

**Structured answer output:** `rag_answer()` returns a TypedDict with `answer`, `citations`, `confidence`, `supported_claims`, and `cannot_answer`. The `supported_claims` list (one entry per factual claim with its source) is what powers the hallucination rate metric.
