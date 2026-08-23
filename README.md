# Medical RAG — NICE Hypertension Guidelines

> **Project Reconstruction:** This repository is a reconstruction and further refinement of a project originally developed for the **AI Hackathon Egypt 2026**. The current version focuses on rebuilding the pipeline with a cleaner architecture, reproducible ingestion workflow, hybrid retrieval, local Qdrant storage, and cross-encoder reranking.

A hybrid Retrieval-Augmented Generation pipeline over NICE's hypertension guidance (NG136, NG133) and quality standard (QS35): structure-aware PDF parsing → size-bounded chunking → hybrid dense+sparse embedding in a local Qdrant store → hybrid retrieval → cross-encoder reranking.


```
PDF (NICE guideline / quality standard)
        │
        ▼
 text_cleaning.py    strip repeated header/footer boilerplate
        │
        ▼
 parsers.py          structure-aware parsing → RawSection objects
        │             (guideline parser vs. quality-standard parser)
        ▼
 chunker.py           group RawSections into size-bounded Chunks
        │             (never splits mid-recommendation)
        ▼
 ingest.py            orchestrates the above → output/chunks.jsonl
        │
        ▼
 combined_cross_embed_pipeline.py
        │  - dense embeddings (Gemini `gemini-embedding-001`)
        │  - sparse embeddings (TF-IDF)
        │  - hybrid upsert into a local, embedded Qdrant collection (qdrantdb/)
        │  - hybrid query (dense + sparse, min-max normalized & weighted)
        ▼
 cross_encoder.py     reranks hybrid results with FlashRank (query+passage)
```

---

## Project Structure

```
data/                          # raw NICE PDFs — add manually, not tracked
output/                        # generated: chunks.jsonl, query_results.json, reranked results
qdrantdb/                      # generated: local embedded Qdrant vector store
src/
├── __init__.py
├── config.py                  # document registry + chunking parameters (single source of truth)
├── models.py                  # RawSection / ChunkMetadata / Chunk dataclasses
├── text_cleaning.py           # PDF extraction + NICE header/footer boilerplate removal
├── parsers.py                 # structure-aware parsers (guideline vs. quality standard)
├── chunker.py                 # groups RawSections into size-bounded Chunks
├── ingest.py                  # pipeline entry point: PDFs → output/chunks.jsonl
├── combined_cross_embed_pipeline.py   # hybrid (dense+sparse) embedding, Qdrant upsert, query
└── cross_encoder.py            # FlashRank cross-encoder reranking of query results
.env.example                   # required environment variables (copy to .env)
.gitignore
requirements.txt
```

---

## How It Works

### 1. Document registry (`src/config.py`)
Every source PDF is declared once in `DOCUMENTS: list[DocumentSpec]` — its
`doc_id`, filename, title, `doc_type` (`guideline` vs `quality_standard`),
population, source URL, and last-updated date. **This is the only place you
touch when adding a new NICE PDF.**

### 2. Text cleaning (`src/text_cleaning.py`)
NICE PDFs repeat the same title/copyright/page-number block on every page.
`extract_clean_pages()` strips it before any parsing happens, so it never
leaks into a chunk that spans a page boundary.

### 3. Structure-aware parsing (`src/parsers.py`)
Guidelines (NG-prefixed) and quality standards (QS-prefixed) have genuinely
different layouts, so each gets its own regex-driven parser:
- **Guidelines** → one `RawSection` per numbered recommendation (e.g.
  `1.4.32`), plus one per prose paragraph in un-numbered chapters (Overview,
  Context, Rationale and impact, ...) so nothing is silently dropped.
- **Quality standards** → one `RawSection` per numbered quality statement.

Both parsers emit the same `RawSection` shape, so everything downstream is
format-agnostic.

### 4. Chunking (`src/chunker.py`)
Turns `RawSection`s into final, size-bounded `Chunk`s:
1. A `RawSection` (one recommendation / one quality statement) is **never**
   split — unless it alone exceeds `max_tokens`, in which case it falls back
   to sentence-level splitting just for that section.
2. Consecutive sections under the same heading are grouped together up to
   `max_tokens`.
3. Chunks smaller than `min_tokens` are merged into a neighbouring chunk
   under the same heading rather than emitting a near-empty embedding.
4. Every chunk is prefixed with a short context header
   (`Document: ... / Section: ...`) before embedding — this measurably
   improves retrieval quality for section-numbered regulatory text.

Token size is an approximation (`word_count × 1.3`) — good enough for
sizing decisions without a tokenizer dependency.

### 5. Ingestion entry point (`src/ingest.py`)
Runs steps 2–4 for every document in the registry and writes the combined
result to `output/chunks.jsonl` — one JSON object per chunk, each tagged
with its `doc_id` so a single collection can later be filtered back down to
one source document.

### 6. Hybrid embedding & storage (`src/combined_cross_embed_pipeline.py`)
- **Dense** embeddings via Google's `gemini-embedding-001` (3072-dim,
  cosine distance), with retry/backoff on failures.
- **Sparse** embeddings via a TF-IDF vectorizer, fit once and persisted to
  `qdrantdb/tfidf_vectorizer.pkl` so index-time and query-time vocabulary
  always match.
- Both are upserted into a **local, embedded Qdrant instance** (no server
  needed) at `qdrantdb/`, skipping chunks that are already indexed.
- `query()` runs both searches, min-max normalizes each score to `[0, 1]`,
  and combines them as `0.7 × dense + 0.3 × sparse` by default.

### 7. Reranking (`src/cross_encoder.py`)
Takes a saved query-result JSON and reranks it with a FlashRank
cross-encoder (`ms-marco-MiniLM-L-12-v2` by default) — scores each
`(query, passage)` pair jointly rather than comparing independent
embeddings, which typically improves precision at the top of the list.

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Gemini API key

```bash
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=...
```

### 3. Add the source PDFs

Place the three NICE PDFs referenced in `src/config.py` into `data/`:

```
data/hypertension-in-adults-diagnosis-and-management-pdf-66141722710213.pdf   (NG136)
data/hypertension-in-pregnancy-diagnosis-and-management-pdf-66141717671365.pdf (NG133)
data/hypertension-in-pregnancy-pdf-2098607923141.pdf                          (QS35)
```

(Download links are in each `DocumentSpec.source_url` in `src/config.py`.)

---

## Usage

### Step 1 — Parse & chunk

```bash
python -m src.ingest
```

Produces `output/chunks.jsonl` and logs a 3-chunk sanity sample.

### Step 2 — Embed & index (hybrid, local Qdrant)

```bash
python -m src.combined_cross_embed_pipeline --input output/chunks.jsonl --collection nice_guidelines
```

Optionally run a test query and save results in one go:

```bash
python -m src.combined_cross_embed_pipeline \
  --input output/chunks.jsonl \
  --collection nice_guidelines \
  --test-query "What blood pressure target should be used in pregnancy?"
```

This writes `output/query_results.json` by default.

### Step 3 — Rerank (optional)

```bash
python -m src.cross_encoder --input output/query_results.json --output output/query_results_reranked.json
```

---

## Configuration

All tunable parameters live in `src/config.py`:

| Setting | Default | Meaning |
|---|---|---|
| `ChunkingConfig.target_tokens` | 300 | Soft target chunk size |
| `ChunkingConfig.max_tokens` | 250 | Hard ceiling before a section is force-split or a chunk is closed |
| `ChunkingConfig.min_tokens` | 20 | Below this, a chunk is merged forward |
| `ChunkingConfig.token_per_word` | 1.3 | Word→token approximation multiplier |

And in `src/combined_cross_embed_pipeline.py`:

| Setting | Default | Meaning |
|---|---|---|
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Dense embedding model |
| `DENSE_WEIGHT` / `SPARSE_WEIGHT` | `0.7` / `0.3` | Hybrid score blend |
| `QDRANT_PATH` | `./qdrantdb` | Local Qdrant persistence directory (override via env var) |
| `BATCH_SIZE` | 5 | Chunks embedded per API call |

---

## Requirements

| Package | Purpose |
|---|---|
| `pypdf` | PDF text extraction |
| `google-genai` | Gemini embedding API client |
| `qdrant-client` | Local embedded vector store (dense + sparse hybrid) |
| `scikit-learn` | TF-IDF sparse vectorizer |
| `tqdm` | Progress bars during embedding/upsert |
| `flashrank` | Cross-encoder reranking (no torch dependency, uses onnxruntime) |

```bash
pip install -r requirements.txt
```

---

## Notes

- The local Qdrant store (`qdrantdb/`) and all generated files in `output/`
  are reproducible from `data/` + the pipeline, so they're git-ignored by
  default (see `.gitignore`) — regenerate them by re-running the steps
  above.
- Adding a 4th NICE document only requires a new `DocumentSpec` entry in
  `src/config.py` plus the PDF itself in `data/`; no other code changes are
  needed unless its text layout doesn't match the existing guideline/QS
  regex patterns (see the module docstring in `src/parsers.py` for how to
  debug that).
