from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pickle

from google import genai
from google.genai import types
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseVector,
    PointStruct,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 3072
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
BATCH_SIZE = 5
MAX_RETRIES = 5
BASE_BACKOFF_SECONDS = 15
REQUEST_PAUSE_SECONDS = 4
QDRANT_PATH = os.environ.get("QDRANT_PATH", "./qdrantdb")
API_TIMEOUT_MS = 30_000
VECTORIZER_FILENAME = "tfidf_vectorizer.pkl"  # persisted so query-time vocab matches index-time vocab
DENSE_WEIGHT = 0.7
SPARSE_WEIGHT = 0.3


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any]


def configure_gemini() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        sys.exit(
            "GEMINI_API_KEY not set.\n"
            "Set it in PowerShell with:\n"
            "$env:GEMINI_API_KEY='your-key-here'"
        )

    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=API_TIMEOUT_MS),
    )


def vectorizer_path(qdrant_path: str) -> Path:
    return Path(qdrant_path) / VECTORIZER_FILENAME


def fit_or_load_sparse_vectorizer(chunk_texts: list[str], qdrant_path: str) -> TfidfVectorizer:
    """Pure Python/numpy TF-IDF sparse encoder — no onnxruntime, no native DLLs.
    Persisted to disk so the exact same vocabulary is used for indexing and
    querying. Loaded (not refit) on subsequent runs so existing points stay
    consistent; delete the .pkl file (or the whole QDRANT_PATH folder) to
    rebuild the vocabulary from scratch, e.g. after adding a lot of new docs.
    """
    path = vectorizer_path(qdrant_path)

    if path.exists():
        print(f"  [debug] loading existing TF-IDF vocabulary from {path}...", flush=True)
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  [debug] fitting new TF-IDF vocabulary on {len(chunk_texts)} chunks...", flush=True)
    vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", sublinear_tf=True)
    vectorizer.fit(chunk_texts)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)

    return vectorizer


def load_chunks(path: str | Path) -> list[Chunk]:
    chunks: list[Chunk] = []

    with open(path, "r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, 1):
            line = raw_line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Bad JSON on line {line_no} of {path}: {e}") from e

            chunks.append(
                Chunk(
                    id=obj["chunk_id"],
                    text=obj["text"],
                    metadata=obj.get("metadata", {}),
                )
            )

    return chunks


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}

    for key, value in meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (list, tuple)):
            clean[key] = ", ".join(str(x) for x in value)
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)

    return clean


def batched(items: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(items), size):
        yield items[i:i + size]


def embed_texts_with_retry(
    client: genai.Client,
    texts: list[str],
    task_type: str,
) -> list[list[float]]:

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        start = time.time()
        try:
            print(
                f"  [debug] calling embed_content for {len(texts)} texts (attempt {attempt})...",
                flush=True,
            )
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            print(f"  [debug] embed_content returned in {time.time() - start:.1f}s", flush=True)

            if not response.embeddings:
                raise RuntimeError("Gemini returned no embeddings.")

            embeddings = []
            for embedding in response.embeddings:
                if embedding.values is None:
                    raise RuntimeError("Gemini returned an embedding without values.")
                embeddings.append(list(embedding.values))

            if len(embeddings) != len(texts):
                raise RuntimeError(
                    f"Expected {len(texts)} embeddings, but received {len(embeddings)}."
                )

            return embeddings

        except Exception as error:
            last_error = error
            wait = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(
                f"[warn] embedding failed (attempt {attempt}/{MAX_RETRIES}): {error}",
                flush=True,
            )
            if attempt < MAX_RETRIES:
                time.sleep(wait)

    raise RuntimeError(f"Failed to embed batch after {MAX_RETRIES} attempts.") from last_error


def embed_sparse(vectorizer: TfidfVectorizer, texts: list[str]) -> list[SparseVector]:
    matrix = vectorizer.transform(texts).tocsr()
    vectors = []
    for row_idx in range(matrix.shape[0]):
        row = matrix.getrow(row_idx)
        vectors.append(
            SparseVector(
                indices=row.indices.tolist(),
                values=row.data.tolist(),
            )
        )
    return vectors


# --------------------------------------------------------------------------- #
# Qdrant storage (local embedded mode — no server needed)
# --------------------------------------------------------------------------- #

def chunk_id_to_point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def get_client() -> QdrantClient:
    return QdrantClient(path=QDRANT_PATH)


def ensure_collection(client: QdrantClient, collection_name: str) -> None:
    try:
        client.get_collection(collection_name)
    except Exception:
        print(f"  [debug] creating collection '{collection_name}' (dense + sparse)...", flush=True)
        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            },
            sparse_vectors_config={
                SPARSE_VECTOR_NAME: SparseVectorParams(),
            },
        )


def upsert_chunks(
    gemini_client: genai.Client,
    vectorizer: TfidfVectorizer,
    chunks: list[Chunk],
    collection_name: str,
) -> None:

    qdrant = get_client()
    ensure_collection(qdrant, collection_name)

    print(f"{len(chunks)} chunks total.", flush=True)

    all_point_ids = [chunk_id_to_point_id(chunk.id) for chunk in chunks]
    existing_ids: set[str] = set()
    try:
        records = qdrant.retrieve(
            collection_name=collection_name, ids=all_point_ids, with_payload=False
        )
        existing_ids = {str(r.id) for r in records}
    except Exception as e:
        print(f"  [debug] could not check existing points ({e}); embedding all.", flush=True)

    todo = [
        chunk for chunk in chunks if chunk_id_to_point_id(chunk.id) not in existing_ids
    ]
    print(
        f"{len(existing_ids)} already embedded, {len(todo)} left to embed.", flush=True
    )

    for batch in tqdm(list(batched(todo, BATCH_SIZE)), desc="Embedding & upserting"):
        texts = [chunk.text for chunk in batch]

        dense_embeddings = embed_texts_with_retry(gemini_client, texts, task_type="RETRIEVAL_DOCUMENT")
        sparse_embeddings = embed_sparse(vectorizer, texts)

        points = [
            PointStruct(
                id=chunk_id_to_point_id(chunk.id),
                vector={
                    DENSE_VECTOR_NAME: dense_vec,
                    SPARSE_VECTOR_NAME: sparse_vec,
                },
                payload={
                    **sanitize_metadata(chunk.metadata),
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                },
            )
            for chunk, dense_vec, sparse_vec in zip(batch, dense_embeddings, sparse_embeddings)
        ]

        print(f"  [debug] upserting {len(points)} points...", flush=True)
        qdrant.upsert(collection_name=collection_name, points=points)
        print("  [debug] upsert complete.", flush=True)

        time.sleep(REQUEST_PAUSE_SECONDS)


def _normalize(scores: list[float]) -> list[float]:
    """Min-max normalize to [0, 1] so dense (cosine) and sparse (TF-IDF)
    scores are on the same scale before combining them."""
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def query(
    gemini_client: genai.Client,
    vectorizer: TfidfVectorizer,
    collection_name: str,
    question: str,
    n_results: int = 5,
    dense_weight: float = DENSE_WEIGHT,
    sparse_weight: float = SPARSE_WEIGHT,
    candidate_pool: int = 30,
    query_filter=None,
):
    """Hybrid search: combines dense (semantic) + sparse (TF-IDF keyword)
    scores directly, after min-max normalizing each to [0, 1].

    final_score = dense_weight * normalized_dense_score
                + sparse_weight * normalized_sparse_score
    """
    qdrant = get_client()

    dense_query = embed_texts_with_retry(
        gemini_client, [question], task_type="RETRIEVAL_QUERY"
    )[0]
    sparse_query = embed_sparse(vectorizer, [question])[0]

    dense_results = qdrant.query_points(
        collection_name=collection_name,
        query=dense_query,
        using=DENSE_VECTOR_NAME,
        limit=candidate_pool,
        query_filter=query_filter,
    ).points

    sparse_results = qdrant.query_points(
        collection_name=collection_name,
        query=sparse_query,
        using=SPARSE_VECTOR_NAME,
        limit=candidate_pool,
        query_filter=query_filter,
    ).points

    dense_norm = _normalize([p.score for p in dense_results])
    sparse_norm = _normalize([p.score for p in sparse_results])

    dense_scores = {p.id: score for p, score in zip(dense_results, dense_norm)}
    sparse_scores = {p.id: score for p, score in zip(sparse_results, sparse_norm)}

    payload_by_id = {p.id: p.payload for p in dense_results}
    payload_by_id.update({p.id: p.payload for p in sparse_results})

    all_ids = set(dense_scores) | set(sparse_scores)

    scored = [
        (
            point_id,
            dense_weight * dense_scores.get(point_id, 0.0)
            + sparse_weight * sparse_scores.get(point_id, 0.0),
        )
        for point_id in all_ids
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)

    @dataclass
    class ScoredPoint:
        id: Any
        score: float
        payload: dict[str, Any]

    return [
        ScoredPoint(id=point_id, score=score, payload=payload_by_id.get(point_id, {}))
        for point_id, score in scored[:n_results]
    ]


def save_results_to_json(question: str, results: list, output_path: str | Path) -> None:
    """Save query results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "query": question,
        "results": [
            {
                "rank": i + 1,
                "score": round(float(p.score), 6),
                "doc_id": (p.payload or {}).get("doc_id"),
                "section_title": (p.payload or {}).get("section_title"),
                "text": (p.payload or {}).get("text"),
            }
            for i, p in enumerate(results)
        ],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"  [debug] saved results to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed parsed RAG chunks into Qdrant (local, embedded) with hybrid dense+sparse search."
    )
    parser.add_argument("--input", required=True, help="Path to chunks.jsonl")
    parser.add_argument("--collection", default="nice_guidelines", help="Qdrant collection name")
    parser.add_argument("--test-query", default=None, help="Optional test query after ingestion")
    parser.add_argument(
        "--dense-weight", type=float, default=DENSE_WEIGHT,
        help=f"Weight for dense results (default {DENSE_WEIGHT})"
    )
    parser.add_argument(
        "--sparse-weight", type=float, default=SPARSE_WEIGHT,
        help=f"Weight for sparse results (default {SPARSE_WEIGHT})"
    )
    default_output_json = str(Path(__file__).resolve().parent.parent / "output" / "query_results.json")
    parser.add_argument(
        "--output-json", default=default_output_json,
        help=f"Path to save query results as JSON (default '{default_output_json}')"
    )
    args = parser.parse_args()

    print("Starting embedding pipeline...", flush=True)

    gemini_client = configure_gemini()
    print(f"Using embedding model: {EMBEDDING_MODEL}", flush=True)

    print(f"Loading chunks from: {args.input}", flush=True)
    chunks = load_chunks(args.input)
    print(f"Loaded {len(chunks)} chunks.", flush=True)

    vectorizer = fit_or_load_sparse_vectorizer([c.text for c in chunks], QDRANT_PATH)

    upsert_chunks(gemini_client, vectorizer, chunks, args.collection)

    print(
        f"Done. Collection '{args.collection}' persisted at '{QDRANT_PATH}'.",
        flush=True,
    )

    if args.test_query:
        print("\n--- Hybrid (dense + TF-IDF) results ---", flush=True)
        results = query(
            gemini_client, vectorizer, args.collection, args.test_query,
            dense_weight=args.dense_weight, sparse_weight=args.sparse_weight,
        )

        save_results_to_json(args.test_query, results, args.output_json)

        for point in results:
            payload = point.payload or {}
            print(
                f"\n[{payload.get('doc_id')} | {payload.get('section_title')} | score={point.score:.3f}]"
            )
            print(payload.get("text") or "")


if __name__ == "__main__":
    main()