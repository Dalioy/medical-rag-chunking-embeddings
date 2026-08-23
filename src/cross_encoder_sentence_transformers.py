"""
Cross-Encoder Reranking Script (sentence-transformers version)
----------------------------------------------------------------
Reranks hybrid retrieval results by scoring each (query, passage) pair
jointly with a real cross-encoder model, using the `sentence-transformers`
library directly (not the onnxruntime-based flashrank).

Why sentence-transformers (vs. the flashrank variant)?
- Runs the original PyTorch cross-encoder checkpoints, so scores match
  published benchmarks exactly (flashrank uses an onnx-converted copy).
- Generally the higher-accuracy option when raw quality matters more than
  latency/footprint.
- Trade-off: requires `torch` + `sentence-transformers` installed (heavier
  install, slower cold start, no GPU = noticeably slower than flashrank on
  CPU-only machines). If you're constrained on install size or latency,
  use `cross_encoder.py` (flashrank) instead — same input/output contract.

Usage:
    python -m src.cross_encoder_sentence_transformers \
        --input output/query_results.json \
        --output output/query_results_reranked.json
"""

from __future__ import annotations

import argparse
import json

from sentence_transformers import CrossEncoder

DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rerank(data: dict, model_name: str = DEFAULT_MODEL) -> dict:
    """Score every (query, passage) pair with the cross-encoder and re-sort by it.

    Mirrors the flashrank variant's contract exactly: takes the same
    {"query": ..., "results": [...]} shape and returns the same shape back,
    with `cross_encoder_score` and `rerank` added to each result.
    """
    query = data["query"]
    results = data["results"]

    model = CrossEncoder(model_name)

    pairs = [(query, r["text"]) for r in results]
    scores = model.predict(pairs)

    scored_results = [
        {**r, "cross_encoder_score": float(score)}
        for r, score in zip(results, scores)
    ]
    scored_results.sort(key=lambda r: r["cross_encoder_score"], reverse=True)

    reranked = []
    for i, r in enumerate(scored_results, start=1):
        r["rerank"] = i
        reranked.append(r)

    return {"query": query, "results": reranked}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerank hybrid retrieval results with a sentence-transformers cross-encoder."
    )
    parser.add_argument("--input", required=True, help="Path to the original JSON file")
    parser.add_argument("--output", required=True, help="Path to save reranked results")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"sentence-transformers cross-encoder model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    data = load_data(args.input)
    result = rerank(data, model_name=args.model)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"Reranked results saved to: {args.output}\n")
    print("New ranking based on Cross-Encoder score:")
    for r in result["results"]:
        print(
            f'  rank {r["rerank"]}: '
            f'{r.get("doc_id")} - {r.get("section_title")} '
            f'(score={r["cross_encoder_score"]:.4f})'
        )


if __name__ == "__main__":
    main()
