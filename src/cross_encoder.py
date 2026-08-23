"""
Cross-Encoder Reranking Script (flashrank version)
-----------------------------------------------------
Same idea as the first script (rerank using query + document together)
but using flashrank instead of sentence-transformers.

Why flashrank?
- Does not require torch (uses onnxruntime)
- Lightweight and fast
- Uses the cross-encoder approach: each (query, passage) gets a score.

Usage:
    python cross_encoder_rerank_flashrank.py --input ../output/bp_target.json --output ../output/bp_target_reranked.json
"""

import json
import argparse
from flashrank import Ranker, RerankRequest


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rerank(data, model_name="ms-marco-MiniLM-L-12-v2"):
    query = data["query"]
    results = data["results"]

    ranker = Ranker(model_name=model_name, cache_dir="./flashrank_cache")

    passages = [
        {"id": i, "text": r["text"], "meta": r}
        for i, r in enumerate(results)
    ]

    rerank_request = RerankRequest(query=query, passages=passages)
    reranked_raw = ranker.rerank(rerank_request)

    reranked = []
    for i, item in enumerate(reranked_raw, start=1):
        original = item["meta"]
        original["cross_encoder_score"] = float(item["score"])
        original["rerank"] = i
        reranked.append(original)

    return {"query": query, "results": reranked}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to the original JSON file")
    parser.add_argument("--output", required=True, help="Path to save reranked results")
    parser.add_argument(
        "--model",
        default="ms-marco-MiniLM-L-12-v2",
        help="FlashRank model name",
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
            f'{r["doc_id"]} - {r["section_title"]} '
            f'(score={r["cross_encoder_score"]:.4f})'
        )


if __name__ == "__main__":
    main()
    