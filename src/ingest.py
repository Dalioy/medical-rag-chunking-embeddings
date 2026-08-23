"""
Pipeline entry point.

    python -m src.ingest

For each DocumentSpec in config.DOCUMENTS:
    1. Extract + clean page text        (text_cleaning.py)
    2. Parse into RawSections           (parsers.py, per doc_type)
    3. Group into size-bounded Chunks   (chunker.py)

All documents are parsed independently (each doc_type gets its own
regex-driven logic) but every chunk lands in the SAME output file /
vector-store collection, tagged with doc_id in its metadata. This is what
lets a single retrieval query pull relevant passages across NG136, NG133
and QS35 at once, while still allowing you to filter down to one source
document when needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.chunker import group_into_chunks
from src.config import CHUNKING, DATA_DIR, DOCUMENTS, OUTPUT_DIR
from src.models import Chunk
from src.parsers import parse_document
from src.text_cleaning import extract_clean_pages

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run() -> list[Chunk]:
    all_chunks: list[Chunk] = []

    for spec in DOCUMENTS:
        pdf_path = DATA_DIR / spec.filename
        logger.info("=== Processing %s: %s ===", spec.doc_id, spec.filename)

        try:
            pages = extract_clean_pages(pdf_path)
        except FileNotFoundError:
            logger.error("Missing file, skipping: %s", pdf_path)
            continue

        sections = parse_document(pages, spec)
        if not sections:
            logger.warning(
                "No sections parsed for %s -- check parser regexes against "
                "this document's actual text layout.",
                spec.doc_id,
            )
            continue

        chunks = group_into_chunks(sections, spec, CHUNKING)
        all_chunks.extend(chunks)

    logger.info("TOTAL chunks across all documents: %d", len(all_chunks))
    return all_chunks


def write_jsonl(chunks: list[Chunk], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    logger.info("Wrote %d chunks to %s", len(chunks), out_path)


if __name__ == "__main__":
    chunks = run()
    write_jsonl(chunks, OUTPUT_DIR / "chunks.jsonl")

    # Small sanity sample in the log so you can eyeball chunk quality
    # immediately after a run, without opening the JSONL file.
    for c in chunks[:3]:
        logger.info("--- Sample chunk %s ---\n%s", c.chunk_id, c.text[:300])
