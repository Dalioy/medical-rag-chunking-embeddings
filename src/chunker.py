"""
Turns a flat list of RawSection objects into final Chunk objects.

Chunking rule of thumb used here:
  1. NEVER split inside a single recommendation / quality statement --
     RawSection is the atomic, indivisible unit (unless it alone exceeds
     max_tokens, in which case we fall back to sentence-level splitting
     just for that one oversized section).
  2. Group consecutive RawSections that share the same section_title,
     accumulating until adding the next one would exceed max_tokens.
  3. If a resulting chunk is smaller than min_tokens (e.g. a lone short
     recommendation followed by a section-title change), merge it forward
     rather than emitting a near-empty embedding.
  4. Each chunk is prefixed with a short human-readable context header
     ("Document ... / Section ...") before embedding text. This costs a
     few tokens but measurably improves retrieval quality for section-
     numbered regulatory text, since the raw recommendation text alone
     ("Offer an ACE inhibitor...") is ambiguous out of context.
"""

from __future__ import annotations

import logging
import re

from src.config import ChunkingConfig, DocumentSpec
from src.models import Chunk, ChunkMetadata, RawSection

logger = logging.getLogger(__name__)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def approx_token_count(text: str, token_per_word: float) -> int:
    """Cheap token estimate: word count * empirical multiplier.

    Good enough for chunk-sizing decisions without a tokenizer dependency.
    If you later plug in a real tokenizer (tiktoken, the target model's
    own tokenizer), swap the body of this function only -- every caller
    is already written against "approximate token count".
    """
    if not text:
        return 0
    return max(1, round(len(text.split()) * token_per_word))


def _split_oversized(section: RawSection, cfg: ChunkingConfig) -> list[RawSection]:
    """Sentence-split a single RawSection that alone exceeds max_tokens.

    Keeps the same section_number/title/page/year for every fragment;
    downstream chunk_ids are disambiguated with a suffix.
    """
    sentences = _SENTENCE_SPLIT_RE.split(section.text)
    fragments: list[RawSection] = []
    buffer: list[str] = []
    buffer_tokens = 0

    for sentence in sentences:
        s_tokens = approx_token_count(sentence, cfg.token_per_word)
        if buffer and buffer_tokens + s_tokens > cfg.max_tokens:
            fragments.append(
                RawSection(
                    section_number=section.section_number,
                    section_title=section.section_title,
                    text=" ".join(buffer),
                    page_number=section.page_number,
                    evidence_year=section.evidence_year,
                    recommendation_ids=section.recommendation_ids,
                )
            )
            buffer, buffer_tokens = [], 0
        buffer.append(sentence)
        buffer_tokens += s_tokens

    if buffer:
        fragments.append(
            RawSection(
                section_number=section.section_number,
                section_title=section.section_title,
                text=" ".join(buffer),
                page_number=section.page_number,
                evidence_year=section.evidence_year,
                recommendation_ids=section.recommendation_ids,
            )
        )

    logger.warning(
        "Section %s (%s) exceeded max_tokens; split into %d fragments",
        section.section_number,
        section.section_title,
        len(fragments),
    )
    return fragments


def _build_chunk_text(spec: DocumentSpec, section_title: str, body: str) -> str:
    header = f"Document: {spec.title} ({spec.doc_id}). Section: {section_title}."
    return f"{header}\n\n{body}"


def group_into_chunks(
    sections: list[RawSection], spec: DocumentSpec, cfg: ChunkingConfig
) -> list[Chunk]:
    """Main entry point: RawSections -> final, size-bounded Chunks."""
    # Pre-expand any section that alone exceeds max_tokens.
    expanded: list[RawSection] = []
    for section in sections:
        if approx_token_count(section.text, cfg.token_per_word) > cfg.max_tokens:
            expanded.extend(_split_oversized(section, cfg))
        else:
            expanded.append(section)

    raw_chunks: list[list[RawSection]] = []
    current_group: list[RawSection] = []
    current_tokens = 0

    for section in expanded:
        s_tokens = approx_token_count(section.text, cfg.token_per_word)
        same_section = (
            current_group and current_group[-1].section_title == section.section_title
        )
        fits = current_tokens + s_tokens <= cfg.max_tokens

        if current_group and same_section and fits:
            current_group.append(section)
            current_tokens += s_tokens
            continue

        if current_group:
            raw_chunks.append(current_group)
        current_group = [section]
        current_tokens = s_tokens

    if current_group:
        raw_chunks.append(current_group)

    # Merge chunks that ended up below min_tokens into a neighbour with the
    # same section_title, where possible.
    merged: list[list[RawSection]] = []
    for group in raw_chunks:
        group_tokens = sum(
            approx_token_count(s.text, cfg.token_per_word) for s in group
        )
        if (
            merged
            and group_tokens < cfg.min_tokens
            and merged[-1][-1].section_title == group[0].section_title
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)

    chunks: list[Chunk] = []
    for i, group in enumerate(merged):
        body = "\n".join(s.text for s in group)
        section_title = group[0].section_title
        rec_ids = [rid for s in group for rid in s.recommendation_ids]
        years = {s.evidence_year for s in group if s.evidence_year}
        chunk_id = f"{spec.doc_id}-{group[0].section_number}".replace(" ", "_")
        # Disambiguate in the rare case two groups start with the same
        # section_number (shouldn't normally happen, but be defensive).
        chunk_id = f"{chunk_id}-{i:04d}"

        metadata = ChunkMetadata(
            doc_id=spec.doc_id,
            doc_title=spec.title,
            doc_type=spec.doc_type,
            population=spec.population,
            section_number=group[0].section_number,
            section_title=section_title,
            recommendation_ids=rec_ids,
            evidence_year=sorted(years)[0] if years else None,
            page_number=group[0].page_number,
            last_updated=spec.last_updated,
            source_url=spec.source_url,
        )
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=_build_chunk_text(spec, section_title, body),
                metadata=metadata,
            )
        )

    logger.info(
        "%s: %d raw sections -> %d chunks (target=%d, max=%d tokens)",
        spec.doc_id,
        len(sections),
        len(chunks),
        cfg.target_tokens,
        cfg.max_tokens,
    )
    return chunks