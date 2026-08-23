"""
Structure-aware parsers for NICE documents.

Design decision (see conversation / README): guidelines and quality
standards have genuinely different layouts, so each gets its own parser.
Both return the same `RawSection` shape (models.py) so everything
downstream — chunking, metadata, storage — is format-agnostic.

These parsers are regex-driven heuristics tuned against the actual
extracted text of NG136 / NG133 / QS35. PDF text extraction is never
perfectly clean (wrapped lines, occasional merged words), so if you add a
new NICE document and sections stop lining up, this is the file to adjust
first — start by printing `text_cleaning.extract_clean_pages(...)` for the
new PDF and comparing its line shapes against the regexes below.
"""

from __future__ import annotations

import logging
import re

from src.config import DocumentSpec, DocType
from src.models import RawSection

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared patterns
# ---------------------------------------------------------------------------

# Trailing evidence-year tag, e.g. "[2019]" or "[2011, amended 2019]"
_YEAR_TAG_RE = re.compile(r"\[(\d{4}(?:,\s*amended\s*\d{4})?)\]\s*$")


def _extract_year_tag(text: str) -> tuple[str, str | None]:
    """Strip a trailing [YYYY] / [YYYY, amended YYYY] tag, return (clean_text, year)."""
    match = _YEAR_TAG_RE.search(text)
    if not match:
        return text.strip(), None
    year = match.group(1)
    clean = _YEAR_TAG_RE.sub("", text).strip()
    return clean, year


# ---------------------------------------------------------------------------
# Guideline parser (NG-prefixed docs: NG136, NG133, ...)
# ---------------------------------------------------------------------------

# Top-level, un-numbered chapter headings that appear as standalone lines.
# Used only to give generic (non-recommendation) prose a sensible
# section_title; matching is exact-line, case-sensitive.
_GUIDELINE_CHAPTERS = {
    "Overview",
    "Who is it for?",
    "Recommendations",
    "Terms used in this guideline",
    "Recommendations for research",
    "Key recommendations for research",
    "Other recommendations for research",
    "Rationale and impact",
    "Context",
    "Finding more information and committee details",
    "Update information",
}

# "1.4 Treating and monitoring hypertension" -- exactly two numeric levels.
# Three-level numbers (recommendation IDs) never match this because the
# pattern requires whitespace immediately after the second number.
_SECTION_HEADER_RE = re.compile(r"^(\d+\.\d+)\s+([A-Z][A-Za-z0-9 ,\-'/&]+)$")

# "1.4.32 Offer an ACE inhibitor ..." -- three (or more) numeric levels.
_RECOMMENDATION_ID_RE = re.compile(r"^(\d+\.\d+\.\d+)\s+(.*)$")


def _buffer_ends_closed(buffer_lines: list[str]) -> bool:
    """True if the accumulated buffer already ends with a [YYYY] evidence tag.

    Every NICE recommendation ends with exactly one such tag. If we hit
    that tag and the following line does NOT match a known header/rec-id
    pattern, it is virtually always a bare, un-numbered sub-heading (e.g.
    "Step 1 treatment", "Lifestyle interventions") rather than a
    continuation of the sentence we just closed. Treating "buffer already
    closed" as a hard boundary avoids gluing those sub-headings onto the
    tail of the previous recommendation's text -- see README for why a
    pure regex/header-list approach can't catch these reliably on its own.
    """
    if not buffer_lines:
        return False
    joined = " ".join(l.strip() for l in buffer_lines if l.strip())
    return bool(_YEAR_TAG_RE.search(joined))


def parse_guideline(
    pages: list[tuple[int, str]], spec: DocumentSpec
) -> list[RawSection]:
    """Parse an NG-style guideline into RawSections.

    Two kinds of RawSection come out of this:
      * one per numbered recommendation (the common case, e.g. "1.4.32")
      * one per prose paragraph in un-numbered chapters (Context, Overview,
        Rationale and impact, ...), so that guideline material isn't lost
        just because it lacks a recommendation ID.
    """
    sections: list[RawSection] = []

    current_section_number = ""
    current_section_title = "Overview"

    # Buffer for the item currently being accumulated (a recommendation OR
    # a generic paragraph), plus the id/page it started on.
    buffer_lines: list[str] = []
    buffer_rec_id: str | None = None
    buffer_page: int | None = None
    paragraph_counter = 0

    def flush() -> None:
        nonlocal buffer_lines, buffer_rec_id, buffer_page, paragraph_counter
        if not buffer_lines:
            return
        raw_text = " ".join(l.strip() for l in buffer_lines if l.strip())
        if not raw_text:
            buffer_lines = []
            return
        clean_text, year = _extract_year_tag(raw_text)
        if buffer_rec_id:
            section_number = buffer_rec_id
            rec_ids = [buffer_rec_id]
        else:
            paragraph_counter += 1
            section_number = f"{_slug(current_section_title)}-p{paragraph_counter}"
            rec_ids = []
        sections.append(
            RawSection(
                section_number=section_number,
                section_title=current_section_title,
                text=clean_text,
                page_number=buffer_page or 1,
                evidence_year=year,
                recommendation_ids=rec_ids,
            )
        )
        buffer_lines = []
        buffer_rec_id = None
        buffer_page = None

    for page_num, page_text in pages:
        for line in page_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                # Blank line: only meaningful as a paragraph break when we
                # are accumulating generic (non-recommendation) prose.
                if buffer_rec_id is None and buffer_lines:
                    flush()
                continue

            if stripped in _GUIDELINE_CHAPTERS:
                flush()
                current_section_title = stripped
                current_section_number = ""
                paragraph_counter = 0
                continue

            header_match = _SECTION_HEADER_RE.match(stripped)
            if header_match:
                flush()
                current_section_number = header_match.group(1)
                current_section_title = header_match.group(2).strip()
                paragraph_counter = 0
                continue

            rec_match = _RECOMMENDATION_ID_RE.match(stripped)
            if rec_match:
                flush()
                buffer_rec_id = rec_match.group(1)
                buffer_page = page_num
                buffer_lines = [rec_match.group(2)]
                continue

            # The previous recommendation already closed with a [YYYY] tag,
            # and this line didn't match any known header/recommendation
            # pattern -- treat it as the start of a new, un-numbered
            # sub-heading / paragraph rather than gluing it onto the tail
            # of what we just closed.
            if buffer_rec_id is not None and _buffer_ends_closed(buffer_lines):
                flush()
                buffer_page = page_num
                buffer_lines = [stripped]
                continue

            # Continuation of whatever we're currently accumulating.
            if buffer_page is None:
                buffer_page = page_num
            buffer_lines.append(stripped)

    flush()
    logger.info(
        "Parsed %d raw sections from guideline %s (%s)",
        len(sections),
        spec.doc_id,
        spec.filename,
    )
    return sections


# ---------------------------------------------------------------------------
# Quality standard parser (QS-prefixed docs: QS35, ...)
# ---------------------------------------------------------------------------

# "Quality statement 3: Antenatal blood pressure targets"
_QS_STATEMENT_RE = re.compile(r"^Quality statement (\d+):\s*(.+)$")


def parse_quality_standard(
    pages: list[tuple[int, str]], spec: DocumentSpec
) -> list[RawSection]:
    """Parse a QS-style quality standard into one RawSection per statement.

    Quality standards are organised around numbered "quality statements"
    (a single sentence) followed by rationale / measures / audience
    guidance prose. We keep each statement's full block together as one
    RawSection — the chunker will split it further only if it exceeds
    max_tokens.
    """
    sections: list[RawSection] = []

    current_number: str | None = None
    current_title = ""
    buffer_lines: list[str] = []
    buffer_page: int | None = None

    def flush() -> None:
        nonlocal buffer_lines, buffer_page
        if current_number is None or not buffer_lines:
            buffer_lines = []
            return
        text = " ".join(l.strip() for l in buffer_lines if l.strip()).strip()
        if not text:
            buffer_lines = []
            return
        sections.append(
            RawSection(
                section_number=f"QS{spec.doc_id.lstrip('QS')}-{current_number}",
                section_title=current_title,
                text=text,
                page_number=buffer_page or 1,
                evidence_year=None,
                recommendation_ids=[f"statement-{current_number}"],
            )
        )
        buffer_lines = []
        buffer_page = None

    for page_num, page_text in pages:
        for line in page_text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue

            stmt_match = _QS_STATEMENT_RE.match(stripped)
            if stmt_match:
                flush()
                current_number = stmt_match.group(1)
                current_title = stmt_match.group(2).strip()
                buffer_page = page_num
                continue

            if current_number is not None:
                buffer_lines.append(stripped)

    flush()
    logger.info(
        "Parsed %d raw sections from quality standard %s (%s)",
        len(sections),
        spec.doc_id,
        spec.filename,
    )
    return sections


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:40]


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def parse_document(
    pages: list[tuple[int, str]], spec: DocumentSpec
) -> list[RawSection]:
    """Route to the correct parser based on the document's declared type."""
    if spec.doc_type == DocType.GUIDELINE:
        return parse_guideline(pages, spec)
    if spec.doc_type == DocType.QUALITY_STANDARD:
        return parse_quality_standard(pages, spec)
    raise ValueError(f"No parser registered for doc_type={spec.doc_type!r}")
