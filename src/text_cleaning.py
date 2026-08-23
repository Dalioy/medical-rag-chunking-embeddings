"""
PDF text extraction and boilerplate removal.

NICE PDFs repeat the same header/footer block on every single page
(document title, copyright notice, "Page X of Y"). Left in place, this text
gets tokenized into every chunk that happens to span a page boundary,
wasting embedding budget and diluting semantic signal. We strip it before
any parsing happens.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from pypdf import PdfReader

logger = logging.getLogger(__name__)


# Matches footers like:
#   "Hypertension in adults: diagnosis and management (NG136)"
#   "© NICE 2026. All rights reserved. Subject to Notice of rights
#    (https://www.nice.org.uk/terms-and-conditions#notice-of-rights)."
#   "Page 12 of 52"
_TITLE_FOOTER_RE = re.compile(
    r"^.{0,120}\(\s*(?:NG|QS|CG)\d+\s*\)\s*$", re.MULTILINE
)
_COPYRIGHT_RE = re.compile(
    r"©\s*NICE\s*\d{4}\.\s*All rights reserved\.[^\n]*?"
    r"(?:notice-of-rights\)\.?|rights\)\.?)",
    re.DOTALL,
)
_PAGE_NUMBER_RE = re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.MULTILINE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """Return a list of (1-indexed page number, raw page text)."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    logger.info("Extracted %d pages from %s", len(pages), pdf_path.name)
    return pages


def clean_page_text(text: str) -> str:
    """Strip repeated NICE header/footer boilerplate and normalise whitespace.

    Order matters: remove the copyright block (which can itself span
    several lines) before the single-line title/footer pattern, then strip
    page numbers, then collapse whitespace.
    """
    text = _COPYRIGHT_RE.sub("", text)
    text = _TITLE_FOOTER_RE.sub("", text)
    text = _PAGE_NUMBER_RE.sub("", text)
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_BLANK_RE.sub("\n\n", text)
    return text.strip()


def extract_clean_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """extract_pages() + clean_page_text() in one call."""
    return [(num, clean_page_text(text)) for num, text in extract_pages(pdf_path)]
