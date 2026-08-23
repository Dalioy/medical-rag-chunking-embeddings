"""
Central configuration for the NICE hypertension ingestion pipeline.

Everything that is "data about the run" (which files to read, how big chunks
should be, where output goes) lives here so the rest of the codebase stays
free of hard-coded paths and magic numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# This file lives in <project_root>/src/config.py, so project root is one
# level up. Data is expected at <project_root>/../data as requested, i.e.
# sibling to the project root -> adjust ROOT_DIR if your layout differs.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATA_DIR = (PROJECT_ROOT / "data").resolve()
OUTPUT_DIR = (PROJECT_ROOT / "output").resolve()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Document types
# ---------------------------------------------------------------------------

class DocType(str, Enum):
    """The two structural "families" of NICE PDF we know how to parse.

    Guidelines (NG-prefixed) are organised around numbered recommendations
    (e.g. 1.4.32). Quality standards (QS-prefixed) are organised around
    numbered "quality statements" and have a materially different layout.
    Each family gets its own parser in parsers.py.
    """

    GUIDELINE = "guideline"
    QUALITY_STANDARD = "quality_standard"


class Population(str, Enum):
    ADULTS = "adults"
    PREGNANCY = "pregnancy"


@dataclass(frozen=True)
class DocumentSpec:
    """Everything the pipeline needs to know about one source PDF."""

    doc_id: str                # short stable code, e.g. "NG136"
    filename: str              # file name inside DATA_DIR
    title: str                 # human-readable document title
    doc_type: DocType
    population: Population
    source_url: str
    last_updated: str          # ISO date string, best known value


# ---------------------------------------------------------------------------
# Document registry
#
# This is the ONLY place you touch when you add a 4th, 5th, ... NICE PDF.
# ---------------------------------------------------------------------------

DOCUMENTS: list[DocumentSpec] = [
    DocumentSpec(
        doc_id="NG136",
        filename="hypertension-in-adults-diagnosis-and-management-pdf-66141722710213.pdf",
        title="Hypertension in adults: diagnosis and management",
        doc_type=DocType.GUIDELINE,
        population=Population.ADULTS,
        source_url="https://www.nice.org.uk/guidance/ng136",
        last_updated="2026-02-26",
    ),
    DocumentSpec(
        doc_id="NG133",
        filename="hypertension-in-pregnancy-diagnosis-and-management-pdf-66141717671365.pdf",
        title="Hypertension in pregnancy: diagnosis and management",
        doc_type=DocType.GUIDELINE,
        population=Population.PREGNANCY,
        source_url="https://www.nice.org.uk/guidance/ng133",
        last_updated="2023-04-17",
    ),
    DocumentSpec(
        doc_id="QS35",
        filename="hypertension-in-pregnancy-pdf-2098607923141.pdf",
        title="Hypertension in pregnancy",
        doc_type=DocType.QUALITY_STANDARD,
        population=Population.PREGNANCY,
        source_url="https://www.nice.org.uk/guidance/qs35",
        last_updated="2019-07-23",
    ),
]


# ---------------------------------------------------------------------------
# Chunking parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ChunkingConfig:
    # Soft target for chunk size. We approximate tokens as
    # len(text.split()) * TOKEN_PER_WORD (see chunker.approx_token_count),
    # which is accurate enough for sizing decisions without pulling in a
    # tokenizer dependency.
    target_tokens: int = 300
    max_tokens: int = 250
    min_tokens: int = 20          # below this, merge into the next chunk
    token_per_word: float = 1.3


CHUNKING = ChunkingConfig()

