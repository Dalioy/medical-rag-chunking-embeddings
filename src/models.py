"""
Data models shared across the pipeline.

Keeping these as explicit dataclasses (rather than passing dicts around)
gives us type checking, autocomplete, and a single source of truth for the
chunk schema that downstream consumers (embedding step, vector store) rely
on.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

from src.config import DocType, Population


@dataclass
class RawSection:
    """One logical unit extracted from a PDF before chunking.

    For guidelines: usually one numbered recommendation or a short run of
    tightly related recommendations under the same sub-heading.
    For quality standards: one quality statement.

    This is the atomic unit the chunker groups into final Chunks — it is
    never split mid-recommendation / mid-statement.
    """

    section_number: str           # e.g. "1.4.32" or "QS3"
    section_title: str            # nearest enclosing heading, e.g. "Step 1 treatment"
    text: str
    page_number: int
    evidence_year: Optional[str] = None   # parsed from trailing [2019] style tag
    recommendation_ids: list[str] = field(default_factory=list)


@dataclass
class ChunkMetadata:
    doc_id: str
    doc_title: str
    doc_type: DocType
    population: Population
    section_number: str
    section_title: str
    recommendation_ids: list[str]
    evidence_year: Optional[str]
    page_number: int
    last_updated: str
    source_url: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["doc_type"] = self.doc_type.value
        d["population"] = self.population.value
        return d


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: ChunkMetadata

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata.to_dict(),
        }
