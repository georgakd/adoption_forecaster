"""The source-store interface the agent searches, and a synthetic implementation over generated documents.

Only snippet metadata and the prose text are exposed; ground-truth fields of the documents never leave the store.
"""

import re
from dataclasses import dataclass
from typing import Protocol

from quotas_forecasting.synthetic.sources import SourceDoc


@dataclass(frozen=True)
class SourceSnippet:
    doc_id: str
    publisher: str
    country: str
    year: int
    title: str


class SourceStore(Protocol):
    def search(self, query: str, country: str | None = None, year: int | None = None) -> list[SourceSnippet]: ...

    def search_with_total(self, query: str, country: str | None = None, year: int | None = None) -> tuple[list[SourceSnippet], int]:
        """Like ``search``, plus the number of matches before the result cap (so callers can tell results were cut off)."""
        ...

    def read(self, doc_id: str) -> str: ...


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class SyntheticSourceStore:
    """Keyword search over in-memory documents: filter by country/year, rank by query-token overlap, drop non-matches."""

    def __init__(self, docs: list[SourceDoc], limit: int = 20):
        self._docs = {doc.doc_id: doc for doc in docs}
        self.limit = limit

    def search(self, query: str, country: str | None = None, year: int | None = None) -> list[SourceSnippet]:
        return self.search_with_total(query, country, year)[0]

    def search_with_total(self, query: str, country: str | None = None, year: int | None = None) -> tuple[list[SourceSnippet], int]:
        query_tokens = _tokens(query)
        scored = []
        for doc in self._docs.values():
            if (country is not None and doc.country != country) or (year is not None and doc.year != year):
                continue
            score = len(query_tokens & _tokens(f"{doc.title} {doc.publisher} {doc.country} {doc.text}"))
            if score or not query_tokens:
                scored.append((-score, doc.doc_id, doc))
        scored.sort(key=lambda item: item[:2])
        return [SourceSnippet(d.doc_id, d.publisher, d.country, d.year, d.title) for _, _, d in scored[: self.limit]], len(scored)

    def read(self, doc_id: str) -> str:
        return self._docs[doc_id].text
