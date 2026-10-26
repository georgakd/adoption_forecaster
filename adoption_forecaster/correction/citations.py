"""Deterministic citation check: the quote is in the document, and its number, year and unit match the claim."""

import re

from adoption_forecaster.correction.sources import SourceStore
from adoption_forecaster.correction.types import Citation, Correction

_YEAR = re.compile(r"(?<![\d.])(?:19|20)\d{2}(?![\d%])")
_PERCENT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)%")
_SHARE = re.compile(r"(?<![\d.])([01]\.\d+)(?![\d%])")
_EPS = 1e-12


_CLAUSE_BREAK = re.compile(r"(?<=[.!?])\s+|;")


def _stated_pairs(quote: str, unit: str) -> list[tuple[float, int]]:
    """(share-scale number in ``unit``, year) pairs: each number takes the nearest year in its own clause, else the quote's only year."""
    number_re, scale = (_PERCENT, 100.0) if unit == "percent" else (_SHARE, 1.0)
    quote_years = {int(y) for y in _YEAR.findall(quote)}
    pairs = []
    for clause in _CLAUSE_BREAK.split(quote):
        years = [(m.start(), int(m.group())) for m in _YEAR.finditer(clause)]
        for m in number_re.finditer(clause):
            if years:
                year = min(years, key=lambda y: abs(y[0] - m.start()))[1]
            elif len(quote_years) == 1:
                year = next(iter(quote_years))
            else:
                continue
            pairs.append((float(m.group(1)) / scale, year))
    return pairs


def verify_citation(citation: Citation, store: SourceStore, tolerance: float) -> bool:
    """True only if the doc exists, the quote occurs verbatim, and a number stated in the cited unit matches ``value`` AND its own year matches."""
    try:
        if citation.unit not in ("percent", "share") or not citation.quoted_text:
            return False
        if citation.quoted_text not in store.read(citation.doc_id):
            return False
        return any(
            year == citation.year and abs(number - float(citation.value)) <= tolerance + _EPS
            for number, year in _stated_pairs(citation.quoted_text, citation.unit)
        )
    except Exception:
        return False


def citation_supports_correction(correction: Correction, store: SourceStore, tolerance: float) -> bool:
    """verify_citation plus ties to the correction: same year, cited value == new_value (within tolerance), country named in the document."""
    try:
        citation = correction.citation
        return (
            verify_citation(citation, store, tolerance)
            and citation.year == correction.year
            and abs(float(citation.value) - float(correction.new_value)) <= tolerance + _EPS
            and correction.country in store.read(citation.doc_id)
        )
    except Exception:
        return False
