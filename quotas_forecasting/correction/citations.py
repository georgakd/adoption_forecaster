"""Deterministic citation check: the quote is in the document, and its number, year and unit match the claim."""

import re

from quotas_forecasting.correction.sources import SourceStore
from quotas_forecasting.correction.types import Citation

_YEAR = re.compile(r"(?<![\d.])(?:19|20)\d{2}(?![\d%])")
_PERCENT = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)%")
_SHARE = re.compile(r"(?<![\d.])([01]\.\d+)(?![\d%])")
_EPS = 1e-12


def verify_citation(citation: Citation, store: SourceStore, tolerance: float) -> bool:
    """True only if the doc exists, the quote occurs verbatim, a number in the quote in the cited unit matches ``value``, and the year matches."""
    try:
        if citation.unit not in ("percent", "share") or not citation.quoted_text:
            return False
        if citation.quoted_text not in store.read(citation.doc_id):
            return False
        if citation.year not in {int(y) for y in _YEAR.findall(citation.quoted_text)}:
            return False
        if citation.unit == "percent":
            numbers = [float(n) / 100 for n in _PERCENT.findall(citation.quoted_text)]
        else:
            numbers = [float(n) for n in _SHARE.findall(citation.quoted_text)]
        return any(abs(n - float(citation.value)) <= tolerance + _EPS for n in numbers)
    except Exception:
        return False
