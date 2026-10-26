"""Shared types for the correction package."""

from dataclasses import dataclass

import pandas as pd

# Long-format forecast table with columns: country (str), segment (str), year (int), value (float, NaN allowed), is_forecast (bool).
ForecastTable = pd.DataFrame

TABLE_COLUMNS = ["country", "segment", "year", "value", "is_forecast"]


@dataclass
class Violation:
    """A hard-constraint violation. kind is one of: bounds, sum_to_one, historical_changed, yoy_cap."""

    kind: str
    country: str
    segment: str
    year: int
    detail: str


@dataclass
class Citation:
    """Evidence for a correction: the verbatim quote, the share-scale value it states (``unit`` is the quote's unit), and its year."""

    doc_id: str
    quoted_text: str
    value: float
    year: int
    unit: str  # "percent" or "share"


@dataclass
class Correction:
    country: str
    segment: str
    year: int
    old_value: float
    new_value: float
    citation: Citation
    confidence: float
    reason: str


QUOTE_RECORD_CHARS = 200


def correction_record(correction: Correction, module: str) -> dict:
    """JSON-safe summary of an accepted correction for result files and logs (the quote is cut to QUOTE_RECORD_CHARS)."""
    return {
        "module": module,
        "country": correction.country,
        "segment": correction.segment,
        "year": int(correction.year),
        "old_value": float(correction.old_value),
        "new_value": float(correction.new_value),
        "doc_id": correction.citation.doc_id,
        "quote": correction.citation.quoted_text[:QUOTE_RECORD_CHARS],
        "confidence": float(correction.confidence),
    }
