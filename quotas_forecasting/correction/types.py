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
