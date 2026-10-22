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
