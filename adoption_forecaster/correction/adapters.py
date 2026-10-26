"""Adapters between the pipeline DataFrames and the long-format ForecastTable.

Accepted stages (the Task 11 hook must call these at one of them):
  - education: the frame returned by ``EducationPredictor.predict`` (index ``(country, education)``, int year columns, NaN for missing).
  - age_gender: the frame returned by ``AgeGenderPredictor.run_with_data`` (columns ``country``, ``Individual type`` and year columns;
    year labels may be ints or strings, NaN for missing), or the same frame after ``fillna(":")``, before ``set_index("country")``.
    Missing cells holding the ``":"`` placeholder become NaN in the table; ``from_table`` restores ``":"`` exactly where the template had it.
"""

import numpy as np
import pandas as pd

from adoption_forecaster.correction.types import TABLE_COLUMNS, ForecastTable

PLACEHOLDER = ":"
AGE_GENDER_ID_COLUMNS = ["country", "Individual type"]


def _check_module(module: str) -> None:
    if module not in ("education", "age_gender"):
        raise ValueError(f"Unknown module: {module}. Must be 'education' or 'age_gender'")


def _wide(df: pd.DataFrame, module: str) -> pd.DataFrame:
    """Return a float frame indexed by (country, segment) with int year columns."""
    if module == "education":
        wide = df.copy()
        wide.index = pd.MultiIndex.from_arrays([wide.index.get_level_values(0), wide.index.get_level_values(1)], names=["country", "segment"])
    else:
        wide = df.set_index(AGE_GENDER_ID_COLUMNS)
        wide.index.names = ["country", "segment"]
    wide = wide.replace(PLACEHOLDER, np.nan).astype(float)
    wide.columns = [int(c) for c in wide.columns]
    return wide


def to_table(df: pd.DataFrame, module: str, last_year: int) -> ForecastTable:
    """Convert a pipeline frame (see module docstring for accepted stages) to a long-format ForecastTable."""
    _check_module(module)
    long = _wide(df, module).stack(future_stack=True).rename("value").reset_index()
    long.columns = ["country", "segment", "year", "value"]
    long["country"] = long["country"].astype(str)
    long["segment"] = long["segment"].astype(str)
    long["year"] = long["year"].astype(int)
    long["value"] = long["value"].astype(float)
    long["is_forecast"] = long["year"] > last_year
    return long[TABLE_COLUMNS].reset_index(drop=True)


def from_table(table: ForecastTable, template: pd.DataFrame, module: str) -> pd.DataFrame:
    """Rebuild a frame with the exact index, columns and dtypes of ``template`` from ``table``.

    Cells that are NaN in the table are written as ``":"`` where the template holds ``":"`` there, otherwise NaN.
    """
    _check_module(module)
    wide = table.pivot(index=["country", "segment"], columns="year", values="value")
    template_wide = _wide(template, module)
    wide = wide.reindex(index=template_wide.index, columns=template_wide.columns)
    values = wide.to_numpy(dtype=float)

    out = template.copy()
    if module == "education":
        for pos, col in enumerate(template.columns):
            out[col] = values[:, pos]
    else:
        for pos, col in enumerate(c for c in template.columns if c not in AGE_GENDER_ID_COLUMNS):
            is_placeholder = (template[col] == PLACEHOLDER).to_numpy()
            column = pd.Series(values[:, pos], index=template.index, dtype=object if is_placeholder.any() else float)
            column[np.isnan(values[:, pos]) & is_placeholder] = PLACEHOLDER
            out[col] = column
    return out.astype(template.dtypes.to_dict())
