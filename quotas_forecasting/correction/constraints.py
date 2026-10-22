"""Deterministic hard-constraint verifier for corrected forecast tables."""

import numpy as np
import pandas as pd

from quotas_forecasting.correction.types import ForecastTable, Violation

SUM_TOLERANCE = 1e-9
YOY_EPSILON = 1e-12  # float noise only: exactly max_yoy_change passes, max_yoy_change + 1e-6 fails


def check_constraints(baseline: ForecastTable, corrected: ForecastTable, module: str, max_yoy_change: float) -> list[Violation]:
    """Return all hard-constraint violations of ``corrected`` relative to ``baseline`` (pure, sorted deterministically).

    Rules:
      - bounds: corrected forecast values must be in [0, 1]; a NaN where the baseline had a value is also a ``bounds`` violation.
      - historical_changed: non-forecast cells must equal the baseline (NaN equals NaN); a value where baseline was NaN counts as a change.
      - sum_to_one (education only): secondary + tertiary == 1 within 1e-9 for forecast years where both are present.
      - yoy_cap: |corrected(y) - corrected(y-1)| <= max_yoy_change for forecast cells, when the previous year is present.
    A value where the baseline was NaN in a forecast year is allowed as long as it passes the rules above.
    """
    key = ["country", "segment", "year"]
    merged = baseline[key + ["value", "is_forecast"]].merge(corrected[key + ["value"]], on=key, how="outer", suffixes=("_base", "_corr"))
    historical_years = baseline.loc[~baseline["is_forecast"], "year"]
    last_year = historical_years.max() if len(historical_years) else -np.inf
    # cells present only in corrected take is_forecast from the baseline's last historical year
    merged["is_forecast"] = merged["is_forecast"].where(merged["is_forecast"].notna(), merged["year"] > last_year).astype(bool)
    merged = merged.sort_values(key).reset_index(drop=True)
    violations: list[Violation] = []

    def add(kind, row, detail):
        violations.append(Violation(kind, str(row.country), str(row.segment), int(row.year), detail))

    for row in merged.itertuples(index=False):
        base, corr = row.value_base, row.value_corr
        if pd.isna(corr):
            if not pd.isna(base):
                add("bounds", row, "value removed (NaN) where baseline had a value")
            continue
        if not row.is_forecast:
            if pd.isna(base) or corr != base:
                add("historical_changed", row, f"historical value changed from {base} to {corr}")
            continue
        if corr < 0 or corr > 1:
            add("bounds", row, f"value {corr} outside [0, 1]")

    lookup = {(r.country, r.segment, r.year): r.value_corr for r in merged.itertuples(index=False)}
    for row in merged.itertuples(index=False):
        if not row.is_forecast or pd.isna(row.value_corr):
            continue
        prev = lookup.get((row.country, row.segment, row.year - 1), np.nan)
        if not pd.isna(prev) and abs(row.value_corr - prev) > max_yoy_change + YOY_EPSILON:
            add("yoy_cap", row, f"change {row.value_corr - prev:+.6f} from {row.year - 1} exceeds {max_yoy_change}")

    if module == "education":
        for (country, year), group in merged[merged["is_forecast"]].groupby(["country", "year"]):
            values = group.set_index("segment")["value_corr"]
            if "secondary" in values.index and "tertiary" in values.index and not values[["secondary", "tertiary"]].isna().any():
                total = values["secondary"] + values["tertiary"]
                if abs(total - 1) > SUM_TOLERANCE:
                    violations.append(Violation("sum_to_one", str(country), "secondary+tertiary", int(year), f"sum is {total}, expected 1"))

    return sorted(violations, key=lambda v: (v.country, v.segment, v.year, v.kind))
