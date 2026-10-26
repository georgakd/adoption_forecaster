import numpy as np
import pandas as pd
import pytest

from adoption_forecaster.correction.constraints import check_constraints

LAST_YEAR = 2024
MAX_YOY = 0.10


def _table(rows):
    df = pd.DataFrame(rows, columns=["country", "segment", "year", "value"])
    df["is_forecast"] = df["year"] > LAST_YEAR
    return df


def _education(secondary_2025=0.5, secondary_2026=0.52):
    rows = []
    for year, sec in [(2023, 0.5), (2024, 0.5), (2025, secondary_2025), (2026, secondary_2026)]:
        rows += [("A", "secondary", year, sec), ("A", "tertiary", year, 1 - sec)]
    return _table(rows)


def _set(table, segment, year, value):
    out = table.copy()
    out.loc[(out["segment"] == segment) & (out["year"] == year), "value"] = value
    return out


def _kinds(violations):
    return [v.kind for v in violations]


def test_clean_correction_has_no_violations():
    assert check_constraints(_education(), _education(0.55, 0.6), "education", MAX_YOY) == []


def test_bounds_violation():
    baseline = _education()
    corrected = _set(_set(baseline, "secondary", 2025, 1.2), "tertiary", 2025, -0.2)
    violations = check_constraints(baseline, corrected, "age_gender", 10.0)
    assert [(v.kind, v.segment, v.year) for v in violations] == [("bounds", "secondary", 2025), ("bounds", "tertiary", 2025)]


def test_sum_to_one_violation_education_only():
    baseline = _education()
    corrected = _set(baseline, "secondary", 2025, 0.55)
    assert _kinds(check_constraints(baseline, corrected, "education", MAX_YOY)) == ["sum_to_one"]
    assert check_constraints(baseline, corrected, "age_gender", MAX_YOY) == []


def test_historical_change_detected():
    baseline = _education()
    corrected = _set(baseline, "secondary", 2024, 0.51)
    violations = check_constraints(baseline, corrected, "age_gender", MAX_YOY)
    assert [(v.kind, v.year) for v in violations] == [("historical_changed", 2024)]


@pytest.mark.parametrize("delta,expected", [(0.10, []), (0.10 + 1e-6, ["yoy_cap"])])
def test_yoy_cap_violation_at_boundary(delta, expected):
    baseline = _table([("A", "x", 2024, 0.5), ("A", "x", 2025, 0.5)])
    corrected = _set(baseline, "x", 2025, 0.5 + delta)
    assert _kinds(check_constraints(baseline, corrected, "age_gender", MAX_YOY)) == expected


def test_nan_values_never_violate():
    baseline = _table([("A", "x", 2023, np.nan), ("A", "x", 2024, np.nan), ("A", "x", 2025, np.nan), ("A", "y", 2025, 0.5)])
    assert check_constraints(baseline, baseline.copy(), "age_gender", MAX_YOY) == []
    removed = _set(baseline, "y", 2025, np.nan)
    violations = check_constraints(baseline, removed, "age_gender", MAX_YOY)
    assert [(v.kind, v.segment, v.year) for v in violations] == [("bounds", "y", 2025)]


def test_verifier_is_pure():
    baseline, corrected = _education(), _education(0.9, 0.2)
    before = corrected.copy()
    first = check_constraints(baseline, corrected, "education", MAX_YOY)
    assert first == check_constraints(baseline, corrected, "education", MAX_YOY) and first
    pd.testing.assert_frame_equal(corrected, before)


def test_out_of_bounds_value_written_to_historical_cell_is_a_historical_change():
    baseline = _education()
    corrected = _set(baseline, "secondary", 2024, 1.5)
    violations = check_constraints(baseline, corrected, "age_gender", MAX_YOY)
    # the changed historical value also feeds the yoy check of the first forecast year
    assert [(v.kind, v.segment, v.year) for v in violations] == [("historical_changed", "secondary", 2024), ("yoy_cap", "secondary", 2025)]


def test_unchanged_out_of_bounds_historical_cell_is_not_a_violation():
    baseline = _table([("A", "x", 2024, 1.05), ("A", "x", 2025, 1.0)])
    assert check_constraints(baseline, baseline.copy(), "age_gender", MAX_YOY) == []


def test_yoy_cap_applies_across_last_year_boundary():
    baseline = _table([("A", "x", 2024, 0.5), ("A", "x", 2025, 0.5), ("A", "x", 2026, 0.5)])
    corrected = _set(baseline, "x", 2025, 0.65)
    violations = check_constraints(baseline, corrected, "age_gender", MAX_YOY)
    assert [(v.kind, v.year) for v in violations] == [("yoy_cap", 2025), ("yoy_cap", 2026)]


def test_country_missing_from_corrected_is_reported():
    baseline = _table([("A", "x", 2024, 0.5), ("A", "x", 2025, 0.5), ("B", "x", 2024, 0.4), ("B", "x", 2025, 0.4)])
    corrected = baseline[baseline["country"] == "A"].copy()
    violations = check_constraints(baseline, corrected, "age_gender", MAX_YOY)
    assert [(v.kind, v.country, v.year) for v in violations] == [("bounds", "B", 2024), ("bounds", "B", 2025)]


def test_extra_rows_in_corrected():
    baseline = _table([("A", "x", 2024, 0.5), ("A", "x", 2025, 0.5)])
    extra_historical = pd.concat([baseline, _table([("C", "x", 2024, 0.5)])], ignore_index=True)
    violations = check_constraints(baseline, extra_historical, "age_gender", MAX_YOY)
    assert [(v.kind, v.country, v.year) for v in violations] == [("historical_changed", "C", 2024)]
    # an extra forecast row is treated like a filled NaN cell: allowed if in bounds (the agent tool never lets one through)
    extra_forecast = pd.concat([baseline, _table([("C", "x", 2025, 0.5)])], ignore_index=True)
    assert check_constraints(baseline, extra_forecast, "age_gender", MAX_YOY) == []
    out_of_bounds = pd.concat([baseline, _table([("C", "x", 2025, 1.5)])], ignore_index=True)
    assert _kinds(check_constraints(baseline, out_of_bounds, "age_gender", MAX_YOY)) == ["bounds"]


def test_nan_baseline_forecast_cell_filled_is_allowed_by_verifier():
    baseline = _table([("A", "x", 2024, 0.5), ("A", "x", 2025, np.nan)])
    assert check_constraints(baseline, _set(baseline, "x", 2025, 0.55), "age_gender", MAX_YOY) == []


def test_unchanged_baseline_exceeding_yoy_cap_is_reported():
    baseline = _table([("A", "x", 2024, 0.5), ("A", "x", 2025, 0.7)])
    violations = check_constraints(baseline, baseline.copy(), "age_gender", MAX_YOY)
    assert [(v.kind, v.year) for v in violations] == [("yoy_cap", 2025)]
