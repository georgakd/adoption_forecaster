import numpy as np
import pandas as pd
import pytest

from quotas_forecasting.correction.constraints import check_constraints

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
