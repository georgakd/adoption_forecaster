import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pygam")

from quotas_forecasting.forecasting.experimental.gam import gam_forecast

YEARS = np.arange(2004, 2023)


def _logistic(years, midpoint=2012.0, rate=0.3, lo=0.2, hi=0.9):
    return lo + (hi - lo) / (1 + np.exp(-rate * (years - midpoint)))


@pytest.mark.parametrize("seed", range(5))
def test_gam_stays_in_unit_interval_and_is_monotone(seed):
    rng = np.random.default_rng(seed)
    # noisy series hugging both bounds, including exact 0 and 1 values
    values = np.clip(np.linspace(-0.05, 1.05, len(YEARS)) + rng.normal(0, 0.05, len(YEARS)), 0, 1)
    out = gam_forecast(pd.Series(values, index=YEARS), horizon=5, monotone=True)
    assert out.shape == (5,)
    assert np.all((out >= 0) & (out <= 1))
    assert np.all(np.diff(out) >= 0)


def test_gam_handles_missing_values_and_short_series():
    series = pd.Series([0.5, np.nan, 0.55, 0.6, np.nan], index=[2017, 2018, 2019, 2020, 2021])
    out = gam_forecast(series, horizon=2)
    assert out.shape == (2,) and np.all(np.isfinite(out)) and np.all((out >= 0) & (out <= 1))


def test_gam_recovers_logistic_truth_within_tolerance():
    rng = np.random.default_rng(0)
    full_years = np.arange(2004, 2025)
    truth = _logistic(full_years)
    observed = pd.Series(truth[:-2] + rng.normal(0, 0.003, len(full_years) - 2), index=full_years[:-2])
    out = gam_forecast(observed, horizon=2)
    assert np.max(np.abs(out - truth[-2:])) < 0.02
