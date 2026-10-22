import numpy as np
import pandas as pd
import pytest

from quotas_forecasting.forecasting.experimental.balancing import (
    InfeasibleBalanceError,
    entropy_balance_weights,
    kernel_balance_weights,
    proxy_forecast,
)


def _donors(seed=0, n=30, k=2):
    return np.random.default_rng(seed).normal(size=(n, k)) * np.array([0.2, 0.01])[:k] + np.array([0.6, 0.02])[:k]


@pytest.mark.parametrize("seed", range(3))
def test_entropy_weights_are_nonnegative_sum_to_one_and_match_target_means(seed):
    donors = _donors(seed)
    target = donors[:10].mean(axis=0)  # strictly inside the hull
    w = entropy_balance_weights(donors, target)
    assert w.shape == (len(donors),)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1) < 1e-9
    np.testing.assert_allclose(w @ donors, target, atol=1e-6)


def test_entropy_weights_are_uniform_when_target_is_the_donor_mean():
    donors = _donors(1)
    w = entropy_balance_weights(donors, donors.mean(axis=0))
    np.testing.assert_allclose(w, np.full(len(donors), 1 / len(donors)), atol=1e-6)


def test_entropy_raises_when_target_outside_hull():
    donors = _donors(2)
    with pytest.raises(InfeasibleBalanceError):
        entropy_balance_weights(donors, donors.max(axis=0) + 0.1)
    assert issubclass(InfeasibleBalanceError, ValueError)


@pytest.mark.parametrize("length_scale", [None, 0.5, 3.0])
def test_kernel_weights_lie_on_simplex(length_scale):
    donors = _donors(3)
    w = kernel_balance_weights(donors, np.array([0.9, 0.05]), length_scale=length_scale)  # also works outside the hull
    assert w.shape == (len(donors),)
    assert np.all(w >= 0)
    assert abs(w.sum() - 1) < 1e-9


def test_kernel_weights_concentrate_on_a_donor_equal_to_the_target():
    donors = _donors(4)
    w = kernel_balance_weights(donors, donors[7], length_scale=0.1)
    assert np.argmax(w) == 7 and w[7] > 0.9


@pytest.mark.parametrize("method", ["entropy", "kernel"])
def test_proxy_forecast_shifts_donor_paths_to_the_target_level(method):
    years = list(range(2010, 2023))
    # every donor grows by exactly 0.02 a year, so any weighting gives a +0.02/+0.04 path from the target's last value
    donors = pd.DataFrame([[0.1 * i + 0.02 * k for k in range(len(years))] for i in range(1, 6)], columns=years, index=list("abcde"))
    target = pd.Series([0.25 + 0.02 * k for k in range(8)], index=years[:8])
    out = proxy_forecast(target, donors, method)
    assert list(out.index) == [years[7] + 1, years[7] + 2]
    np.testing.assert_allclose(out.values, [target.iloc[-1] + 0.02, target.iloc[-1] + 0.04], atol=1e-6)


def test_proxy_forecast_is_clipped_to_unit_interval_and_rejects_unknown_method():
    years = list(range(2010, 2020))
    donors = pd.DataFrame([[min(1.0, 0.5 + 0.1 * k) for k in range(len(years))]] * 3, columns=years, index=list("abc"))
    target = pd.Series([0.95, 0.97, 0.99, 1.0], index=years[:4])
    out = proxy_forecast(target, donors, "kernel")
    assert out.between(0, 1).all()
    with pytest.raises(ValueError):
        proxy_forecast(target, donors, "lasso")
