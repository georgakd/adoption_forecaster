"""Evidence for docs/MODEL_REVIEW.md: characterization tests that confirm or refute each suspect in the existing models.

These pin the CURRENT behaviour, including behaviour the review calls a bug; a test here failing after a fix means the review's
verdict needs updating, not that the fix is wrong.
"""

import copy
import random
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quotas_forecasting.forecasting.quotas_forecasting as qf_module
from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.data_handlers.missing_data_handlers import MissingDataHandler
from quotas_forecasting.data_handlers.proxy_handlers import ProxyHandler
from quotas_forecasting.forecasting.predictors import EducationPredictor
from quotas_forecasting.forecasting.utils import (
    adjust_100prc_predictions_cases,
    fit_predict_logistic,
    fit_predict_theta,
    predict_random_walk,
    train_lasso,
)
from quotas_forecasting.synthetic.benchmark import run_pipeline_on_inputs
from quotas_forecasting.synthetic.world import generate_world
from quotas_forecasting.synthetic.writers import write_age_gender_inputs, write_education_inputs

YEARS = list(range(2004, 2023))


@pytest.fixture(scope="module")
def config():
    return copy.deepcopy(load_config())


@pytest.fixture(scope="module")
def world():
    return generate_world(7)


# (a) predictors.py:90-91 applies the same backward interpolation twice


@pytest.mark.parametrize("seed", range(20))
def test_a_second_backward_interpolation_changes_nothing(seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.random((6, 12)), columns=range(2010, 2022))
    df = df.mask(rng.random(df.shape) < 0.35)  # leading, interior and trailing gaps
    once = df.apply(lambda x: x.interpolate(limit_direction="backward"), axis=1)
    twice = once.apply(lambda x: x.interpolate(limit_direction="backward"), axis=1)
    pd.testing.assert_frame_equal(once, twice)


def _raise(*args, **kwargs):
    raise AssertionError("dead code was called")


def test_a_and_c_lasso_paths_are_never_called_by_the_default_pipeline(world, config, monkeypatch):
    for name in ("train_test_split_", "train_test_split_UK", "Lasso", "combine_regressors", "backtest_UK"):
        monkeypatch.setattr(EducationPredictor, name, _raise)
    monkeypatch.setattr(ProxyHandler, "apply_LASSO_for_the_three_countries_with_missing_data", _raise)
    monkeypatch.setattr(qf_module, "train_lasso", _raise)
    with tempfile.TemporaryDirectory() as d:
        write_education_inputs(world, Path(d) / "Education" / "input")
        write_age_gender_inputs(world, Path(d) / "Age_Gender" / "input")
        assert len(run_pipeline_on_inputs(Path(d), "education", 2024, 2026, config=config))
        assert len(run_pipeline_on_inputs(Path(d), "age_gender", 2024, 2026, config=config))


# (c) what the LASSO proxy path would do if target_markets were set


def _age_gender_with_gap(world, d):
    path = write_age_gender_inputs(world, Path(d) / "Age_Gender" / "input")[0]
    df = pd.read_csv(path)
    df.loc[df["country"] == "Austria", ["2023", "2024"]] = np.nan  # the target market misses its last two years
    df.to_csv(path, index=False)


def test_c_age_gender_lasso_path_crashes_when_enabled(world, config):
    enabled = copy.deepcopy(config)
    enabled["target_markets"] = {"Austria": ["Belgium", "Denmark", "France", "Germany"]}
    with tempfile.TemporaryDirectory() as d:
        _age_gender_with_gap(world, d)
        with pytest.raises(TypeError, match="not subscriptable"):  # backtest_model(lasso, X, y) is called without config
            run_pipeline_on_inputs(Path(d), "age_gender", 2024, 2026, config=enabled)


def test_c_age_gender_lasso_path_writes_one_percent_when_the_crash_is_bypassed(world, config, monkeypatch):
    enabled = copy.deepcopy(config)
    enabled["target_markets"] = {"Austria": ["Belgium", "Denmark", "France", "Germany"]}
    monkeypatch.setattr(qf_module, "backtest_model", lambda *a, **k: (0.0, 0.0, 0.0))
    with tempfile.TemporaryDirectory() as d:
        _age_gender_with_gap(world, d)
        table = run_pipeline_on_inputs(Path(d), "age_gender", 2024, 2026, config=enabled)
    filled = table[(table["country"] == "Austria") & table["year"].isin([2023, 2024])]["value"]
    # the data are still in percent when the LASSO prediction is clipped to [0, 1]; after the later /100 every filled cell is 0.01
    assert np.allclose(filled, 0.01)


def test_c_education_lasso_path_needs_exactly_three_hardcoded_targets(config):
    idx = pd.MultiIndex.from_product([["A", "B", "C", "D"], ["secondary", "tertiary"]], names=["country", "education"])
    df = pd.DataFrame(np.random.default_rng(0).random((8, 6)), index=idx, columns=range(2017, 2023))
    with pytest.raises(IndexError):
        ProxyHandler(config).apply_LASSO_for_the_three_countries_with_missing_data(df, ["A"], ["secondary", "tertiary"], 2022)


# (d) clipping to [0, 1] versus the logistic ceiling K_max


def test_d_logistic_component_cannot_reach_a_saturated_series(config):
    saturated = pd.Series([1.0] * 6, index=range(2017, 2023))
    out = fit_predict_logistic(saturated, np.arange(2023, 2025), 2022, config)
    assert (out <= config["modules"]["age_gender"]["settings"]["K_max"] + 1e-12).all() and (out < 1).all()


def test_d_ceiling_rule_b_random_factor_is_one_constant(config):
    random.seed(42)
    factor = random.uniform(1, 1 / config["modules"]["age_gender"]["settings"]["K_max"])
    assert factor == pytest.approx(1.0032132, abs=1e-7)  # the seed is reset before every row, so every row gets this factor


def test_d_ceiling_rule_needs_three_observations_per_series(config):
    train = pd.DataFrame({2019: [np.nan], 2020: [0.9], 2021: [0.95]}, index=["X 16_24 male"])
    preds = pd.DataFrame({"country_age": ["X 16_24"], "gender": ["male"], "2023_logistic": [0.9], "2023_rw": [0.9], "2023_theta": [0.9]})
    for col in ("2022_logistic", "2022_rw", "2022_theta"):
        preds[col] = 0.9
    with pytest.raises(KeyError):  # this is why the real age_gender backtest cannot run at origins 2019 and 2020
        adjust_100prc_predictions_cases(preds, train, 2023, config)


# (e) .replace(0, np.nan) on the Education ratios


def _abs_users(level3_missing_year=None):
    idx = pd.MultiIndex.from_product([["X"], [1, 2, 3]], names=["country", "education"])
    df = pd.DataFrame([[10.0, 11.0, 12.0], [50.0, 50.0, 50.0], [30.0, 32.0, 34.0]], index=idx, columns=[2020, 2021, 2022])
    if level3_missing_year is not None:
        df.loc[("X", 3), level3_missing_year] = np.nan
    return df


def test_e_replace_zero_undoes_the_groupby_sum_of_an_all_missing_year():
    users = _abs_users()
    users[2021] = np.nan
    merged = MissingDataHandler.merge_education_levels(MissingDataHandler.calculate_internet_users_ratio(users)).astype(float)
    assert (merged[2021] == 0).all()  # groupby().sum() turns all-NaN into 0 ...
    assert merged.replace(0, np.nan)[2021].isna().all()  # ... and the replace restores the gap


def test_e_a_missing_tertiary_level_makes_secondary_exactly_one():
    merged = MissingDataHandler.merge_education_levels(MissingDataHandler.calculate_internet_users_ratio(_abs_users(2021)))
    merged = merged.astype(float).replace(0, np.nan)
    assert merged.loc[("X", "secondary"), 2021] == 1.0 and np.isnan(merged.loc[("X", "tertiary"), 2021])


# (f) LASSO CV with very few proxy rows


def test_f_one_or_two_proxy_rows_cannot_be_cross_validated(config):
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):  # age_gender path: cv = n - 1 = 1
        train_lasso(pd.DataFrame(rng.random((2, 5))), pd.Series(rng.random(2)), config)
    idx = pd.MultiIndex.from_product([["T", "P"], ["secondary", "tertiary"]], names=["country", "education"])
    df = pd.DataFrame(rng.random((4, 6)), index=idx, columns=range(2017, 2023))
    df.loc[("T", "tertiary"), 2022] = np.nan
    proxies = {"secondary": [{"P"}], "tertiary": [{"P"}]}
    with pytest.raises(ValueError):  # education path: KFold(n_splits=min(5, 1))
        ProxyHandler(config).extrapolate_with_Lasso(df, ["T"], {"secondary": [2022], "tertiary": [2021]}, proxies, 2022, ["tertiary"])


# (g) secondary + tertiary = 1 by construction; the per-level forecasts already sum to 1


@pytest.mark.parametrize("seed", range(5))
def test_g_random_walk_and_theta_forecasts_of_the_two_shares_sum_to_one(config, seed):
    rng = np.random.default_rng(seed)
    tertiary = pd.Series(np.clip(0.3 + 0.01 * np.arange(len(YEARS)) + rng.normal(0, 0.004, len(YEARS)), 0, 1), index=YEARS)
    assert predict_random_walk(tertiary, config) + predict_random_walk(1 - tertiary, config) == pytest.approx(1, abs=1e-12)
    pred_idx = YEARS + [2023, 2024]
    theta = fit_predict_theta(tertiary, pred_idx, config, 2022).values + fit_predict_theta(1 - tertiary, pred_idx, config, 2022).values
    np.testing.assert_allclose(theta, 1, atol=1e-9)


# (i) the two-year horizon is hard-coded in the Education predictor


def test_i_education_predictor_cannot_forecast_three_years_ahead(config):
    idx = pd.MultiIndex.from_product([["X"], ["secondary", "tertiary"]], names=["country", "education"])
    ter = 0.3 + 0.01 * np.arange(len(YEARS))
    df = pd.DataFrame([1 - ter, ter], index=idx, columns=YEARS)
    EducationPredictor(pred_year=2024, config=config).predict(df.copy(), 2022, 2024)  # horizon 2 works
    with pytest.raises(KeyError):
        EducationPredictor(pred_year=2025, config=config).predict(df.copy(), 2022, 2025)
