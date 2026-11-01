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

import adoption_forecaster.forecasting.adoption_forecaster as qf_module
from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.data_handlers.missing_data_handlers import MissingDataHandler
from adoption_forecaster.data_handlers.proxy_handlers import ProxyHandler
from adoption_forecaster.forecasting.predictors import EducationPredictor
from adoption_forecaster.forecasting.utils import (
    adjust_100prc_predictions_cases,
    fit_predict_logistic,
    fit_predict_theta,
    predict_random_walk,
    train_lasso,
)
from adoption_forecaster.synthetic.benchmark import run_pipeline_on_inputs
from adoption_forecaster.synthetic.world import generate_world
from adoption_forecaster.synthetic.writers import write_age_gender_inputs, write_education_inputs

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


# (c) the LASSO proxy path when target_markets is set


def _age_gender_with_gap(world, d):
    path = write_age_gender_inputs(world, Path(d) / "Age_Gender" / "input")[0]
    df = pd.read_csv(path)
    truth = df.loc[df["country"] == "Austria", ["2023", "2024"]].to_numpy() / 100
    df.loc[df["country"] == "Austria", ["2023", "2024"]] = np.nan  # the target market misses its last two years
    df.to_csv(path, index=False)
    return truth


def _age_gender_lasso_enabled(config):
    enabled = copy.deepcopy(config)
    enabled["target_markets"] = {"Austria": ["Belgium", "Denmark", "France", "Germany"]}
    return enabled


def test_c_age_gender_lasso_path_fills_the_gap_on_the_share_scale(world, config):
    # (c) fixed: used to raise TypeError (backtest_model called without config), and with that bypassed every filled cell was 0.01
    with tempfile.TemporaryDirectory() as d:
        truth = _age_gender_with_gap(world, d)
        table = run_pipeline_on_inputs(Path(d), "age_gender", 2024, 2026, config=_age_gender_lasso_enabled(config))
    filled = table[(table["country"] == "Austria") & table["year"].isin([2023, 2024])]["value"].to_numpy()
    assert len(filled) == truth.size
    assert ((filled > 0.01) & (filled <= 1)).all()
    assert np.abs(np.sort(filled) - np.sort(truth.ravel())).mean() < 0.1  # same scale as the truth, not percent or 1%


def test_c_age_gender_lasso_target_is_the_predicted_year_not_a_feature(world, config, monkeypatch):
    # (c) fixed: y_train used to be the year before the predicted one, which is also the last column of X_train
    seen = []

    def spy(X_train, y_train, config=None):
        seen.append((list(X_train.columns), y_train.name))
        return train_lasso(X_train, y_train, config=config)

    monkeypatch.setattr(qf_module, "train_lasso", spy)
    with tempfile.TemporaryDirectory() as d:
        _age_gender_with_gap(world, d)
        run_pipeline_on_inputs(Path(d), "age_gender", 2024, 2026, config=_age_gender_lasso_enabled(config))
    assert seen
    assert all(target not in features and int(target) == int(features[-1]) + 1 for features, target in seen)


def _education_with_gap(countries, missing):
    idx = pd.MultiIndex.from_product([countries, ["secondary", "tertiary"]], names=["country", "education"])
    years = list(range(2010, 2023))
    rng = np.random.default_rng(0)
    rows = []
    for k in range(len(countries)):
        ter = 0.2 + 0.01 * (k + 1) * np.arange(len(years)) + rng.normal(0, 0.002, len(years))
        rows += [1 - ter, ter]
    df = pd.DataFrame(rows, index=idx, columns=years)
    for country in missing:
        df.loc[country, 2022] = np.nan
    return df


@pytest.mark.parametrize("targets", [["A"], ["A", "B"], ["A", "B", "C", "D"]])
def test_c_education_lasso_path_takes_any_number_of_targets(config, targets):
    # (c) fixed: used to raise IndexError unless there were exactly three targets (Iceland, North Macedonia, UK by position)
    countries = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    df = _education_with_gap(countries, targets)
    out = ProxyHandler(config).apply_LASSO_for_the_three_countries_with_missing_data(df, targets, ["secondary", "tertiary"], 2022)
    assert out.loc[targets, 2022].notna().all()
    assert (out.loc[targets, 2022].between(0, 1)).all()


# (d) clipping to [0, 1] versus the logistic ceiling K_max


def test_d_logistic_component_cannot_reach_a_saturated_series(config):
    saturated = pd.Series([1.0] * 6, index=range(2017, 2023))
    out = fit_predict_logistic(saturated, np.arange(2023, 2025), 2022, config)
    assert (out <= config["modules"]["age_gender"]["settings"]["K_max"] + 1e-12).all() and (out < 1).all()


def test_d_ceiling_rule_b_random_factor_is_one_constant(config):
    random.seed(42)
    factor = random.uniform(1, 1 / config["modules"]["age_gender"]["settings"]["K_max"])
    assert factor == pytest.approx(1.0032132, abs=1e-7)  # the seed is reset before every row, so every row gets this factor


def _ceiling_preds(value):
    preds = pd.DataFrame({"country_age": ["X 16_24"], "gender": ["male"]})
    for year in (2022, 2023):
        for model in ("logistic", "rw", "theta", "combined"):
            preds[f"{year}_{model}"] = value
    return preds


def test_j_ceiling_rule_leaves_short_series_below_the_ceiling_alone(config):
    # (j) fixed: fewer than three observations used to raise KeyError (real age_gender origins 2019 and 2020 could not run)
    train = pd.DataFrame({2019: [np.nan], 2020: [0.9], 2021: [0.95]}, index=["X 16_24 male"])
    out = adjust_100prc_predictions_cases(_ceiling_preds(0.9), train, 2023, config)
    assert (out[["2022_combined", "2023_combined"]] == 0.9).all(axis=None)


def test_j_ceiling_rule_b_applies_to_a_short_series_at_the_ceiling(config):
    # two observations at 100% are not "the last three years at 100%", so rule (a) cannot apply and rule (b) does
    train = pd.DataFrame({2019: [np.nan], 2020: [1.0], 2021: [1.0]}, index=["X 16_24 male"])
    preds = _ceiling_preds(1.0)
    preds[["2022_logistic", "2023_logistic"]] = config["modules"]["age_gender"]["settings"]["K_max"]
    out = adjust_100prc_predictions_cases(preds, train, 2023, config)
    assert (out[["2022_combined", "2023_combined"]] < 1).all(axis=None)


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


def test_e_a_missing_tertiary_level_makes_the_year_missing_for_both_levels():
    # (e) fixed: secondary used to become exactly 1.0 (Portugal 2006 in the real data)
    merged = MissingDataHandler.merge_education_levels(MissingDataHandler.calculate_internet_users_ratio(_abs_users(2021)))
    merged = merged.astype(float).replace(0, np.nan)
    assert merged[2021].isna().all()
    assert not merged[[2020, 2022]].isna().any(axis=None)


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


# (i) the Education predictor's horizon


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_i_education_predictor_forecasts_the_requested_horizon(config, horizon):
    # (i) fixed: the horizon used to be two years, hard-coded; predict(df, 2022, 2025) raised KeyError
    idx = pd.MultiIndex.from_product([["X"], ["secondary", "tertiary"]], names=["country", "education"])
    ter = 0.3 + 0.01 * np.arange(len(YEARS))
    df = pd.DataFrame([1 - ter, ter], index=idx, columns=YEARS)
    out = EducationPredictor(pred_year=2022 + horizon, config=config).predict(df.copy(), 2022, 2022 + horizon)
    future = out[[2022 + i for i in range(1, horizon + 1)]]
    assert future.notna().all(axis=None)
    np.testing.assert_allclose(future.sum(), 1, atol=1e-10)
    assert (future.loc[("X", "tertiary")].diff().dropna() > 0).all()  # the trend continues year after year
