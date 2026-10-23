import copy

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from quotas_forecasting.backtest.compare import GAM_ANCHOR, METHODS, WIN_MARGIN, compare_methods, decide, gam_anchor_robustness, method_table
from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.constraints import check_constraints
from quotas_forecasting.correction.evaluation import _mae
from quotas_forecasting.forecasting.predictors import get_predictor, prediction_method
from quotas_forecasting.synthetic.benchmark import build_benchmark


@pytest.fixture(scope="module")
def tables():
    return [build_benchmark(0, "education", n_countries=4), build_benchmark(0, "age_gender", n_countries=2)]


def test_compare_methods_is_deterministic_and_has_one_row_per_module_method(tables):
    pytest.importorskip("pygam")
    first = compare_methods(tables, list(METHODS), origins=[2024])
    second = compare_methods(tables, list(METHODS), origins=[2024])
    assert_frame_equal(first, second)
    assert len(first) == 2 * len(METHODS)
    assert not first.duplicated(["module", "method"]).any()
    assert set(first["module"]) == {"education", "age_gender"} and set(first["method"]) == set(METHODS)
    assert np.isfinite(first["mae"]).all() and (first["n_tables"] == 1).all()


def test_ensemble_is_the_existing_pipeline_output(tables):
    result = compare_methods(tables, ["ensemble"], origins=[2024]).set_index("module")
    for bench in tables:
        assert result.loc[bench.module, "mae"] == pytest.approx(_mae(bench.baseline, bench.truth))
        assert_frame_equal(method_table(bench, "ensemble"), bench.baseline)


@pytest.mark.parametrize("method", ["gam", "entropy_proxy", "kernel_proxy"])
def test_method_tables_keep_history_and_satisfy_hard_constraints(tables, method):
    if method == "gam":
        pytest.importorskip("pygam")
    for bench in tables:
        table = method_table(bench, method)
        history = ~bench.baseline["is_forecast"]
        assert_frame_equal(table[history], bench.baseline[history])
        assert table.loc[~history, "value"].notna().all()
        violations = [v for v in check_constraints(bench.baseline, table, bench.module, 1.0) if v.kind in ("bounds", "sum_to_one")]
        assert violations == []


def test_compare_methods_rejects_unknown_method_and_missing_origin(tables):
    with pytest.raises(ValueError):
        compare_methods(tables, ["prophet"], origins=[2024])
    with pytest.raises(ValueError):
        compare_methods(tables, ["ensemble"], origins=[2019])


def test_gam_anchor_robustness_scores_both_anchor_choices(tables):
    pytest.importorskip("pygam")
    small = [tables[0]]  # the education table keeps this fast
    result = gam_anchor_robustness(small, origins=[2024])
    assert list(result.columns[:3]) == ["module", "anchor", "n_tables"]
    assert sorted(result["anchor"]) == [False, True] and (result["module"] == "education").all()
    assert_frame_equal(result, gam_anchor_robustness(small, origins=[2024]))
    configured = result.set_index("anchor").loc[GAM_ANCHOR, "mae"]
    assert configured == pytest.approx(compare_methods(small, ["gam"], origins=[2024])["mae"].iloc[0])


def test_win_margin_is_stated_before_results():
    assert WIN_MARGIN == 0.05


def test_decide_requires_the_margin_on_both_benchmarks():
    def frame(gam_mae, proxy_mae):
        return pd.DataFrame({"module": ["education"] * 3, "method": ["ensemble", "gam", "kernel_proxy"], "mae": [0.010, gam_mae, proxy_mae]})

    result = decide(frame(0.0094, 0.0094), frame(0.0094, 0.0096)).set_index("method")
    assert bool(result.loc["gam", "wins"]) and not bool(result.loc["kernel_proxy", "wins"])
    assert result.loc["gam", "gain_real"] == pytest.approx(0.06)


def test_default_method_leaves_goldens_unchanged():
    """The default is "ensemble"; the Task 2 golden tests (tests/e2e/test_pipeline_local.py) run untouched with it."""
    config = load_config()
    assert config["prediction"]["method"] == "ensemble"
    assert prediction_method(config) == "ensemble"
    legacy = copy.deepcopy(config)
    del legacy["prediction"]["method"]  # a config written before the key existed still means the ensemble
    assert prediction_method(legacy) == "ensemble"
    bogus = copy.deepcopy(config)
    bogus["prediction"]["method"] = "prophet"
    with pytest.raises(ValueError):
        get_predictor("education", config=bogus)
