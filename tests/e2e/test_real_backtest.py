"""Real-data backtest: the real Eurostat fetcher against the GCS emulator, cached inputs, the real pipeline cut at cut_year.

Skipped unless QF_REALDATA=1 and the emulator env vars are set (see docker-compose.emulators.yml). Values come from the live
Eurostat API on the day the cache was filled, so these tests assert shapes, ranges and structure, never exact numbers.
"""

import copy
import math
import os
import shutil

import pytest
import requests

from quotas_forecasting.backtest import eurostat_backtest
from quotas_forecasting.backtest.eurostat_backtest import (
    DEFAULT_CACHE_DIR,
    MissingInputFilesError,
    build_real_backtest,
    cached_input_files,
    expected_input_files,
    fetch_eurostat_inputs,
)
from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.agent import run_correction
from quotas_forecasting.correction.constraints import check_constraints
from quotas_forecasting.correction.evaluation import evaluate
from quotas_forecasting.correction.llm import LLMResponse, ScriptedClient
from quotas_forecasting.correction.sources import SyntheticSourceStore

CUT_YEAR = 2022
PRED_YEAR = 2024

pytestmark = [
    pytest.mark.realdata,
    pytest.mark.skipif(
        not (os.environ.get("QF_REALDATA") == "1" and os.environ.get("STORAGE_EMULATOR_HOST") and os.environ.get("PUBSUB_EMULATOR_HOST")),
        reason="needs QF_REALDATA=1, STORAGE_EMULATOR_HOST and PUBSUB_EMULATOR_HOST (see docker-compose.emulators.yml)",
    ),
]


def _no_network(*args, **kwargs):
    raise AssertionError("network access attempted")


@pytest.fixture(scope="module")
def config():
    return copy.deepcopy(load_config())


@pytest.fixture(scope="module")
def cache(config):
    return fetch_eurostat_inputs(DEFAULT_CACHE_DIR, PRED_YEAR, config)


@pytest.fixture(scope="module", params=["education", "age_gender"])
def bench(request, cache):
    return build_real_backtest(cache, request.param, cut_year=CUT_YEAR, pred_year=PRED_YEAR, seed=0)


def test_fetch_populates_cache_with_all_expected_input_files(cache, config):
    assert cache == DEFAULT_CACHE_DIR
    found = cached_input_files(cache, config)
    assert set(found) == set(expected_input_files(config))
    for (namespace, base), path in found.items():
        assert path.parent == cache / namespace / "input" and path.name.startswith(f"{base}_") and path.stat().st_size > 0


def test_second_fetch_uses_cache_without_network(cache, config, monkeypatch):
    monkeypatch.setattr(requests, "get", _no_network)
    monkeypatch.setattr(eurostat_backtest.storage, "Client", _no_network)
    assert fetch_eurostat_inputs(cache, PRED_YEAR, config) == cache


def test_backtest_baseline_error_is_finite_and_sane(bench):
    report = evaluate(bench, bench.baseline, [], SyntheticSourceStore(bench.docs), load_config())
    assert bench.world is None and (bench.last_year, bench.pred_year) == (CUT_YEAR, PRED_YEAR)
    assert math.isfinite(report.mae_before) and 0 < report.mae_before < 0.2
    assert bench.truth["is_forecast"].all() and bench.truth["year"].min() > CUT_YEAR
    assert bench.truth.dropna(subset=["value"])["value"].between(0, 1).all()
    assert bench.truth["value"].notna().mean() > 0.8  # gaps (no published value) exist but are rare
    assert bench.docs and all(d.year > CUT_YEAR for d in bench.docs)


def test_backtest_runs_through_evaluate_with_stub_agent(bench):
    config = load_config()
    store = SyntheticSourceStore(bench.docs)
    llm = ScriptedClient([LLMResponse(text='{"suspects": []}')])
    result = run_correction(bench.baseline, bench.module, store, llm, config)
    report = evaluate(bench, result.corrected, result.corrections, store, config)
    # the real pipeline's own forecast can already break a hard constraint (e.g. yoy_cap after a jump in the last observed year):
    # that is reported as baseline_violations; proposing nothing introduces no violation
    inherent = check_constraints(bench.baseline, bench.baseline, bench.module, config["correction"]["max_yoy_change"])
    assert result.error is None and report.n_corrections == 0
    assert report.mae_after == report.mae_before and report.hallucinated_citations == 0
    assert report.violations == 0 and report.baseline_violations == len(inherent)


def test_missing_expected_file_raises_with_its_name(cache, config, tmp_path, monkeypatch):
    copy_dir = tmp_path / "cache"
    shutil.copytree(cache, copy_dir)
    (namespace, base), path = sorted(cached_input_files(copy_dir, config).items())[0]
    (copy_dir / namespace / "input" / path.name).unlink()
    before = sorted((str(p.relative_to(copy_dir)), p.read_bytes()) for p in copy_dir.rglob("*") if p.is_file())
    monkeypatch.setattr(requests, "get", _no_network)  # the real fetcher cannot reach Eurostat, so it produces nothing this run
    with pytest.raises(MissingInputFilesError) as exc:
        fetch_eurostat_inputs(copy_dir, PRED_YEAR, config)
    # every expected file is named, including the ones still cached from an earlier fetch: stale copies do not count as produced
    assert base in str(exc.value) and exc.value.missing == [f"{ns}/{b}" for ns, b in expected_input_files(config)]
    assert sorted((str(p.relative_to(copy_dir)), p.read_bytes()) for p in copy_dir.rglob("*") if p.is_file()) == before
