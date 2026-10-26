import copy
import json
import os
import statistics

import numpy as np
import pandas as pd
import pytest

from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.correction.agent import run_correction
from adoption_forecaster.correction.caps import compute_yoy_caps
from adoption_forecaster.correction.constraints import check_constraints
from adoption_forecaster.correction.evaluation import evaluate
from adoption_forecaster.correction.llm import LLMResponse, ScriptedClient
from adoption_forecaster.correction.prompts import planner_system_prompt, researcher_system_prompt
from adoption_forecaster.correction.sources import SyntheticSourceStore
from adoption_forecaster.correction.tools import CorrectionSession
from adoption_forecaster.synthetic.benchmark import Benchmark, build_benchmark
from adoption_forecaster.synthetic.sources import SourceDoc

LAST_YEAR = 2024
COUNTRIES = ["Alpha", "Beta", "Gamma", "Delta"]
MOVE = 0.08  # above the stable segment's cap (3 x 0.02), below the volatile one's (3 x 0.06) and the fixed 0.10


@pytest.fixture(scope="module")
def config():
    return load_config()


def _with_mode(config, mode):
    changed = copy.deepcopy(config)
    changed["correction"]["yoy_cap"]["mode"] = mode
    return changed


def _table(series=None):
    """History 2018-2024 and a flat forecast for 2025-2026; ``stable`` moves 0.02 a year, ``volatile`` jumps 0.06 every year."""
    series = series or {}
    rows = []
    for i, country in enumerate(COUNTRIES):
        stable = {year: 0.4 + 0.01 * i + 0.02 * (year - 2018) for year in range(2018, LAST_YEAR + 1)}
        volatile = {year: 0.5 + 0.01 * i + (0.06 if year % 2 else 0.0) for year in range(2018, LAST_YEAR + 1)}
        for segment, history in [("stable", stable), ("volatile", volatile)]:
            history = series.get((country, segment), history)
            last = history[max(history)]
            for year, value in {**history, 2025: last, 2026: last}.items():
                rows.append((country, segment, year, value, year > LAST_YEAR))
    return pd.DataFrame(rows, columns=["country", "segment", "year", "value", "is_forecast"])


def _value(table, country, segment, year):
    return table.loc[(table["country"] == country) & (table["segment"] == segment) & (table["year"] == year), "value"].item()


def _doc(country, segment, year, value):
    text = f"According to the Office, the share of internet users in {country} reached {value:.3f} in {year}."
    return SourceDoc(f"{country}-{segment}-{year}", "Office", country, year, "share", segment, text, value, title=f"{country} {year}")


def _propose(session, doc):
    citation = {"doc_id": doc.doc_id, "quoted_text": doc.text, "value": doc.value, "year": doc.year, "unit": "share"}
    return session.propose_correction(doc.country, doc.segment, doc.year, doc.value, citation, 0.9, "published figure")


# ---- compute_yoy_caps ----


def test_caps_use_only_historical_cells(config):
    bench = build_benchmark(0, "age_gender", n_countries=5)
    settings = config["correction"]["yoy_cap"]
    before = compute_yoy_caps(bench.baseline, settings, 0.10)
    scrambled = bench.baseline.copy()
    forecast = scrambled["is_forecast"]
    scrambled.loc[forecast, "value"] = np.random.default_rng(0).uniform(0, 1, forecast.sum())
    assert compute_yoy_caps(scrambled, settings, 0.10) == before
    assert set(before) == set(zip(bench.baseline["country"], bench.baseline["segment"]))


def test_volatile_segment_gets_wider_cap_than_stable_one(config):
    caps = compute_yoy_caps(_table(), config["correction"]["yoy_cap"], 0.10)
    for country in COUNTRIES:
        assert caps[(country, "stable")] == pytest.approx(0.06)
        assert caps[(country, "volatile")] == pytest.approx(0.18)


def test_floor_and_ceiling_are_respected(config):
    flat = {year: 0.5 for year in range(2018, LAST_YEAR + 1)}
    wild = {year: 0.1 + (0.5 if year % 2 else 0.0) for year in range(2018, LAST_YEAR + 1)}
    table = _table({(c, "stable"): flat for c in COUNTRIES} | {(c, "volatile"): wild for c in COUNTRIES})
    settings = config["correction"]["yoy_cap"]
    caps = compute_yoy_caps(table, settings, 0.10)
    assert all(caps[(c, "stable")] == settings["floor"] for c in COUNTRIES)
    assert all(caps[(c, "volatile")] == settings["ceiling"] for c in COUNTRIES)


def test_short_history_falls_back_to_peers_then_to_scalar(config):
    settings = config["correction"]["yoy_cap"]
    short = {2023: 0.9, 2024: 0.5}  # one change of 0.4: too short to count, so the peers' 0.02 decides
    table = _table({("Alpha", "stable"): short} | {(c, "volatile"): {2024: 0.5} for c in COUNTRIES})
    caps = compute_yoy_caps(table, settings, 0.123)
    assert caps[("Alpha", "stable")] == pytest.approx(0.06)
    assert all(caps[(c, "volatile")] == 0.123 for c in COUNTRIES)  # no usable history anywhere in the segment


def test_fixed_mode_reproduces_scalar_cap(config):
    bench = build_benchmark(0, "education", n_countries=3)
    fixed = {**config["correction"]["yoy_cap"], "mode": "fixed"}
    caps = compute_yoy_caps(bench.baseline, fixed, 0.10)
    assert set(caps.values()) == {0.10}
    corrected = bench.baseline.copy()
    corrected.loc[corrected["is_forecast"], "value"] += 0.15
    assert check_constraints(bench.baseline, corrected, "education", 0.10, caps) == check_constraints(bench.baseline, corrected, "education", 0.10)


# ---- verifier, gate, evaluator, prompts ----


def test_check_constraints_uses_per_key_cap_and_none_is_unchanged():
    table = _table()
    corrected = table.copy()
    corrected.loc[(corrected["country"] == "Alpha") & (corrected["year"] == 2025), "value"] += MOVE
    caps = {(c, "stable"): 0.06 for c in COUNTRIES} | {(c, "volatile"): 0.18 for c in COUNTRIES}
    flagged = check_constraints(table, corrected, "age_gender", 0.10, caps)
    assert [(v.kind, v.country, v.segment, v.year) for v in flagged] == [("yoy_cap", "Alpha", "stable", 2025), ("yoy_cap", "Alpha", "stable", 2026)]
    assert "0.06" in flagged[0].detail
    assert check_constraints(table, corrected, "age_gender", 0.10, None) == check_constraints(table, corrected, "age_gender", 0.10) == []
    assert len(check_constraints(table, corrected, "age_gender", 0.05)) == 4  # the scalar still applies to every key


def test_gate_uses_adaptive_cap(config):
    table = _table()
    docs = [_doc("Alpha", segment, 2025, round(_value(table, "Alpha", segment, 2025) + MOVE, 3)) for segment in ("stable", "volatile")]
    session = CorrectionSession(table, "age_gender", SyntheticSourceStore(docs), config)
    assert session.caps[("Alpha", "stable")] == pytest.approx(0.06)

    stable = _propose(session, docs[0])
    assert not stable["accepted"] and "yoy_cap" in stable["reason"]
    assert _propose(session, docs[1]) == {"accepted": True, "reason": "accepted"}

    fixed = CorrectionSession(table, "age_gender", SyntheticSourceStore(docs), _with_mode(config, "fixed"))
    assert _propose(fixed, docs[0])["accepted"]


def test_evaluate_counts_against_adaptive_caps(config):
    table = _table()
    bench = Benchmark("age_gender", None, table, table[table["is_forecast"]].reset_index(drop=True), [], LAST_YEAR, 2026)
    corrected = table.copy()
    corrected.loc[(corrected["country"] == "Alpha") & (corrected["year"] == 2025), "value"] += MOVE
    store = SyntheticSourceStore([])
    assert evaluate(bench, corrected, [], store, config).violations == 2  # stable 2025 and 2026; volatile is within its cap
    assert evaluate(bench, corrected, [], store, _with_mode(config, "fixed")).violations == 0


def test_prompts_state_per_segment_cap_not_a_global_number(config):
    for prompt in (planner_system_prompt(), researcher_system_prompt(4)):
        assert "0.1" not in prompt and "cap stated" in prompt.lower()

    llm = ScriptedClient([LLMResponse(text=json.dumps({"suspects": [{"country": "Alpha", "segment": "stable"}]})), LLMResponse(text="done")])
    run_correction(_table(), "age_gender", SyntheticSourceStore([]), llm, config)
    summary = llm.calls[0]["messages"][1]["content"]
    assert "Alpha | stable" in summary and "cap=0.060" in summary and "cap=0.180" in summary
    assert "0.060" in llm.calls[1]["messages"][1]["content"]


# ---- real data (offline: reads the complete .cache/eurostat, no network) ----


def _published_moves(bench):
    """History up to the cut plus the published values after it, as a table whose post-cut cells are checked like forecast cells."""
    return pd.concat([bench.baseline[~bench.baseline["is_forecast"]], bench.truth], ignore_index=True)


def _yoy_flags(bench, caps):
    published = _published_moves(bench)
    return [v for v in check_constraints(published, published, bench.module, 0.10, caps) if v.kind == "yoy_cap"]


@pytest.fixture(scope="module")
def real(config):
    from adoption_forecaster.backtest.eurostat_backtest import DEFAULT_CACHE_DIR, build_real_backtest, cached_input_files, expected_input_files

    if os.environ.get("QF_REALDATA") != "1" or set(cached_input_files(DEFAULT_CACHE_DIR, config)) != set(expected_input_files(config)):
        pytest.skip("needs QF_REALDATA=1 and a complete .cache/eurostat")
    return {module: build_real_backtest(DEFAULT_CACHE_DIR, module, cut_year=2022, pred_year=2024) for module in ("age_gender", "education")}


@pytest.mark.realdata
def test_real_age_gender_adaptive_cap_flags_few_published_moves(real, config):
    bench = real["age_gender"]
    caps = compute_yoy_caps(bench.baseline, config["correction"]["yoy_cap"], 0.10)
    assert len(_yoy_flags(bench, None)) == 15
    assert len(_yoy_flags(bench, caps)) <= 2


@pytest.mark.realdata
def test_real_education_adaptive_cap_flags_nothing_and_is_stricter(real, config):
    bench = real["education"]
    caps = compute_yoy_caps(bench.baseline, config["correction"]["yoy_cap"], 0.10)
    assert _yoy_flags(bench, caps) == []
    assert statistics.median(caps.values()) < 0.10
