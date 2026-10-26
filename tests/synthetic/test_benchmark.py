import dataclasses

import pytest
from pandas.testing import assert_frame_equal

from adoption_forecaster.correction.sources import SourceSnippet, SyntheticSourceStore
from adoption_forecaster.synthetic.benchmark import DEV_SEEDS, HELDOUT_SEEDS, build_benchmark
from adoption_forecaster.synthetic.sources import DEV_STYLE, HELDOUT_STYLE, format_value, make_docs

SEED = 0
WRONG_THRESHOLD = 0.03  # noise=0.01 is clipped to +-0.02; wrong offsets are >= 0.06
KEY = ["country", "segment", "year"]


@pytest.fixture(scope="module")
def benchmarks():
    return {module: build_benchmark(SEED, module) for module in ("education", "age_gender")}


def _truth_lookup(bench):
    return {tuple(k): v for *k, v in bench.truth[KEY + ["value"]].itertuples(index=False)}


def test_seed_constants():
    assert list(DEV_SEEDS) == list(range(0, 10))
    assert list(HELDOUT_SEEDS) == list(range(100, 110))


def test_benchmark_deterministic_by_seed(benchmarks):
    first = benchmarks["education"]
    again = build_benchmark(SEED, "education")
    assert_frame_equal(first.baseline, again.baseline)
    assert_frame_equal(first.truth, again.truth)
    assert first.docs == again.docs
    assert first.world.countries == again.world.countries and first.world.shocks == again.world.shocks
    assert make_docs(first.truth, SEED + 1, 0.01, 0.2, 0.15) != first.docs


@pytest.mark.parametrize("module", ["education", "age_gender"])
def test_baseline_error_is_positive_for_shocked_countries(benchmarks, module):
    bench = benchmarks[module]
    assert bench.module == module and (bench.last_year, bench.pred_year) == (2024, 2026)
    assert len(bench.world.countries) == 8
    merged = bench.baseline.merge(bench.truth, on=KEY, suffixes=("_base", "_true"), validate="one_to_one")
    assert len(merged) == len(bench.baseline) == len(bench.truth)
    future = merged[merged["is_forecast_base"]].copy()
    future["error"] = (future["value_base"] - future["value_true"]).abs()
    shocks = bench.world.shocks
    shocked = future.apply(lambda r: r["country"] in shocks and r["year"] >= shocks[r["country"]].year, axis=1)
    unshocked = ~future["country"].isin(list(shocks))
    assert shocked.any() and unshocked.any()
    assert future.loc[shocked, "error"].mean() > future.loc[unshocked, "error"].mean()


@pytest.mark.parametrize("module", ["education", "age_gender"])
def test_some_docs_are_wrong_and_some_conflict(benchmarks, module):
    bench = benchmarks[module]
    truth = _truth_lookup(bench)
    docs = bench.docs
    assert docs
    for doc in docs:
        assert doc.unit in ("share", "percent") and 0.0 <= doc.value <= 1.0
    wrong = sum(abs(doc.value - truth[(doc.country, doc.segment, doc.year)]) > WRONG_THRESHOLD for doc in docs)
    assert abs(wrong - 0.2 * len(docs)) <= 1

    by_key: dict[tuple, list[float]] = {}
    for doc in docs:
        by_key.setdefault((doc.country, doc.segment, doc.year), []).append(doc.value)
    conflicting = sum(len(v) > 1 and max(v) - min(v) > WRONG_THRESHOLD for v in by_key.values())
    assert conflicting > 0
    assert abs(conflicting - 0.15 * len(by_key)) <= 1


def test_some_countries_have_no_docs(benchmarks):
    for bench in benchmarks.values():
        with_docs = {doc.country for doc in bench.docs}
        countries = set(bench.truth["country"])
        assert with_docs and with_docs < countries


def test_store_search_filters_by_country_and_year(benchmarks):
    bench = benchmarks["age_gender"]
    store = SyntheticSourceStore(bench.docs)
    doc = bench.docs[0]
    results = store.search("internet", country=doc.country, year=doc.year)
    assert results and all(isinstance(s, SourceSnippet) for s in results)
    assert all(s.country == doc.country and s.year == doc.year for s in results)
    assert {f.name for f in dataclasses.fields(SourceSnippet)} == {"doc_id", "publisher", "country", "year", "title"}
    assert {s.year for s in store.search("internet", year=doc.year)} == {doc.year}
    no_docs = sorted(set(bench.truth["country"]) - {d.country for d in bench.docs})[0]
    assert store.search("internet", country=no_docs) == []


def test_store_read_returns_text_containing_the_value_with_unit(benchmarks):
    bench = benchmarks["education"]
    store = SyntheticSourceStore(bench.docs)
    for doc in bench.docs:
        text = store.read(doc.doc_id)
        assert text == doc.text and format_value(doc.value, doc.unit) in text
    assert {doc.unit for doc in bench.docs} == {"share", "percent"}
    assert format_value(0.5234, "percent") == "52.3%" and format_value(0.5234, "share") == "0.523"
    with pytest.raises(KeyError):
        store.read("no-such-doc")


def test_heldout_seeds_vary_publishers_and_templates(benchmarks):
    truth = benchmarks["education"].truth
    dev = make_docs(truth, DEV_SEEDS[0], 0.01, 0.2, 0.15)
    heldout = make_docs(truth, HELDOUT_SEEDS[0], 0.01, 0.2, 0.15)
    assert not {d.publisher for d in dev} & {d.publisher for d in heldout}
    for pool in ("publishers", "target", "distractor", "title"):
        assert not set(DEV_STYLE[pool]) & set(HELDOUT_STYLE[pool]), pool
