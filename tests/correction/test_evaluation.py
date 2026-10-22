import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.evaluation import evaluate
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.correction.types import Citation, Correction
from quotas_forecasting.synthetic.benchmark import build_benchmark


@pytest.fixture(scope="module")
def bench():
    return build_benchmark(0, "education", n_countries=3)


@pytest.fixture(scope="module")
def config():
    return load_config()


def _correction(bench, citation):
    row = bench.baseline[bench.baseline["is_forecast"]].iloc[0]
    return Correction(row["country"], row["segment"], int(row["year"]), float(row["value"]), 0.5, citation, 0.9, "test")


def test_evaluate_perfect_correction_gives_zero_mae_after(bench, config):
    report = evaluate(bench, bench.truth.copy(), [], SyntheticSourceStore(bench.docs), config)
    assert report.mae_after == pytest.approx(0.0, abs=1e-12)
    assert report.mae_before > 0


def test_evaluate_no_corrections_equals_baseline_mae(bench, config):
    report = evaluate(bench, bench.baseline.copy(), [], SyntheticSourceStore(bench.docs), config)
    assert report.mae_after == report.mae_before
    assert (report.hallucinated_citations, report.violations, report.n_corrections) == (0, 0, 0)


def test_evaluate_counts_hallucinated_citation_and_violation(bench, config):
    corrected = bench.baseline.copy()
    idx = corrected[corrected["is_forecast"]].index[0]
    corrected.loc[idx, "value"] = 1.5  # out of bounds
    fake = Citation(bench.docs[0].doc_id, "not in the doc", 0.5, 2026, "share")
    report = evaluate(bench, corrected, [_correction(bench, fake)], SyntheticSourceStore(bench.docs), config)
    assert report.hallucinated_citations == 1
    assert report.violations >= 1
    assert report.n_corrections == 1


def test_evaluate_excludes_nan_rows(bench, config):
    corrected = bench.baseline.copy()
    corrected.loc[corrected[corrected["is_forecast"]].index[0], "value"] = float("nan")
    report = evaluate(bench, corrected, [], SyntheticSourceStore(bench.docs), config)
    assert report.mae_after == report.mae_after  # not NaN
