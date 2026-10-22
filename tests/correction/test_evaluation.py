import pandas as pd
import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.evaluation import evaluate
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.correction.types import TABLE_COLUMNS, Citation, Correction
from quotas_forecasting.synthetic.benchmark import Benchmark, build_benchmark


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


def test_evaluate_counts_untied_citation_as_hallucinated(bench, config):
    doc = bench.docs[0]
    import re

    sentence = re.split(r"(?<=\.)\s+", doc.text)[0]
    year = int(re.search(r"\b(?:19|20)\d{2}\b", sentence).group())
    number = re.search(r"(\d+\.\d)%|(0\.\d{3})", sentence)
    value = float(number.group(1)) / 100 if number.group(1) else float(number.group(2))
    unit = "percent" if number.group(1) else "share"
    citation = Citation(doc.doc_id, sentence, value, year, unit)
    row = bench.baseline[bench.baseline["is_forecast"]].iloc[0]
    wrong_country = "Nowhereland"
    corr = Correction(wrong_country, row["segment"], year, 0.1, value, citation, 0.9, "t")
    report = evaluate(bench, bench.baseline.copy(), [corr], SyntheticSourceStore(bench.docs), config)
    assert report.hallucinated_citations == 1


def test_mae_is_per_group_then_averaged_not_pooled(bench, config):
    import dataclasses

    import pandas as pd

    truth = pd.DataFrame(
        {"country": ["A"] * 3 + ["B"], "segment": ["s"] * 4, "year": [2025, 2026, 2027, 2026], "value": [0.5] * 4, "is_forecast": [True] * 4}
    )
    pred = truth.copy()
    pred["value"] = [0.6, 0.6, 0.6, 0.9]  # group A error 0.1 (3 rows), group B 0.4 (1 row): mean 0.25, pooled 0.175
    b = dataclasses.replace(bench, baseline=truth, truth=truth)
    from quotas_forecasting.correction.evaluation import _mae

    assert _mae(pred, b.truth) == pytest.approx(0.25)


SEGMENT = "Male, 16 to 24"


def _yoy_bench(v2025, v2026):
    """age_gender table with one segment: 2024 observed 0.5; the baseline's 2025 forecast already breaks the 0.10 yoy cap."""
    rows = [("Spain", SEGMENT, 2024, 0.5, False), ("Spain", SEGMENT, 2025, v2025, True), ("Spain", SEGMENT, 2026, v2026, True)]
    table = pd.DataFrame(rows, columns=TABLE_COLUMNS)
    return Benchmark("age_gender", None, table, table.copy(), [], 2024, 2026)


def _with(bench, year, value):
    table = bench.baseline.copy()
    table.loc[table["year"] == year, "value"] = value
    return table


def test_baseline_violation_is_reported_separately_not_as_agent_violation(config):
    bench = _yoy_bench(0.65, 0.66)
    report = evaluate(bench, bench.baseline.copy(), [], SyntheticSourceStore([]), config)
    assert (report.violations, report.baseline_violations) == (0, 1)


@pytest.mark.parametrize(
    "year,value,introduced",
    [
        (2026, 0.80, True),  # new: 0.65 -> 0.80 breaks the cap where the baseline did not
        (2025, 0.70, True),  # worsens the existing 2024 -> 2025 jump (0.15 -> 0.20)
        (2025, 0.62, False),  # reduces the existing jump (0.15 -> 0.12): still a violation, but not the agent's
    ],
)
def test_only_introduced_or_worsened_violations_count(config, year, value, introduced):
    bench = _yoy_bench(0.65, 0.66)
    report = evaluate(bench, _with(bench, year, value), [], SyntheticSourceStore([]), config)
    assert report.baseline_violations == 1
    assert (report.violations >= 1) is introduced and (report.violations == 0) is not introduced
