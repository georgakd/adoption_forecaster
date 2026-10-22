"""Evaluation harness: MAE against truth before/after correction, plus hallucinated-citation and constraint-violation counts."""

from dataclasses import dataclass

from quotas_forecasting.correction.citations import verify_citation
from quotas_forecasting.correction.constraints import check_constraints
from quotas_forecasting.correction.sources import SourceStore
from quotas_forecasting.correction.types import Correction, ForecastTable
from quotas_forecasting.synthetic.benchmark import Benchmark

KEY = ["country", "segment", "year"]


@dataclass
class EvalReport:
    mae_before: float
    mae_after: float
    hallucinated_citations: int
    violations: int
    n_corrections: int


def _mae(predicted: ForecastTable, truth: ForecastTable) -> float:
    """MAE over forecast-year rows per (country, segment), averaged over (country, segment); rows with a NaN on either side are dropped."""
    forecast = truth[truth["is_forecast"]][KEY + ["value"]].rename(columns={"value": "truth"})
    merged = forecast.merge(predicted[KEY + ["value"]], on=KEY, how="inner").dropna(subset=["truth", "value"])
    if merged.empty:
        return float("nan")
    merged["abs_err"] = (merged["value"] - merged["truth"]).abs()
    return float(merged.groupby(["country", "segment"])["abs_err"].mean().mean())


def evaluate(benchmark: Benchmark, corrected: ForecastTable, corrections: list[Correction], store: SourceStore, config: dict) -> EvalReport:
    settings = config["correction"]
    hallucinated = sum(not verify_citation(c.citation, store, settings["citation_tolerance"]) for c in corrections)
    violations = check_constraints(benchmark.baseline, corrected, benchmark.module, settings["max_yoy_change"])
    return EvalReport(
        mae_before=_mae(benchmark.baseline, benchmark.truth),
        mae_after=_mae(corrected, benchmark.truth),
        hallucinated_citations=hallucinated,
        violations=len(violations),
        n_corrections=len(corrections),
    )
