"""Evaluation harness: MAE against truth before/after correction, plus hallucinated-citation and constraint-violation counts.

``violations`` counts only violations the corrections introduced or worsened (the same rule ``propose_correction`` enforces);
violations already in the baseline forecast are reported separately as ``baseline_violations``.

``hallucinated_citations`` counts a correction whose citation fails the gate's own check (``citation_supports_correction``) OR, independently
of the gate, whose cited document is not one of the benchmark's documents or whose own ``country``/``segment``/``year`` metadata (benchmark
ground truth, never shown to the agent) differs from the corrected cell. Only the second part can catch what the gate lets through, such as a
quote about another segment of the same country and year.
"""

from dataclasses import dataclass

from quotas_forecasting.correction.caps import compute_yoy_caps
from quotas_forecasting.correction.citations import citation_supports_correction
from quotas_forecasting.correction.constraints import check_constraints
from quotas_forecasting.correction.sources import SourceStore
from quotas_forecasting.correction.tools import introduced_violations
from quotas_forecasting.correction.types import Correction, ForecastTable
from quotas_forecasting.synthetic.benchmark import Benchmark
from quotas_forecasting.synthetic.sources import SourceDoc

KEY = ["country", "segment", "year"]


@dataclass
class EvalReport:
    mae_before: float
    mae_after: float
    hallucinated_citations: int
    violations: int
    n_corrections: int
    baseline_violations: int


def _mae(predicted: ForecastTable, truth: ForecastTable) -> float:
    """MAE over forecast-year rows per (country, segment), averaged over (country, segment); rows with a NaN on either side are dropped."""
    forecast = truth[truth["is_forecast"]][KEY + ["value"]].rename(columns={"value": "truth"})
    merged = forecast.merge(predicted[KEY + ["value"]], on=KEY, how="inner").dropna(subset=["truth", "value"])
    if merged.empty:
        return float("nan")
    merged["abs_err"] = (merged["value"] - merged["truth"]).abs()
    return float(merged.groupby(["country", "segment"])["abs_err"].mean().mean())


def _cites_another_cell(correction: Correction, docs: dict[str, SourceDoc]) -> bool:
    """Independent of the gate: the cited document's own metadata must name the corrected cell."""
    doc = docs.get(correction.citation.doc_id)
    return doc is None or (doc.country, doc.segment, int(doc.year)) != (correction.country, correction.segment, int(correction.year))


def evaluate(benchmark: Benchmark, corrected: ForecastTable, corrections: list[Correction], store: SourceStore, config: dict) -> EvalReport:
    settings = config["correction"]
    docs = {doc.doc_id: doc for doc in benchmark.docs}
    hallucinated = sum(
        not citation_supports_correction(c, store, settings["citation_tolerance"]) or _cites_another_cell(c, docs) for c in corrections
    )
    base, module, cap = benchmark.baseline, benchmark.module, settings["max_yoy_change"]
    caps = compute_yoy_caps(base, settings["yoy_cap"], cap)  # the same per-segment caps the gate uses
    violations = introduced_violations(base, base, corrected, module, cap, caps)
    return EvalReport(
        mae_before=_mae(benchmark.baseline, benchmark.truth),
        mae_after=_mae(corrected, benchmark.truth),
        hallucinated_citations=hallucinated,
        violations=len(violations),
        n_corrections=len(corrections),
        baseline_violations=len(check_constraints(base, base, module, cap, caps)),
    )
