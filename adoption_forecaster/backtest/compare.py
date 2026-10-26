"""Rolling-origin comparison of the production ensemble against the experimental GAM and balancing forecasters.

Every method sees exactly the history in the benchmark's baseline table (the pipeline's own processed history up to
``last_year``) and forecasts the same cells; MAE is ``correction.evaluation._mae`` against the benchmark truth. "ensemble" is the
existing pipeline output (``Benchmark.baseline``), never a re-implementation. Education forecasts only the tertiary share and sets
secondary = 1 - tertiary, so the sum-to-one constraint holds by construction.

Regenerate every number in docs/MODEL_REVIEW.md (offline, from the .cache/eurostat inputs) with:

    python -m adoption_forecaster.backtest.compare --out benchmark_results/model_review.md
"""

import argparse
import copy
import logging
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from adoption_forecaster.backtest.eurostat_backtest import DEFAULT_CACHE_DIR, _copy_inputs, build_real_backtest
from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.correction.evaluation import _mae
from adoption_forecaster.correction.types import ForecastTable
from adoption_forecaster.forecasting.experimental.balancing import proxy_forecast
from adoption_forecaster.forecasting.experimental.gam import gam_forecast
from adoption_forecaster.synthetic.benchmark import DEV_SEEDS, HELDOUT_SEEDS, Benchmark, build_benchmark, run_pipeline_on_inputs
from adoption_forecaster.synthetic.writers import write_age_gender_inputs, write_education_inputs

METHODS = ("ensemble", "gam", "entropy_proxy", "kernel_proxy")
MODULES = ("education", "age_gender")
REAL_ORIGINS = (2019, 2020, 2021, 2022)
# Real origin used (Education only) with the dev seeds to choose GAM options. NOT fully out of sample: its scored years (2019-2020)
# overlap the truth of the earliest reported origin (2019 is scored on 2020-2021), and nothing earlier can run for Age_Gender;
# gam_anchor_robustness() checks that the decision does not depend on the choice made here.
TUNING_ORIGIN = 2018
# Decision rule, fixed before any result was looked at: a method becomes an option for a module only if its MAE is at least
# WIN_MARGIN (relative) below the ensemble's on BOTH the synthetic held-out seeds and the mean over the real origins.
WIN_MARGIN = 0.05
GAM_ANCHOR = True  # chosen on dev seeds + TUNING_ORIGIN (see tune_gam(); anchoring won on all three); see the caveat above
BLEND_WEIGHTS = {
    "education": [(0.7, 0.3), (1.0, 0.0), (0.0, 1.0), (0.5, 0.5)],  # (theta, random walk); first entry is the config value
    "age_gender": [(1 / 3, 1 / 3, 1 / 3), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (0.5, 0.5, 0.0)],  # (theta, rw, logistic)
}
KEY = ["country", "segment", "year"]


def _series_frame(bench: Benchmark) -> pd.DataFrame:
    """History of every (country, segment) as rows, year columns up to last_year."""
    history = bench.baseline[~bench.baseline["is_forecast"]]
    return history.pivot(index=["country", "segment"], columns="year", values="value").sort_index()


def _forecast_rows(bench: Benchmark, method: str, gam_anchor: bool) -> tuple[dict, list[str]]:
    """{(country, segment, year): value} for every forecast cell, plus which weights each proxy forecast used."""
    frame = _series_frame(bench)
    years = sorted(bench.baseline.loc[bench.baseline["is_forecast"], "year"].unique())
    if bench.module == "education":
        frame = frame.xs("tertiary", level="segment", drop_level=False)
    pool = frame.droplevel("segment") if bench.module == "education" else frame
    values, used = {}, []
    for (country, segment), row in frame.iterrows():
        row = row.dropna()
        horizon = years[-1] - int(row.index[-1])
        if method == "gam":
            path = pd.Series(gam_forecast(row, horizon, monotone=True, anchor=gam_anchor), index=range(int(row.index[-1]) + 1, years[-1] + 1))
        else:
            path = proxy_forecast(row, pool, "entropy" if method == "entropy_proxy" else "kernel", horizon=horizon)
            used.append(path.attrs["weights"])
        for year in years:
            values[(country, segment, year)] = float(path[year])
            if bench.module == "education":
                values[(country, "secondary", year)] = 1.0 - float(path[year])
    return values, used


def method_table(bench: Benchmark, method: str, gam_anchor: bool = GAM_ANCHOR) -> ForecastTable:
    """The benchmark's baseline table with its forecast cells replaced by ``method``'s forecasts (history untouched)."""
    return _method_table(bench, method, gam_anchor)[0]


def _method_table(bench: Benchmark, method: str, gam_anchor: bool) -> tuple[ForecastTable, float]:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}. Must be one of {METHODS}")
    if method == "ensemble":
        return bench.baseline, float("nan")
    values, used = _forecast_rows(bench, method, gam_anchor)
    table = bench.baseline.copy()
    mask = table["is_forecast"]
    table.loc[mask, "value"] = [values[k] for k in table.loc[mask, KEY].itertuples(index=False, name=None)]
    fallback = float(np.mean([u != method.split("_")[0] for u in used])) if used else float("nan")
    return table, fallback


def compare_methods(tables: list[Benchmark], methods: list[str], origins: list[int], gam_anchor: bool = GAM_ANCHOR) -> pd.DataFrame:
    """Rolling-origin MAE per (module, method): the mean over the tables whose ``last_year`` is in ``origins``.

    Columns: module, method, n_tables, mae, ``mae_<origin>`` per origin, and fallback_share (share of proxy forecasts whose
    entropy weights were infeasible and replaced by kernel weights, or that had too little history; NaN for other methods).
    """
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        raise ValueError(f"Unknown methods {unknown}. Must be drawn from {METHODS}")
    selected = [t for t in tables if t.last_year in origins]
    missing = sorted(set(origins) - {t.last_year for t in selected})
    if missing:
        raise ValueError(f"No table for origins {missing}")
    rows = []
    for module in [m for m in MODULES if any(t.module == m for t in selected)]:
        module_tables = [t for t in selected if t.module == module]
        for method in methods:
            scored = [(t.last_year, *_scored(t, method, gam_anchor)) for t in module_tables]
            row = {"module": module, "method": method, "n_tables": len(scored), "mae": float(np.mean([s[1] for s in scored]))}
            for origin in sorted({s[0] for s in scored}):
                row[f"mae_{origin}"] = float(np.mean([s[1] for s in scored if s[0] == origin]))
            row["fallback_share"] = float(np.nanmean([s[2] for s in scored])) if method.endswith("_proxy") else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def _scored(bench: Benchmark, method: str, gam_anchor: bool) -> tuple[float, float]:
    table, fallback = _method_table(bench, method, gam_anchor)
    return _mae(table, bench.truth), fallback


# ---------------------------------------------------------------------------------------------------------------------------
# experiment drivers (used by main(); not needed by compare_methods)
# ---------------------------------------------------------------------------------------------------------------------------


def synthetic_tables(seeds, modules=MODULES) -> list[Benchmark]:
    return [build_benchmark(seed, module) for module in modules for seed in seeds]


def real_tables(origins, modules=MODULES, cache_dir: Path = DEFAULT_CACHE_DIR) -> tuple[list[Benchmark], list[str]]:
    """Real backtests (cut c, predict c + 2) from the offline cache; origins the pipeline cannot run are reported, not hidden."""
    tables, skipped = [], []
    for module in modules:
        for origin in origins:
            try:
                tables.append(build_real_backtest(cache_dir, module, cut_year=origin, pred_year=origin + 2, seed=0))
            except Exception as exc:  # noqa: BLE001 - the reason is recorded in the review
                skipped.append(f"{module} origin {origin}: {type(exc).__name__}: {str(exc)[:120] or repr(exc)}")
    return tables, skipped


def _weights_config(config: dict, module: str, weights: tuple) -> dict:
    config = copy.deepcopy(config)
    settings = config["modules"][module]["settings"]
    names = ("theta_weight", "random_walk_weight") if module == "education" else ("theta_weight", "random_walk_weight", "logistic_weight")
    settings.update(dict(zip(names, weights)))
    return config


def rerun_baseline(bench: Benchmark, config: dict, cache_dir: Path = DEFAULT_CACHE_DIR) -> ForecastTable:
    """Re-run the real pipeline on the benchmark's own inputs with ``config`` (synthetic world or real cache cut at last_year)."""
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp)
        if bench.world is not None:
            writer = write_education_inputs if bench.module == "education" else write_age_gender_inputs
            writer(bench.world, data_dir / bench.module.title() / "input")
        else:
            _copy_inputs(cache_dir, bench.module, config, data_dir, bench.last_year)
        return run_pipeline_on_inputs(data_dir, bench.module, bench.last_year, bench.pred_year, config=config)


def blend_weight_experiment(tables: list[Benchmark], label: str, cache_dir: Path = DEFAULT_CACHE_DIR) -> pd.DataFrame:
    """MAE of the full pipeline re-run with each weight vector in BLEND_WEIGHTS (component alone = weight 1)."""
    config = load_config()
    rows = []
    for module in MODULES:
        module_tables = [t for t in tables if t.module == module]
        if not module_tables:
            continue
        for weights in BLEND_WEIGHTS[module]:
            cfg = _weights_config(config, module, weights)
            maes = [_mae(rerun_baseline(t, cfg, cache_dir), t.truth) for t in module_tables]
            rows.append({"benchmark": label, "module": module, "weights": "/".join(f"{w:.2f}" for w in weights), "mae": float(np.mean(maes))})
    return pd.DataFrame(rows)


def tune_gam(tables: list[Benchmark]) -> pd.DataFrame:
    """MAE of the GAM with and without anchoring on the tuning tables (dev seeds + TUNING_ORIGIN)."""
    rows = []
    for anchor in (False, True):
        result = compare_methods(tables, ["gam"], origins=sorted({t.last_year for t in tables}), gam_anchor=anchor)
        for _, r in result.iterrows():
            rows.append({"module": r["module"], "anchor": anchor, "mae": r["mae"]})
    return pd.DataFrame(rows)


def gam_anchor_robustness(tables: list[Benchmark], origins: list[int]) -> pd.DataFrame:
    """GAM rolling-origin MAE per (module, anchor) for both anchor choices (columns as compare_methods, ``anchor`` for ``method``)."""
    frames = []
    for anchor in (False, True):
        result = compare_methods(tables, ["gam"], origins, gam_anchor=anchor).drop(columns=["method", "fallback_share"])
        result.insert(1, "anchor", anchor)
        frames.append(result)
    out = pd.concat(frames, ignore_index=True)
    out["_order"] = out["module"].map(MODULES.index)
    return out.sort_values(["_order", "anchor"], kind="stable").drop(columns="_order").reset_index(drop=True)


def diagnostics(tables: list[Benchmark]) -> pd.DataFrame:
    """Data facts behind the review's verdicts, computed on the real tables."""
    rows = []
    for bench in tables:
        frame = _series_frame(bench)
        if bench.module == "education":
            wide = frame.unstack("segment")
            sec, ter = wide.xs("secondary", axis=1, level="segment"), wide.xs("tertiary", axis=1, level="segment")
            rows.append({"origin": bench.last_year, "module": bench.module, "fact": "history cells with secondary == 1 and tertiary NaN",
                         "value": int(((sec == 1.0) & ter.isna()).sum().sum())})
            forecast = bench.baseline[bench.baseline["is_forecast"]].pivot(index=["country", "year"], columns="segment", values="value")
            rows.append({"origin": bench.last_year, "module": bench.module, "fact": "max |secondary + tertiary - 1| in forecast",
                         "value": float((forecast.sum(axis=1) - 1).abs().max())})
        else:
            last = frame.apply(lambda r: r.dropna().iloc[-1], axis=1)
            rows.append({"origin": bench.last_year, "module": bench.module, "fact": "share of series with last value >= K_max (0.995)",
                         "value": float((last >= 0.995).mean())})
            truth = bench.truth.dropna(subset=["value"])
            rows.append({"origin": bench.last_year, "module": bench.module, "fact": "share of truth cells >= 0.995",
                         "value": float((truth["value"] >= 0.995).mean())})
    return pd.DataFrame(rows)


def decide(synthetic: pd.DataFrame, real: pd.DataFrame) -> pd.DataFrame:
    """Relative MAE reduction vs the ensemble on each benchmark; ``wins`` only if both reach WIN_MARGIN."""
    def mae(df, module, method):
        return df[(df["module"] == module) & (df["method"] == method)]["mae"].iloc[0]

    rows = []
    for module in synthetic["module"].unique():
        for method in [m for m in synthetic["method"].unique() if m != "ensemble"]:
            gain_syn = 1 - mae(synthetic, module, method) / mae(synthetic, module, "ensemble")
            gain_real = 1 - mae(real, module, method) / mae(real, module, "ensemble")
            wins = gain_syn >= WIN_MARGIN and gain_real >= WIN_MARGIN
            rows.append({"module": module, "method": method, "gain_synthetic": gain_syn, "gain_real": gain_real, "wins": wins})
    return pd.DataFrame(rows)


def _markdown(df: pd.DataFrame, digits: int = 4) -> str:
    def fmt(v):
        return f"{v:.{digits}f}" if isinstance(v, (float, np.floating)) else str(v)

    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> str:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out", type=Path, default=None, help="also write the markdown report here")
    parser.add_argument("--skip-weights", action="store_true", help="skip the (slow) blend-weight re-runs")
    parser.add_argument("--anchor-robustness", action="store_true", help="only score the GAM with both anchor choices (held-out + real)")
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    warnings.filterwarnings("ignore")
    from threadpoolctl import threadpool_limits  # experiments group (also a scikit-learn dependency)

    start = time.time()
    with threadpool_limits(1):  # many tiny BLAS calls: one thread is several times faster than an oversubscribed pool
        report = _anchor_report(args) if args.anchor_robustness else _report(args)
    report += f"\n\n(generated in {time.time() - start:.0f}s)\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
    print(report)
    return report


def _anchor_report(args) -> str:
    """Does the decision depend on GAM_ANCHOR? GAM with both anchors against the ensemble, on held-out seeds and real origins."""
    heldout = synthetic_tables(HELDOUT_SEEDS)
    real, skipped = real_tables(REAL_ORIGINS, cache_dir=args.cache_dir)
    origins_run = sorted({t.last_year for t in real})
    sections, decisions = [], []
    for label, tables, origins in (("Synthetic, held-out seeds 100-109", heldout, [2024]), (f"Real, origins {origins_run}", real, origins_run)):
        ensemble = compare_methods(tables, ["ensemble"], origins).drop(columns=["fallback_share"])
        gam = gam_anchor_robustness(tables, origins)
        sections.append(f"## {label}: ensemble\n\n" + _markdown(ensemble))
        sections.append(f"## {label}: GAM by anchor\n\n" + _markdown(gam))
        decisions.append((ensemble, gam))
    rows = []
    for anchor in (False, True):
        frames = [pd.concat([e[["module", "method", "mae"]], g[g["anchor"] == anchor].assign(method="gam")[["module", "method", "mae"]]])
                  for e, g in decisions]
        rows.append(decide(*frames).assign(anchor=anchor))
    sections.append(f"## Decision by anchor (WIN_MARGIN = {WIN_MARGIN})\n\n" + _markdown(pd.concat(rows, ignore_index=True)))
    sections.append("## Real origins that could not run\n\n" + "\n".join(f"- {s}" for s in skipped or ["(none)"]))
    return "\n\n".join(sections)


def _report(args) -> str:
    sections = []

    dev = synthetic_tables(DEV_SEEDS)
    tuning_real, tuning_skipped = real_tables([TUNING_ORIGIN], modules=("education",), cache_dir=args.cache_dir)
    sections.append("## GAM tuning (synthetic dev seeds 0-9)\n\n" + _markdown(tune_gam(dev)))
    sections.append(f"## GAM tuning (real Education, origin {TUNING_ORIGIN})\n\n" + _markdown(tune_gam(tuning_real)))

    heldout = synthetic_tables(HELDOUT_SEEDS)
    real, skipped = real_tables(REAL_ORIGINS, cache_dir=args.cache_dir)
    synthetic_result = compare_methods(heldout, list(METHODS), [2024])
    origins_run = sorted({t.last_year for t in real})
    real_result = compare_methods(real, list(METHODS), origins_run)
    sections.append("## Synthetic, dev seeds 0-9 (origin 2024)\n\n" + _markdown(compare_methods(dev, list(METHODS), [2024])))
    sections.append("## Synthetic, held-out seeds 100-109 (origin 2024)\n\n" + _markdown(synthetic_result))
    sections.append(f"## Real Eurostat backtest, origins {origins_run} (cut c, predict c+1, c+2)\n\n" + _markdown(real_result))
    sections.append(f"## Decision (WIN_MARGIN = {WIN_MARGIN})\n\n" + _markdown(decide(synthetic_result, real_result)))
    sections.append("## Real origins that could not run\n\n" + "\n".join(f"- {s}" for s in (skipped + tuning_skipped) or ["(none)"]))
    sections.append("## Real-data diagnostics\n\n" + _markdown(diagnostics(real)))
    if not args.skip_weights:
        weights = pd.concat([blend_weight_experiment(heldout, "synthetic held-out"), blend_weight_experiment(real, "real", args.cache_dir)])
        sections.append("## Blend weights (full pipeline re-run; first row per module is the config value)\n\n" + _markdown(weights))
    return "\n\n".join(sections)


if __name__ == "__main__":
    main()
