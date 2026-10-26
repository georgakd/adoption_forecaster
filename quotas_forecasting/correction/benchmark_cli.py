"""Benchmark runner: ``python -m quotas_forecasting.correction.benchmark_cli --module education --seeds 0-4 [--live] [--heldout] [--real]``.

Stub mode (default) drives the agent with an oracle ScriptedClient and enforces the hard gate (MAE improves, zero hallucinated
citations, zero introduced violations; exit 1 otherwise). Live mode uses the configured NVIDIA model and only reports the gate (exit 0 when the
run completed, 2 when NVIDIA_API_KEY or the model is missing). ``--real`` swaps the synthetic world for the real-data Eurostat backtest
(cached inputs cut at ``--cut-year``, scored on the published values up to ``--pred-year``; each seed only changes the synthetic
documents). Results go to ``<out>/<UTC timestamp>.json``; no secret is ever written.
"""

import argparse
import copy
import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from quotas_forecasting.backtest.eurostat_backtest import DEFAULT_CACHE_DIR, build_real_backtest
from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.agent import run_correction
from quotas_forecasting.correction.evaluation import evaluate
from quotas_forecasting.correction.llm import LLMClient, LLMResponse, MissingApiKeyError, get_llm_client
from quotas_forecasting.correction.oracle import build_oracle_client
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.correction.types import correction_record
from quotas_forecasting.synthetic.benchmark import HELDOUT_SEEDS, build_benchmark

DEFAULT_SEEDS = "0-2"
DEFAULT_N_COUNTRIES = 8


def parse_seeds(text: str) -> list[int]:
    """'0-4' -> [0..4]; '1,5,7' -> [1, 5, 7]; parts may be mixed ('0-1,9')."""
    seeds: list[int] = []
    for part in text.split(","):
        low, sep, high = part.strip().partition("-")
        first, last = int(low), int(high) if sep else int(low)
        if last < first:
            raise ValueError(f"empty seed range: {part!r}")
        seeds += range(first, last + 1)
    return seeds


class CountingClient:
    """Pass-through wrapper that counts chat calls; it does not change any request or response."""

    def __init__(self, inner: LLMClient):
        self._inner = inner
        self.calls = 0

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        self.calls += 1
        return self._inner.chat(messages, tools)


def _aggregate(per_seed: list[dict]) -> dict:
    mean = lambda key: sum(r[key] for r in per_seed) / len(per_seed)
    before, after = mean("mae_before"), mean("mae_after")
    return {
        "mae_before": before,
        "mae_after": after,
        "relative_mae_change": (after - before) / before if before else float("nan"),
        "hallucinated_citations": sum(r["hallucinated_citations"] for r in per_seed),
        "violations": sum(r["violations"] for r in per_seed),
        "baseline_violations": sum(r["baseline_violations"] for r in per_seed),
        "n_corrections": sum(r["n_corrections"] for r in per_seed),
    }


def _gate_passed(agg: dict) -> bool:
    return bool(agg["mae_after"] < agg["mae_before"]) and agg["hallucinated_citations"] == 0 and agg["violations"] == 0


def _table(per_seed: list[dict], agg: dict) -> str:
    header = f"{'seed':>5} {'mae_before':>11} {'mae_after':>10} {'halluc':>7} {'viol':>5} {'base_viol':>9} {'corr':>5}  error"
    rows = [
        f"{r['seed']:>5} {r['mae_before']:>11.4f} {r['mae_after']:>10.4f} {r['hallucinated_citations']:>7} {r['violations']:>5} "
        f"{r['baseline_violations']:>9} {r['n_corrections']:>5}  {r['error'] or ''}"
        for r in per_seed
    ]
    total = (
        f"{'all':>5} {agg['mae_before']:>11.4f} {agg['mae_after']:>10.4f} {agg['hallucinated_citations']:>7} {agg['violations']:>5} "
        f"{agg['baseline_violations']:>9} {agg['n_corrections']:>5}  relative MAE change {agg['relative_mae_change']:+.1%}"
    )
    return "\n".join([header, *rows, total])


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="benchmark_cli", description="Run the correction agent on the synthetic benchmark.")
    p.add_argument("--module", required=True, choices=["education", "age_gender"])
    p.add_argument("--seeds", default=None, help=f"'a-b' ranges and comma lists (default {DEFAULT_SEEDS}; HELDOUT_SEEDS with --heldout)")
    p.add_argument("--heldout", action="store_true", help="use the held-out seeds (100+) when --seeds is not given")
    p.add_argument("--live", action="store_true", help="use the real NVIDIA model instead of the oracle stub")
    p.add_argument("--n-countries", type=int, default=DEFAULT_N_COUNTRIES)
    p.add_argument("--real", action="store_true", help="real-data Eurostat backtest instead of the synthetic world (--n-countries ignored)")
    p.add_argument("--cut-year", type=int, default=2022, help="--real: last observed year given to the pipeline")
    p.add_argument("--pred-year", type=int, default=2024, help="--real: forecast year (pred_year - cut_year must be 2)")
    p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="--real: Eurostat input cache")
    p.add_argument("--model", default=None, help="override correction.model in memory (the config file is not changed)")
    p.add_argument("--out", default="benchmark_results")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        seeds = parse_seeds(args.seeds) if args.seeds else list(HELDOUT_SEEDS) if args.heldout else parse_seeds(DEFAULT_SEEDS)
    except ValueError as exc:
        print(f"error: invalid --seeds: {exc}", file=sys.stderr)
        return 2
    config = copy.deepcopy(load_config())
    if args.model:
        config["correction"]["model"] = args.model
    mode = "live" if args.live else "stub"

    counter: CountingClient | None = None
    if args.live:
        try:
            if not os.environ.get("NVIDIA_API_KEY"):
                raise MissingApiKeyError("NVIDIA_API_KEY environment variable is not set")
            counter = CountingClient(get_llm_client(config))
        except (MissingApiKeyError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

    started = time.monotonic()
    per_seed = []
    for seed in seeds:
        if args.real:
            bench = build_real_backtest(Path(args.cache_dir), args.module, cut_year=args.cut_year, pred_year=args.pred_year, seed=seed)
        else:
            bench = build_benchmark(seed, args.module, n_countries=args.n_countries)
        store = SyntheticSourceStore(bench.docs)
        llm = counter if counter else build_oracle_client(bench, store, config["correction"]["max_suspects"])
        result = run_correction(bench.baseline, args.module, store, llm, config)
        report = evaluate(bench, result.corrected, result.corrections, store, config)
        corrections = [correction_record(c, args.module) for c in result.corrections]
        per_seed.append({"seed": seed, **asdict(report), "error": result.error, "corrections": corrections})
    elapsed = round(time.monotonic() - started, 1)

    agg = _aggregate(per_seed)
    gate = _gate_passed(agg)
    out = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "config": {"mode": mode, "module": args.module, "model": config["correction"]["model"] if args.live else None,
                   "n_countries": None if args.real else args.n_countries, "seeds": seeds,
                   "source": "real-backtest" if args.real else "synthetic",
                   "cut_year": args.cut_year if args.real else None, "pred_year": args.pred_year if args.real else None},
        "per_seed": per_seed,
        "aggregate": agg,
        "gate_passed": gate,
        "llm_calls": counter.calls if counter else None,
        "elapsed_seconds": elapsed,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{out['timestamp']}.json"
    path.write_text(json.dumps(out, indent=2, allow_nan=True))

    print(_table(per_seed, agg))
    extra = f", LLM calls: {counter.calls}" if counter else ""
    print(f"mode: {mode}, gate {'PASSED' if gate else 'FAILED'}, elapsed: {elapsed}s{extra}")
    print(f"results: {path}")
    if args.live:
        return 0  # the live gate is a reported target, not a failure condition
    return 0 if gate else 1


if __name__ == "__main__":
    sys.exit(main())
