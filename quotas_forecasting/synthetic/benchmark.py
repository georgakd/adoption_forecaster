"""Benchmark builder: truth, the real pipeline's baseline forecast, and source documents for one seeded synthetic world."""

import copy
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction.adapters import to_table
from quotas_forecasting.correction.types import ForecastTable
from quotas_forecasting.data_handlers.file_writer import FileWriter
from quotas_forecasting.data_handlers.loader import Loader
from quotas_forecasting.forecasting.predictors import get_predictor
from quotas_forecasting.forecasting.processors import get_processor
from quotas_forecasting.synthetic.sources import HELDOUT_SEED_START, SourceDoc, make_docs
from quotas_forecasting.synthetic.world import DEFAULT_COUNTRIES, World, generate_world
from quotas_forecasting.synthetic.writers import write_age_gender_inputs, write_education_inputs

DEV_SEEDS = range(0, 10)
HELDOUT_SEEDS = range(HELDOUT_SEED_START, HELDOUT_SEED_START + 10)
LAST_YEAR = 2024
PRED_YEAR = 2026
COUNTRY_SEED_OFFSET = 20_000


@dataclass
class Benchmark:
    module: str
    world: World | None  # None for real-data backtests
    baseline: ForecastTable
    truth: ForecastTable
    docs: list[SourceDoc]
    last_year: int
    pred_year: int


class _Capture:
    """Predictor proxy that records the frame the real predictor returns (a copy, before the processor mutates it)."""

    def __init__(self, predictor, method: str):
        self._predictor, self._method, self.frame = predictor, method, None

    def __getattr__(self, name):
        attr = getattr(self._predictor, name)
        if name != self._method:
            return attr

        def wrapper(*args, **kwargs):
            result = attr(*args, **kwargs)
            if self.frame is None:  # education predicts once per age range; the first (16-64) is the baseline
                self.frame = result.copy()
            return result

        return wrapper


def _run_pipeline(world: World, module: str) -> ForecastTable:
    """Run the real local pipeline on the world's visible (<= last_year) inputs and return the predictor's frame as a table."""
    namespace = module.title()
    with tempfile.TemporaryDirectory() as data_dir:
        config = copy.deepcopy(load_config())
        config["defaults"]["local_data_dir"] = data_dir
        config["defaults"]["auto_fetch"] = False
        config["pubsub"]["enable_notifications"] = False
        writer_fn = write_education_inputs if module == "education" else write_age_gender_inputs
        writer_fn(world, Path(data_dir) / namespace / "input")
        loader = Loader(namespace=namespace, mode="local", config=config)
        writer = FileWriter(namespace=namespace, mode="local", pred_year=world.pred_year, config=config)
        predictor = get_predictor(module, mode="local", pred_year=world.pred_year, last_year=world.last_year, loader=loader, config=config)
        capture = _Capture(predictor, "predict" if module == "education" else "run_with_data")
        processor = get_processor(module, config=config)
        if module == "education":
            processor.run_education_pipeline(loader, writer, capture, world.pred_year, world.last_year)
        else:
            processor.run_age_gender_pipeline(loader, writer, capture, world.pred_year, world.last_year)
    if capture.frame is None:
        raise RuntimeError(f"{module} pipeline produced no predictions")
    return to_table(capture.frame, module, world.last_year)


def _pipeline_segment(segment: str) -> str:
    """Generator label 'Males, 16 to 24 years old' -> pipeline label 'Male, 16 to 24'; education labels are unchanged."""
    return re.sub(r"^(Male|Female)s, (\d+ to \d+) years old$", r"\1, \2", segment)


def _truth_table(world: World, module: str, baseline: ForecastTable) -> ForecastTable:
    frame = world.truth_education if module == "education" else world.truth_age_gender
    truth = to_table(frame, module, world.last_year)
    truth["segment"] = truth["segment"].map(_pipeline_segment)
    keys = ["country", "segment", "year"]
    truth = baseline[keys].merge(truth, on=keys, how="left", validate="one_to_one")
    if truth["value"].isna().any():
        raise ValueError(f"truth is missing keys present in the {module} baseline")
    return truth[baseline.columns]


def build_benchmark(
    seed: int,
    module: str,
    n_countries: int = 8,
    noise: float = 0.01,
    wrong_fraction: float = 0.2,
    conflict_fraction: float = 0.15,
) -> Benchmark:
    """Build a benchmark whose baseline is the real pipeline's forecast; it misses the post-last_year shocks that truth contains."""
    if module not in ("education", "age_gender"):
        raise ValueError(f"Unknown module: {module}. Must be 'education' or 'age_gender'")
    if not 1 <= n_countries <= len(DEFAULT_COUNTRIES):
        raise ValueError(f"n_countries must be in [1, {len(DEFAULT_COUNTRIES)}]")
    picked = np.random.default_rng(seed + COUNTRY_SEED_OFFSET).choice(len(DEFAULT_COUNTRIES), size=n_countries, replace=False)
    countries = [DEFAULT_COUNTRIES[i] for i in sorted(picked)]
    world = generate_world(seed, countries=countries, last_year=LAST_YEAR, pred_year=PRED_YEAR)
    baseline = _run_pipeline(world, module)
    truth = _truth_table(world, module, baseline)
    docs = make_docs(truth, seed, noise, wrong_fraction, conflict_fraction)
    return Benchmark(module, world, baseline, truth, docs, world.last_year, world.pred_year)
