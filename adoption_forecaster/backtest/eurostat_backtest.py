"""Real-data backtest: real Eurostat inputs (fetched by the repo's own fetcher), cut at ``cut_year``, scored on the published values.

``fetch_eurostat_inputs`` runs ``EurostatDataFetcher`` against the GCS emulator (it refuses to run without STORAGE_EMULATOR_HOST, so no
real GCP call can happen) and copies the uploaded timestamped CSVs into ``<cache_dir>/<Namespace>/input/``. A complete cache is reused
without touching the network or the emulator. ``build_real_backtest`` runs the real pipeline twice on the cached inputs: once truncated
at ``cut_year`` (the baseline) and once up to ``pred_year`` (whose observed cells after ``cut_year`` are the truth). Source documents
are synthetic prose wrappers around those real numbers (``make_docs``); there is no synthetic world.
"""

import copy
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from google.auth.credentials import AnonymousCredentials
from google.cloud import storage

from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.config.logging_config import get_logger
from adoption_forecaster.correction.types import ForecastTable
from adoption_forecaster.fetch_data.fetch_data_from_Eurostat import EurostatDataFetcher
from adoption_forecaster.synthetic.benchmark import Benchmark, run_pipeline_on_inputs
from adoption_forecaster.synthetic.sources import make_docs

logger = get_logger(__name__)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "eurostat"
MANIFEST = "manifest.json"
TIMESTAMP = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{4}"  # colon style, as the fetcher names its uploads
YEAR_COLUMN = re.compile(r"\d{4}")
KEY = ["country", "segment", "year"]
NOISE, WRONG_FRACTION, CONFLICT_FRACTION = 0.01, 0.2, 0.15  # same document settings as the synthetic benchmark


class MissingInputFilesError(RuntimeError):
    """The fetcher could not produce some expected input files (``missing`` holds ``"<Namespace>/<base name>"`` entries)."""

    def __init__(self, missing: list[str]):
        self.missing = missing
        super().__init__(f"Eurostat fetcher could not produce expected input files: {', '.join(missing)}")


def expected_input_files(config: dict) -> list[tuple[str, str]]:
    """(Namespace, base file name) for every input file listed under ``config["modules"][...]["files"]``."""
    modules = ("education", "age_gender")
    return [(module.title(), name.replace(".csv", "")) for module in modules for name in config["modules"][module]["files"].values()]


def cached_input_files(cache_dir: Path, config: dict) -> dict[tuple[str, str], Path]:
    """The latest cached file per expected input (absent keys are not cached)."""
    found = {}
    for namespace, base in expected_input_files(config):
        pattern = re.compile(rf"^{re.escape(base)}_({TIMESTAMP})\.csv$")
        input_dir = Path(cache_dir) / namespace / "input"
        matches = sorted((p for p in input_dir.glob("*.csv") if pattern.match(p.name)), key=lambda p: pattern.match(p.name).group(1))
        if matches:
            found[(namespace, base)] = matches[-1]
    return found


def _cache_complete(cache_dir: Path, last_year: int, config: dict) -> bool:
    manifest = cache_dir / MANIFEST
    if not manifest.exists() or json.loads(manifest.read_text())["last_year"] < last_year:
        return False
    return len(cached_input_files(cache_dir, config)) == len(expected_input_files(config))


def fetch_eurostat_inputs(cache_dir: Path, last_year: int, config: dict) -> Path:
    """Fill ``<cache_dir>/<Namespace>/input/`` with the fetcher's real inputs (via the GCS emulator) unless the cache is complete."""
    cache_dir = Path(cache_dir)
    if _cache_complete(cache_dir, last_year, config):
        logger.info(f"Using cached Eurostat inputs in {cache_dir}")
        return cache_dir
    if not os.environ.get("STORAGE_EMULATOR_HOST"):
        raise RuntimeError("Eurostat input cache is incomplete and STORAGE_EMULATOR_HOST is not set; refusing to run the fetcher against real GCS")

    client = storage.Client(project=config["defaults"]["project_id"], credentials=AnonymousCredentials())
    bucket_name = config["defaults"]["bucket_name"]
    bucket = client.lookup_bucket(bucket_name) or client.create_bucket(bucket_name)
    prefixes = {ns: f"{config['gcs']['data_source']}/{ns}/{config['gcs']['input_subfolder']}/" for ns, _ in expected_input_files(config)}
    for prefix in prefixes.values():  # emulator only: start clean so blobs left by an earlier run cannot stand in for this fetch
        for blob in bucket.list_blobs(prefix=prefix):
            blob.delete()

    ok = EurostatDataFetcher(last_year, config).fetch_data()
    logger.info(f"Eurostat fetcher finished, success={ok}")

    with tempfile.TemporaryDirectory() as staging:  # the cache changes only when every expected file was produced by this run
        downloaded = {}
        for namespace, base in expected_input_files(config):
            pattern = re.compile(rf"^{re.escape(base)}_({TIMESTAMP})\.csv$")
            blobs = [b for b in bucket.list_blobs(prefix=prefixes[namespace]) if "old_datasets" not in b.name and pattern.match(Path(b.name).name)]
            if blobs:
                latest = max(blobs, key=lambda b: pattern.match(Path(b.name).name).group(1))
                downloaded[(namespace, base)] = Path(staging) / f"{namespace}__{Path(latest.name).name}"
                latest.download_to_filename(str(downloaded[(namespace, base)]))
        missing = [f"{ns}/{base}" for ns, base in expected_input_files(config) if (ns, base) not in downloaded]
        if missing:
            raise MissingInputFilesError(missing)
        if not ok:
            raise RuntimeError("Eurostat fetcher reported failure (see its log); the cache was left unchanged")
        for (namespace, base), staged in downloaded.items():
            pattern = re.compile(rf"^{re.escape(base)}_({TIMESTAMP})\.csv$")
            input_dir = cache_dir / namespace / "input"
            input_dir.mkdir(parents=True, exist_ok=True)
            for old in input_dir.glob("*.csv"):
                if pattern.match(old.name):
                    old.unlink()
            shutil.move(str(staged), str(input_dir / staged.name.split("__", 1)[1]))
    fetched_at = datetime.now(timezone.utc).date().isoformat()
    (cache_dir / MANIFEST).write_text(json.dumps({"last_year": last_year, "fetched_at": fetched_at}, indent=2))
    return cache_dir


def _copy_inputs(cache_dir: Path, module: str, config: dict, data_dir: Path, last_year: int) -> None:
    """Copy the module's cached inputs to ``<data_dir>/<Namespace>/input`` without the year columns after ``last_year``."""
    namespace = module.title()
    input_dir = data_dir / namespace / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for (ns, _), path in cached_input_files(cache_dir, config).items():
        if ns != namespace:
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        df = df[[c for c in df.columns if not (YEAR_COLUMN.fullmatch(c) and int(c) > last_year)]]
        df.to_csv(input_dir / path.name, index=False)


def _truth_after_cut(baseline: ForecastTable, observed: ForecastTable) -> ForecastTable:
    """Observed values on the baseline's forecast keys (NaN where Eurostat published nothing)."""
    keys = baseline.loc[baseline["is_forecast"], KEY]
    values = observed.loc[~observed["is_forecast"], KEY + ["value"]]
    truth = keys.merge(values, on=KEY, how="left", validate="one_to_one")
    truth["is_forecast"] = True
    return truth[baseline.columns].reset_index(drop=True)


def build_real_backtest(cache_dir: Path, module: str, cut_year: int = 2022, pred_year: int = 2024, seed: int = 0) -> Benchmark:
    """Baseline = the real pipeline on inputs cut at ``cut_year``; truth = the published values for the years after ``cut_year``."""
    if module not in ("education", "age_gender"):
        raise ValueError(f"Unknown module: {module}. Must be 'education' or 'age_gender'")
    config = copy.deepcopy(load_config())
    gap = config["validation"]["year_difference"]
    if pred_year - cut_year != gap:
        raise ValueError(f"pred_year - cut_year must equal validation.year_difference ({gap})")
    cache_dir = fetch_eurostat_inputs(Path(cache_dir), pred_year, config)
    with tempfile.TemporaryDirectory() as tmp:
        cut_dir, full_dir = Path(tmp) / "cut", Path(tmp) / "full"
        _copy_inputs(cache_dir, module, config, cut_dir, cut_year)
        _copy_inputs(cache_dir, module, config, full_dir, pred_year)
        baseline = run_pipeline_on_inputs(cut_dir, module, cut_year, pred_year)
        observed = run_pipeline_on_inputs(full_dir, module, pred_year, pred_year + gap)
    truth = _truth_after_cut(baseline, observed)
    history = baseline[~baseline["is_forecast"]]  # real values the agent already sees; used only as distractor numbers in the docs
    docs = make_docs(pd.concat([history, truth], ignore_index=True), seed, NOISE, WRONG_FRACTION, CONFLICT_FRACTION)
    return Benchmark(module, None, baseline, truth, docs, cut_year, pred_year)
