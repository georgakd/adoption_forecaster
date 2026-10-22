"""Offline checks of the Eurostat backtest cache guard (no network, no emulator)."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import requests

from quotas_forecasting.backtest import eurostat_backtest
from quotas_forecasting.backtest.eurostat_backtest import (
    MANIFEST,
    MissingInputFilesError,
    cached_input_files,
    expected_input_files,
    fetch_eurostat_inputs,
)
from quotas_forecasting.config.config_loader import load_config


def _forbidden(*args, **kwargs):
    raise AssertionError("network or GCS access attempted")


def test_incomplete_cache_without_emulator_refuses_to_run_the_fetcher(tmp_path, monkeypatch):
    monkeypatch.delenv("STORAGE_EMULATOR_HOST", raising=False)
    monkeypatch.setattr(requests, "get", _forbidden)
    monkeypatch.setattr(eurostat_backtest.storage, "Client", _forbidden)
    monkeypatch.setattr(eurostat_backtest, "EurostatDataFetcher", _forbidden)
    with pytest.raises(RuntimeError, match="STORAGE_EMULATOR_HOST"):
        fetch_eurostat_inputs(tmp_path, 2024, copy.deepcopy(load_config()))


OLD_TS = "2020-01-01T00:00:00+0000"
NEW_TS = "2026-01-01T00:00:00+0000"
OLD_MANIFEST = {"last_year": 2024, "fetched_at": "2000-01-01"}


class FakeBlob:
    def __init__(self, bucket, name):
        self.bucket, self.name = bucket, name

    def delete(self):
        del self.bucket.blobs[self.name]

    def download_to_filename(self, path):
        Path(path).write_text(self.bucket.blobs[self.name])


class FakeBucket:
    def __init__(self):
        self.blobs = {}

    def list_blobs(self, prefix=""):
        return [FakeBlob(self, n) for n in list(self.blobs) if n.startswith(prefix)]


@pytest.fixture
def fake_gcs(monkeypatch):
    """Stand-ins for the emulator bucket and the fetcher; ``state`` says which bases the fetcher uploads and what it returns."""
    config = copy.deepcopy(load_config())
    bucket = FakeBucket()
    state = {"produce": list(expected_input_files(config)), "ok": True, "calls": 0}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        def lookup_bucket(self, name):
            return bucket

    class FakeFetcher:
        def __init__(self, last_year, config):
            self.config = config

        def fetch_data(self):
            state["calls"] += 1
            source = self.config["gcs"]["data_source"]
            for namespace, base in state["produce"]:
                bucket.blobs[f"{source}/{namespace}/input/{base}_{NEW_TS}.csv"] = f"new {base}"
            return state["ok"]

    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://fake-emulator")
    monkeypatch.setattr(requests, "get", _forbidden)
    monkeypatch.setattr(eurostat_backtest.storage, "Client", FakeClient)
    monkeypatch.setattr(eurostat_backtest, "EurostatDataFetcher", FakeFetcher)
    return config, state


def _stale_cache(cache_dir, config):
    for namespace, base in expected_input_files(config):
        input_dir = cache_dir / namespace / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / f"{base}_{OLD_TS}.csv").write_text(f"old {base}")
    (cache_dir / MANIFEST).write_text(json.dumps(OLD_MANIFEST))


def _cache_state(cache_dir):
    return sorted((str(p.relative_to(cache_dir)), p.read_text()) for p in cache_dir.rglob("*") if p.is_file())


def test_base_not_produced_this_run_raises_even_with_a_stale_cached_copy(tmp_path, fake_gcs):
    config, state = fake_gcs
    _stale_cache(tmp_path, config)
    before = _cache_state(tmp_path)
    missing = state["produce"].pop()
    with pytest.raises(MissingInputFilesError) as exc:
        fetch_eurostat_inputs(tmp_path, 2025, config)
    assert exc.value.missing == [f"{missing[0]}/{missing[1]}"] and missing[1] in str(exc.value)
    assert _cache_state(tmp_path) == before  # neither the files nor the manifest change on a failed fetch


def test_fetcher_reporting_failure_raises(tmp_path, fake_gcs):
    config, state = fake_gcs
    _stale_cache(tmp_path, config)
    before = _cache_state(tmp_path)
    state["ok"] = False
    with pytest.raises(RuntimeError, match="fetcher reported failure"):
        fetch_eurostat_inputs(tmp_path, 2025, config)
    assert _cache_state(tmp_path) == before


def test_all_bases_fetched_this_run_replace_the_cache_and_write_the_manifest(tmp_path, fake_gcs):
    config, state = fake_gcs
    _stale_cache(tmp_path, config)
    assert fetch_eurostat_inputs(tmp_path, 2025, config) == tmp_path
    manifest = json.loads((tmp_path / MANIFEST).read_text())
    assert manifest == {"last_year": 2025, "fetched_at": datetime.now(timezone.utc).date().isoformat()}
    found = cached_input_files(tmp_path, config)
    assert set(found) == set(expected_input_files(config))
    assert all(p.name.endswith(f"_{NEW_TS}.csv") and p.read_text() == f"new {base}" for (_, base), p in found.items())
    assert len(list(tmp_path.rglob("*.csv"))) == len(found)  # the stale versions are gone


def test_complete_cache_is_reused_without_fetcher_or_gcs(tmp_path, fake_gcs):
    config, state = fake_gcs
    _stale_cache(tmp_path, config)
    before = _cache_state(tmp_path)
    assert fetch_eurostat_inputs(tmp_path, 2024, config) == tmp_path
    assert state["calls"] == 0 and _cache_state(tmp_path) == before
