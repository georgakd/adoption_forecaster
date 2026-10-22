"""Offline checks of the Eurostat backtest cache guard (no network, no emulator)."""

import copy

import pytest
import requests

from quotas_forecasting.backtest import eurostat_backtest
from quotas_forecasting.backtest.eurostat_backtest import fetch_eurostat_inputs
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
