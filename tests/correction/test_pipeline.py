import copy

import pandas as pd
import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction import pipeline
from quotas_forecasting.correction.llm import ScriptedClient


@pytest.fixture
def frame():
    index = pd.MultiIndex.from_tuples([("France", "secondary"), ("France", "tertiary")], names=["country", "education"])
    return pd.DataFrame({2024: [0.4, 0.6], 2025: [0.39, 0.61]}, index=index)


def _boom(*args, **kwargs):
    raise AssertionError("must not be called with the flag off")


def test_flag_off_returns_the_same_object_and_touches_nothing(frame, monkeypatch):
    for name in ("run_correction", "get_llm_client", "get_source_store", "to_table", "from_table"):
        monkeypatch.setattr(pipeline, name, _boom)
    config = load_config()
    assert pipeline.maybe_correct(frame, "education", 2024, config, store=None) is frame
    assert pipeline.prepare_correction(config, None, None) == (None, None)


def test_unexpected_failure_falls_back_to_the_baseline(frame, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("store exploded")

    monkeypatch.setattr(pipeline, "run_correction", fail)
    config = copy.deepcopy(load_config())
    config["correction"]["enabled"] = True
    assert pipeline.maybe_correct(frame, "education", 2024, config, store=None, llm=ScriptedClient([])) is frame
