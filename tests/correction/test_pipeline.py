import copy
import logging

import pandas as pd
import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.correction import pipeline
from quotas_forecasting.correction.agent import CorrectionResult
from quotas_forecasting.correction.llm import ScriptedClient
from quotas_forecasting.correction.types import Citation, Correction


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


def test_each_accepted_correction_is_logged_at_info(frame, monkeypatch, caplog):
    quote = "In 2025 the share of internet users with secondary education in France was 0.420. " + "x" * 300
    correction = Correction("France", "secondary", 2025, 0.39, 0.42, Citation("fr-25", quote, 0.42, 2025, "share"), 0.8, "published")

    def run(table, module, store, llm, config):
        corrected = table.copy()
        corrected.loc[(corrected["segment"] == "secondary") & (corrected["year"] == 2025), "value"] = 0.42
        corrected.loc[(corrected["segment"] == "tertiary") & (corrected["year"] == 2025), "value"] = 0.58
        return CorrectionResult(corrected, [correction], [])

    monkeypatch.setattr(pipeline, "run_correction", run)
    config = copy.deepcopy(load_config())
    config["correction"]["enabled"] = True
    with caplog.at_level(logging.INFO, logger="quotas_forecasting.correction.pipeline"):
        pipeline.maybe_correct(frame, "education", 2024, config, store=None, llm=ScriptedClient([]))
    lines = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO and "fr-25" in r.getMessage()]
    assert len(lines) == 1
    line = lines[0]
    assert "France" in line and "secondary" in line and "2025" in line and "0.3900 -> 0.4200" in line and "confidence=0.8" in line
    assert quote[:200] in line and quote[:201] not in line


@pytest.mark.parametrize(
    "override,message",
    [
        ({"mode": "sliding"}, "mode"),
        ({"quantile": 1.5}, "quantile"),
        ({"quantile": -0.1}, "quantile"),
        ({"k": 0}, "k"),
        ({"floor": -0.01}, "floor"),
        ({"floor": 0.4, "ceiling": 0.3}, "floor"),
    ],
)
def test_prepare_correction_rejects_invalid_yoy_cap_settings_before_building_anything(monkeypatch, override, message):
    monkeypatch.setattr(pipeline, "get_llm_client", _boom)
    config = copy.deepcopy(load_config())
    config["correction"]["enabled"] = True
    config["correction"]["yoy_cap"].update(override)
    with pytest.raises(ValueError, match=rf"correction\.yoy_cap.*{message}"):
        pipeline.prepare_correction(config, None, None)


def test_prepare_correction_accepts_the_default_yoy_cap_settings():
    config = copy.deepcopy(load_config())
    config["correction"]["enabled"] = True
    store, llm = object(), ScriptedClient([])
    assert pipeline.prepare_correction(config, store, llm) == (store, llm)
