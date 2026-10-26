"""The correction agent plugged into the real pipeline behind ``correction.enabled``.

The injected source store is built from a benchmark over the same seeded world the pipeline runs on (seed 7, all default countries,
2024 -> 2026), so its documents describe the very baseline the pipeline produces; the LLM is a ScriptedClient (no network, no key).
"""

import copy
import io
import logging
import os
from pathlib import Path

import pandas as pd
import pytest

from quotas_forecasting.correction import pipeline
from quotas_forecasting.correction.llm import LLMUnavailableError, MissingApiKeyError, ScriptedClient
from quotas_forecasting.correction.sources import SyntheticSourceStore
from quotas_forecasting.main import run_single_module
from quotas_forecasting.synthetic.benchmark import build_benchmark
from quotas_forecasting.synthetic.world import DEFAULT_COUNTRIES
from quotas_forecasting.synthetic.writers import write_age_gender_inputs, write_education_inputs
from tests.correction.scripting import plan, research_script
from tests.e2e.test_pipeline_local import AGE_GENDER_OUTPUTS, EDUCATION_OUTPUTS, GOLDEN_DIR
from tests.e2e.emulators import assert_uploaded_and_notified

PRED_YEAR = 2026
LAST_YEAR = 2024
WORLD_SEED = 7  # the seed of the `world` fixture in tests/conftest.py
COUNTRY = "Belgium"  # shocked in 2025 in this world and covered by documents
SEGMENTS = {"education": "secondary", "age_gender": "Male, 16 to 24"}
NAMESPACES = {"education": "Education", "age_gender": "Age_Gender"}
OUTPUTS = {"education": EDUCATION_OUTPUTS, "age_gender": AGE_GENDER_OUTPUTS}


@pytest.fixture(scope="module")
def benches():
    return {module: build_benchmark(WORLD_SEED, module, n_countries=len(DEFAULT_COUNTRIES)) for module in SEGMENTS}


def _enabled(config):
    config = copy.deepcopy(config)
    config["correction"]["enabled"] = True
    config["correction"]["model"] = "test/model"
    # the scripted corrections fix the synthetic world's shocks (0.03-0.08), which exceed the adaptive yoy caps of its smooth history
    config["correction"]["yoy_cap"]["mode"] = "fixed"
    return config


def _write_inputs(world, config):
    data_dir = Path(config["defaults"]["local_data_dir"])
    write_education_inputs(world, data_dir / "Education" / "input")
    write_age_gender_inputs(world, data_dir / "Age_Gender" / "input")


def _run(module, config, mode="local", **kwargs):
    run_single_module(module, mode, config["defaults"]["bucket_name"], config["defaults"]["project_id"], PRED_YEAR, LAST_YEAR, config, **kwargs)


def _read(path_or_buffer):
    df = pd.read_csv(path_or_buffer, index_col=0)
    return df.set_index("Individual type", append=True) if "Individual type" in df.columns else df


def _changed_cells(actual: pd.DataFrame, golden: pd.DataFrame) -> set[tuple]:
    """(row, column) cells whose numeric values differ (NaN and ':' compare equal to themselves)."""
    assert actual.index.equals(golden.index) and list(actual.columns) == list(golden.columns)
    a, g = actual.apply(pd.to_numeric, errors="coerce"), golden.apply(pd.to_numeric, errors="coerce")
    assert (a.isna() == g.isna()).all().all()
    diff = ((a - g).abs() > 1e-12) & a.notna()
    return {(row, col) for col in diff.columns for row in diff.index[diff[col]]}


def _script(module, bench, store):
    """One researched suspect for the (first) baseline; education's second age range (16-74) gets an empty plan."""
    script = [plan((COUNTRY, SEGMENTS[module]))] + research_script(bench, store, COUNTRY, SEGMENTS[module])
    return script + [plan()] if module == "education" else script


def _expected_cells(module, corrections, file_name):
    cells = set()
    for c in corrections:
        if module == "education":
            cols = [f"prediction-secondary-{c.year}", f"prediction-tertiary-{c.year}"]
            cells |= {(c.country, col) for col in cols}
        else:
            cells.add(((c.country, c.segment), f"prediction-{c.year}"))
    return {cell for cell in cells if "with_past_data" in file_name or str(PRED_YEAR) in cell[1]}


@pytest.fixture
def spy(monkeypatch):
    """Record every CorrectionResult the pipeline produces."""
    results = []
    original = pipeline.run_correction

    def wrapper(*args, **kwargs):
        results.append(original(*args, **kwargs))
        return results[-1]

    monkeypatch.setattr(pipeline, "run_correction", wrapper)
    return results


def _boom(*args, **kwargs):
    raise AssertionError("correction code must not be called with the flag off")


def test_benchmark_world_is_the_pipeline_world(benches, world):
    for bench in benches.values():
        assert bench.world.countries == world.countries
        pd.testing.assert_frame_equal(bench.world.truth_education, world.truth_education)
        pd.testing.assert_frame_equal(bench.world.truth_age_gender, world.truth_age_gender)


def test_flag_off_matches_golden(world, local_config, monkeypatch):
    assert local_config["correction"]["enabled"] is False
    for name in ("run_correction", "get_llm_client", "get_source_store", "to_table", "from_table"):
        monkeypatch.setattr(pipeline, name, _boom)
    _write_inputs(world, local_config)
    for module in SEGMENTS:
        _run(module, local_config)
        out_dir = Path(local_config["defaults"]["local_data_dir"]) / NAMESPACES[module] / "output"
        for name in OUTPUTS[module]:
            pd.testing.assert_frame_equal(pd.read_csv(out_dir / name, index_col=0), pd.read_csv(GOLDEN_DIR / name, index_col=0), rtol=1e-6)


@pytest.mark.parametrize("module", list(SEGMENTS))
def test_flag_on_with_scripted_llm_changes_only_cited_cells(module, benches, world, local_config, spy):
    config = _enabled(local_config)
    bench = benches[module]
    store = SyntheticSourceStore(bench.docs)
    script = _script(module, bench, store)
    llm = ScriptedClient(script)
    _write_inputs(world, config)
    _run(module, config, correction_store=store, correction_llm=llm)

    assert len(llm.calls) == len(script)  # the whole script was consumed
    assert len(spy) == (2 if module == "education" else 1) and all(r.error is None for r in spy)
    corrections = spy[0].corrections
    assert [(c.country, c.segment, c.year) for c in corrections] == [(COUNTRY, SEGMENTS[module], 2025), (COUNTRY, SEGMENTS[module], 2026)]
    assert spy[0].rejections == []

    out_dir = Path(config["defaults"]["local_data_dir"]) / NAMESPACES[module] / "output"
    for name in OUTPUTS[module]:
        actual, golden = _read(out_dir / name), _read(GOLDEN_DIR / name)
        first_baseline = module == "age_gender" or "16-64" in name
        expected = _expected_cells(module, corrections, name) if first_baseline else set()
        assert _changed_cells(actual, golden) == expected, name
        for c in corrections if first_baseline else []:
            if module == "education":
                if "with_past_data" in name or c.year == PRED_YEAR:
                    secondary = actual.at[c.country, f"prediction-secondary-{c.year}"]
                    assert secondary == pytest.approx(c.new_value)
                    assert secondary + actual.at[c.country, f"prediction-tertiary-{c.year}"] == pytest.approx(1.0)
            elif "with_past_data" in name or c.year == PRED_YEAR:
                assert actual.at[(c.country, c.segment), f"prediction-{c.year}"] == pytest.approx(c.new_value)
        if module == "education":
            for year in (2025, PRED_YEAR):
                if f"prediction-secondary-{year}" in actual.columns:
                    total = actual[f"prediction-secondary-{year}"] + actual[f"prediction-tertiary-{year}"]
                    assert ((total - 1.0).abs() < 1e-9)[total.notna()].all()


def test_missing_key_with_flag_on_raises_at_startup_and_message_has_no_secret(local_config, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr("quotas_forecasting.main.get_predictor", _boom)  # forecasting must not start
    monkeypatch.setattr("quotas_forecasting.main.Loader", _boom)
    config = _enabled(local_config)
    for module in SEGMENTS:
        with pytest.raises(MissingApiKeyError) as excinfo:
            _run(module, config, correction_store=SyntheticSourceStore([]))
        assert str(excinfo.value) == "NVIDIA_API_KEY environment variable is not set"
        assert excinfo.value.__cause__ is None
    assert not any(Path(config["defaults"]["local_data_dir"]).rglob("*.csv"))


def test_flag_on_without_a_store_fails_fast_with_a_clear_message(local_config, monkeypatch):
    monkeypatch.setattr("quotas_forecasting.main.get_predictor", _boom)
    with pytest.raises(pipeline.NoSourceStoreError, match="no source store is configured; real sources arrive in milestone 2"):
        _run("education", _enabled(local_config), correction_llm=ScriptedClient([]))


class _Unavailable:
    def chat(self, messages, tools):
        raise LLMUnavailableError("LLM request failed: APIConnectionError")


def test_agent_failure_falls_back_to_baseline_and_writes_outputs(world, local_config, benches, spy, caplog):
    config = _enabled(local_config)
    _write_inputs(world, config)
    with caplog.at_level(logging.INFO):
        for module in SEGMENTS:
            _run(module, config, correction_store=SyntheticSourceStore(benches[module].docs), correction_llm=_Unavailable())
            out_dir = Path(config["defaults"]["local_data_dir"]) / NAMESPACES[module] / "output"
            for name in OUTPUTS[module]:
                pd.testing.assert_frame_equal(pd.read_csv(out_dir / name, index_col=0), pd.read_csv(GOLDEN_DIR / name, index_col=0), rtol=1e-6)

    assert len(spy) == 3 and all("LLMUnavailableError" in r.error for r in spy)
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR and "LLMUnavailableError" in r.getMessage()]
    assert any("education" in m for m in errors) and any("age_gender" in m for m in errors)


@pytest.mark.emulators
@pytest.mark.skipif(
    not (os.environ.get("STORAGE_EMULATOR_HOST") and os.environ.get("PUBSUB_EMULATOR_HOST")),
    reason="needs STORAGE_EMULATOR_HOST and PUBSUB_EMULATOR_HOST (see docker-compose.emulators.yml)",
)
def test_corrected_prod_run_notifies_with_emulators(emulator_env, benches, spy):
    config = _enabled(emulator_env["config"])
    emulator_env["config"] = config
    for module in SEGMENTS:
        store = SyntheticSourceStore(benches[module].docs)
        _run(module, config, mode="prod", correction_store=store, correction_llm=ScriptedClient(_script(module, benches[module], store)))
    assert_uploaded_and_notified(emulator_env)

    corrected = {"education": spy[0].corrections, "age_gender": spy[2].corrections}
    assert all(len(c) == 2 for c in corrected.values())
    bucket = emulator_env["client"].bucket(config["defaults"]["bucket_name_external"])
    for module, name in (("education", EDUCATION_OUTPUTS[1]), ("age_gender", AGE_GENDER_OUTPUTS[1])):
        external_name = f"{config['defaults']['dir_external']}/{name.replace('.csv', '_both.csv')}"
        uploaded = _read(io.StringIO(bucket.blob(external_name).download_as_bytes().decode("utf-8-sig")))
        assert _changed_cells(uploaded, _read(GOLDEN_DIR / name)) == _expected_cells(module, corrected[module], name)
