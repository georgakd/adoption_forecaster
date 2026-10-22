import re

import numpy as np
import pandas as pd
import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.data_handlers.loader import Loader
from quotas_forecasting.data_handlers.missing_data_handlers import MissingDataHandler
from quotas_forecasting.synthetic.world import generate_world
from quotas_forecasting.synthetic.writers import write_age_gender_inputs, write_education_inputs

TIMESTAMP = r"\d{4}-\d{2}-\d{2}T\d{2}_\d{2}_\d{2}\+0000"


@pytest.fixture(scope="module")
def config():
    return load_config()


def test_generate_world_is_deterministic_by_seed():
    a, b, c = generate_world(1), generate_world(1), generate_world(2)
    assert a.countries == b.countries
    assert a.truth_education.equals(b.truth_education)
    assert a.truth_age_gender.equals(b.truth_age_gender)
    assert a.shocks == b.shocks
    assert not a.truth_education.equals(c.truth_education)


def test_truth_stays_in_unit_interval_and_education_sums_to_one():
    for seed in range(5):
        w = generate_world(seed)
        values = w.truth_education.to_numpy()
        assert values.min() >= 0 and values.max() <= 1
        ag = w.truth_age_gender.drop(columns=["country", "Individual type"]).to_numpy()
        assert ag.min() >= 0 and ag.max() <= 1
        sec = w.truth_education.xs("secondary", level="education")
        ter = w.truth_education.xs("tertiary", level="education")
        assert np.abs(sec + ter - 1).to_numpy().max() < 1e-9


def test_shocks_only_affect_years_after_last_year():
    w = generate_world(3, shocked_fraction=0.5)
    assert w.shocks
    unshocked = generate_world(3, shocked_fraction=0.0)
    assert not unshocked.shocks
    for country in w.countries:
        a = w.truth_education.loc[(country, "tertiary")]
        b = unshocked.truth_education.loc[(country, "tertiary")]
        if country not in w.shocks:
            assert np.allclose(a, b)
            continue
        shock = w.shocks[country]
        assert shock.year > w.last_year
        diff = a - b
        assert np.allclose(diff[[y for y in diff.index if y < shock.year]], 0)
        assert np.allclose(diff[[y for y in diff.index if y >= shock.year]], shock.delta)
    # history is smooth: no jump bigger than 0.05 up to last_year
    hist = w.truth_education.xs("tertiary", level="education").loc[:, : w.last_year]
    assert hist.diff(axis=1).abs().max().max() < 0.05


def test_written_inputs_match_loader_file_patterns(tmp_path, config):
    w = generate_world(1)
    paths = write_education_inputs(w, tmp_path) + write_age_gender_inputs(w, tmp_path)
    bases = [f for m in ("education", "age_gender") for f in config["modules"][m]["files"].values()]
    assert len(paths) == len(bases) == 4
    for base in bases:
        matches = [p for p in paths if re.fullmatch(rf"{re.escape(base)}_{TIMESTAMP}\.csv", p.name)]
        assert len(matches) == 1
    for p in paths:
        df = pd.read_csv(p)
        years = [int(c) for c in df.columns if re.fullmatch(r"20\d\d", str(c))]
        assert years and max(years) == w.last_year


def _loader(tmp_path, config, namespace):
    loader = Loader.__new__(Loader)
    loader.namespace, loader.mode, loader.config, loader.local_path = namespace, "local", config, str(tmp_path)
    return loader


def test_written_inputs_parse_with_missing_data_handler(tmp_path, config):
    w = generate_world(1)
    write_education_inputs(w, tmp_path)
    write_age_gender_inputs(w, tmp_path)
    proc = config["modules"]["education"]["processing"]
    handler = MissingDataHandler(config)
    df = handler.load_and_process_education_data(
        _loader(tmp_path, config, "Education"), proc["regex_patterns"], proc["country_mapping"], "16_64"
    )
    assert df.index.names == ["country", "education"]
    assert list(df.columns) == list(range(2004, w.last_year + 1))
    assert "Czech Republic" in df.index.get_level_values("country")
    merged = MissingDataHandler.merge_education_levels(df.map(lambda v: np.nan if v is pd.NA else v).astype(float))
    last = merged[w.last_year].unstack("education")
    truth_last = w.truth_education[w.last_year].unstack("education")
    assert np.allclose(last.loc[truth_last.index, "tertiary"], truth_last["tertiary"], atol=2e-3)

    ag = _loader(tmp_path, config, "Age_Gender").read_csv_timestamped(config["modules"]["age_gender"]["files"]["age_gender"])
    ag = MissingDataHandler.standardize_countries(ag, "country", proc["country_mapping"])
    start = config["modules"]["age_gender"]["settings"]["start_year"]
    ag = ag[["country", "Individual type"] + [str(y) for y in range(start, w.last_year + 1)]]
    assert set(ag["country"]) == set(w.countries)
    from quotas_forecasting.forecasting.processors import AgeGenderProcessor

    p = AgeGenderProcessor(config=config)
    assert p.extract_age_groups(ag["Individual type"].iloc[0])
    assert ag.apply(p.extract_gender, axis=1)["gender"].isin(["male", "female"]).all()
