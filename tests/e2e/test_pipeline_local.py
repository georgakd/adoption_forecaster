import os
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from adoption_forecaster.main import run_single_module
from adoption_forecaster.synthetic.writers import write_age_gender_inputs, write_education_inputs

PRED_YEAR = 2026
LAST_YEAR = 2024
GOLDEN_DIR = Path(__file__).resolve().parents[1] / "golden"

EDUCATION_OUTPUTS = [
    f"{PRED_YEAR}_Education_16-64_predictions.csv",
    f"{PRED_YEAR}_Education_16-64_predictions_with_past_data.csv",
    f"{PRED_YEAR}_Education_16-74_predictions.csv",
    f"{PRED_YEAR}_Education_16-74_predictions_with_past_data.csv",
]
AGE_GENDER_OUTPUTS = [
    f"{PRED_YEAR}_Age_Gender_16-74_predictions.csv",
    f"{PRED_YEAR}_Age_Gender_16-74_predictions_with_past_data.csv",
]
PREDICTION_OUTPUTS = [EDUCATION_OUTPUTS[0], EDUCATION_OUTPUTS[2], AGE_GENDER_OUTPUTS[0]]


def _run(module, config):
    run_single_module(module, "local", "unused-bucket", "unused-project", PRED_YEAR, LAST_YEAR, config)


def _out_dir(config, namespace):
    return Path(config["defaults"]["local_data_dir"]) / namespace / "output"


@pytest.fixture
def education_outputs(world, local_config):
    write_education_inputs(world, Path(local_config["defaults"]["local_data_dir"]) / "Education" / "input")
    _run("education", local_config)
    return _out_dir(local_config, "Education")


@pytest.fixture
def age_gender_outputs(world, local_config):
    write_age_gender_inputs(world, Path(local_config["defaults"]["local_data_dir"]) / "Age_Gender" / "input")
    _run("age_gender", local_config)
    return _out_dir(local_config, "Age_Gender")


def _assert_has_data(path):
    assert path.exists(), f"missing output {path}"
    df = pd.read_csv(path, index_col=0)
    assert len(df) > 0 and df.shape[1] > 0, f"empty output {path}"
    return df


def test_education_pipeline_writes_expected_outputs(education_outputs):
    for name in EDUCATION_OUTPUTS:
        _assert_has_data(education_outputs / name)


def test_age_gender_pipeline_writes_expected_outputs(age_gender_outputs):
    for name in AGE_GENDER_OUTPUTS:
        _assert_has_data(age_gender_outputs / name)


def test_outputs_in_unit_interval(education_outputs, age_gender_outputs):
    for name in PREDICTION_OUTPUTS:
        directory = age_gender_outputs if name.startswith(f"{PRED_YEAR}_Age_Gender") else education_outputs
        values = _assert_has_data(directory / name).select_dtypes("number")
        assert values.notna().any().any()
        assert ((values.dropna() >= 0) & (values.dropna() <= 1)).all().all(), name


@pytest.mark.parametrize("name", EDUCATION_OUTPUTS + AGE_GENDER_OUTPUTS)
def test_outputs_match_golden(name, education_outputs, age_gender_outputs):
    directory = age_gender_outputs if name.startswith(f"{PRED_YEAR}_Age_Gender") else education_outputs
    actual = _assert_has_data(directory / name)
    golden_path = GOLDEN_DIR / name
    if not golden_path.exists() and os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN_DIR.mkdir(exist_ok=True)
        actual.to_csv(golden_path)
    assert golden_path.exists(), f"golden missing: {golden_path} (run with UPDATE_GOLDEN=1)"
    assert_frame_equal(actual, pd.read_csv(golden_path, index_col=0), rtol=1e-6)
