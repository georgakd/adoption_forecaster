import copy
from pathlib import Path

import pytest

from quotas_forecasting.forecasting import predictors
from quotas_forecasting.main import run_single_module
from quotas_forecasting.synthetic.writers import write_age_gender_inputs, write_education_inputs

PRED_YEAR = 2026
LAST_YEAR = 2024


@pytest.fixture
def captured_frames(world, local_config, monkeypatch):
    """Run both real pipelines on the synthetic world and capture the frames returned by the predictors."""
    captured = {}

    def wrap(cls, name, key):
        original = getattr(cls, name)

        def wrapper(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            captured.setdefault(key, []).append(copy.deepcopy(result))
            return result

        monkeypatch.setattr(cls, name, wrapper)

    wrap(predictors.EducationPredictor, "predict", "education")
    wrap(predictors.AgeGenderPredictor, "run_with_data", "age_gender")
    data_dir = Path(local_config["defaults"]["local_data_dir"])
    write_education_inputs(world, data_dir / "Education" / "input")
    write_age_gender_inputs(world, data_dir / "Age_Gender" / "input")
    for module in ("education", "age_gender"):
        run_single_module(module, "local", "unused-bucket", "unused-project", PRED_YEAR, LAST_YEAR, local_config)
    return captured
