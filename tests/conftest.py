import copy

import pytest

from quotas_forecasting.config.config_loader import load_config
from quotas_forecasting.synthetic.world import generate_world

WORLD_SEED = 7
LAST_YEAR = 2024
PRED_YEAR = 2026


@pytest.fixture
def local_config(tmp_path):
    config = copy.deepcopy(load_config())
    config["defaults"]["local_data_dir"] = str(tmp_path)
    config["defaults"]["auto_fetch"] = False
    config["pubsub"]["enable_notifications"] = False
    return config


@pytest.fixture(scope="session")
def world():
    return generate_world(WORLD_SEED, last_year=LAST_YEAR, pred_year=PRED_YEAR)
