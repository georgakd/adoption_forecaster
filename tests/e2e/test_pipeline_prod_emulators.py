import os

import pytest

from adoption_forecaster.main import run_single_module
from tests.e2e.emulators import assert_uploaded_and_notified

PRED_YEAR = 2026
LAST_YEAR = 2024

pytestmark = [
    pytest.mark.emulators,
    pytest.mark.skipif(
        not (os.environ.get("STORAGE_EMULATOR_HOST") and os.environ.get("PUBSUB_EMULATOR_HOST")),
        reason="needs STORAGE_EMULATOR_HOST and PUBSUB_EMULATOR_HOST (see docker-compose.emulators.yml)",
    ),
]


def test_prod_pipeline_uploads_and_notifies(emulator_env):
    cfg = emulator_env["config"]
    for module in ("education", "age_gender"):
        run_single_module(module, "prod", cfg["defaults"]["bucket_name"], cfg["defaults"]["project_id"], PRED_YEAR, LAST_YEAR, cfg)
    assert_uploaded_and_notified(emulator_env)
