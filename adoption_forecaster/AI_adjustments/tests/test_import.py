import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("name", ["agent", "client", "tools", "rules", "workflow_enforcer"])
def test_modules_import_from_any_cwd(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    importlib.import_module(f"adoption_forecaster.AI_adjustments.{name}")


def test_cli_help_runs(tmp_path):
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-m", "adoption_forecaster.AI_adjustments.main", "--help"],
        cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
