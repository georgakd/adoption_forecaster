"""pygam is an optional ("experiments" group) dependency: the production image never installs it, and imports stay lazy."""

import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_pygam_is_in_an_optional_group_not_in_main_dependencies():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    poetry = pyproject["tool"]["poetry"]
    assert "pygam" not in poetry["dependencies"]
    group = poetry["group"]["experiments"]
    assert group["optional"] is True
    assert {"pygam", "threadpoolctl"} <= set(group["dependencies"])


def test_modules_import_without_pygam_and_the_gam_raises_a_clear_error():
    code = """
import sys
sys.modules["pygam"] = None  # makes `import pygam` raise ImportError
import pandas as pd
import quotas_forecasting.backtest.compare
from quotas_forecasting.forecasting.experimental.gam import gam_forecast
try:
    gam_forecast(pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=range(2017, 2022)), horizon=2)
except ImportError as exc:
    print("IMPORT_ERROR:", exc)
"""
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "IMPORT_ERROR:" in result.stdout and "poetry install --with experiments" in result.stdout
