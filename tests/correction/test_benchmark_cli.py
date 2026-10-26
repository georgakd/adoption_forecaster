import copy
import functools
import json

import pytest

from adoption_forecaster.config.config_loader import load_config
from adoption_forecaster.correction import benchmark_cli
from adoption_forecaster.correction.llm import LLMResponse, ScriptedClient, get_llm_client  # noqa: F401
from adoption_forecaster.correction.oracle import build_oracle_client
from adoption_forecaster.synthetic.benchmark import build_benchmark

ARGS = ["--module", "education", "--seeds", "0", "--n-countries", "5"]


@pytest.fixture(autouse=True)
def fixed_yoy_cap(monkeypatch):
    # the oracle corrects the synthetic world's shocks (0.03-0.08), which exceed the adaptive yoy caps of its smooth history;
    # the fixed scalar cap keeps these tests about the CLI (the adaptive path is covered in test_caps.py)
    config = copy.deepcopy(load_config())
    config["correction"]["yoy_cap"]["mode"] = "fixed"
    monkeypatch.setattr(benchmark_cli, "load_config", lambda: config)


def _results(out_dir):
    files = list(out_dir.glob("*.json"))
    assert len(files) == 1
    return json.loads(files[0].read_text())


def test_stub_mode_gate_passes_with_oracle_script(tmp_path, capsys):
    code = benchmark_cli.main([*ARGS, "--out", str(tmp_path)])
    data = _results(tmp_path)
    agg = data["aggregate"]
    assert code == 0
    assert agg["mae_after"] < agg["mae_before"]
    assert agg["hallucinated_citations"] == 0 and agg["violations"] == 0 and agg["n_corrections"] > 0
    assert data["config"]["mode"] == "stub" and data["config"]["seeds"] == [0] and data["config"]["n_countries"] == 5
    assert data["per_seed"][0]["seed"] == 0 and data["per_seed"][0]["error"] is None
    assert data["per_seed"][0]["baseline_violations"] == 0 and agg["baseline_violations"] == 0
    out = capsys.readouterr().out
    assert "mae_before" in out and "base_viol" in out


def test_accepted_corrections_are_persisted_per_seed(tmp_path):
    benchmark_cli.main([*ARGS, "--out", str(tmp_path)])
    seed = _results(tmp_path)["per_seed"][0]
    assert seed["n_corrections"] > 0 and len(seed["corrections"]) == seed["n_corrections"]
    for record in seed["corrections"]:
        assert set(record) == {"module", "country", "segment", "year", "old_value", "new_value", "doc_id", "quote", "confidence"}
        assert record["module"] == "education" and record["old_value"] != record["new_value"] and len(record["quote"]) <= 200


def test_stub_mode_gate_fails_when_script_hallucinates(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_cli, "build_oracle_client", functools.partial(build_oracle_client, fabricate=True))
    assert benchmark_cli.main([*ARGS, "--out", str(tmp_path)]) != 0


def test_live_flag_without_key_exits_with_clear_message(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    code = benchmark_cli.main([*ARGS, "--live", "--out", str(tmp_path)])
    out = capsys.readouterr()
    assert code == 2
    assert "NVIDIA_API_KEY" in out.out + out.err and "nvapi" not in out.out + out.err
    assert not list(tmp_path.glob("*.json"))


def test_seed_range_parsing():
    assert benchmark_cli.parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert benchmark_cli.parse_seeds("1,5,7") == [1, 5, 7]
    assert benchmark_cli.parse_seeds("0-1,9") == [0, 1, 9]
    with pytest.raises(ValueError):
        benchmark_cli.parse_seeds("4-1")


def test_json_has_no_secret_when_env_key_is_set(tmp_path, monkeypatch, capsys):
    secret = "nvapi-test-secret"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    fake = ScriptedClient([LLMResponse(text='{"suspects": []}')])
    monkeypatch.setattr(benchmark_cli, "get_llm_client", lambda config: fake)
    code = benchmark_cli.main([*ARGS, "--live", "--model", "some/model", "--out", str(tmp_path)])
    out = capsys.readouterr()
    data = _results(tmp_path)
    assert code == 0  # live mode reports the gate, it does not fail on a missed target
    assert data["config"]["mode"] == "live" and data["config"]["model"] == "some/model" and data["llm_calls"] == 1
    assert secret not in out.out + out.err
    assert secret not in (next(tmp_path.glob("*.json"))).read_text()
    assert "NVIDIA_API_KEY" not in (next(tmp_path.glob("*.json"))).read_text()


def test_real_flag_runs_stub_oracle_on_the_backtest_and_records_years(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_backtest(cache_dir, module, cut_year, pred_year, seed):
        calls.append((module, cut_year, pred_year, seed))
        bench = build_benchmark(seed, module, n_countries=5)
        bench.world = None  # real backtests have no synthetic world
        return bench

    monkeypatch.setattr(benchmark_cli, "build_real_backtest", fake_backtest)
    code = benchmark_cli.main(["--module", "education", "--real", "--seeds", "0", "--out", str(tmp_path)])
    data = _results(tmp_path)
    assert code == 0 and calls == [("education", 2022, 2024, 0)]
    assert data["config"]["source"] == "real-backtest" and data["config"]["cut_year"] == 2022 and data["config"]["pred_year"] == 2024
    assert data["config"]["mode"] == "stub" and data["aggregate"]["n_corrections"] > 0
