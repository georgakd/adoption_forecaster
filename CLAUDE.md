# Adoption Forecaster

Forecasts internet-penetration shares by segment (Education, Age_Gender) from Eurostat data. An optional LLM agent corrects the statistical baseline using cited evidence; a deterministic gate accepts or rejects every proposal. Python 3.11, Poetry.

## Layout
- `adoption_forecaster/main.py`: pipeline entry point (load, forecast, write, notify).
- `adoption_forecaster/forecasting/`, `data_handlers/`, `services/`: the statistical pipeline and GCS/Pub/Sub clients.
- `adoption_forecaster/correction/`: autonomous correction agent (gate, evaluator, caps, benchmark CLI). Off by default (`correction.enabled`).
- `adoption_forecaster/AI_adjustments/`: older standalone LLM CLI with its own README, `docs/` and `tests/`. Not used by the pipeline or by `correction/`.
- `adoption_forecaster/synthetic/`, `backtest/`: seeded synthetic world and the real-data Eurostat backtest.
- `docs/`: architecture, design patterns, `MODEL_REVIEW.md` (maths review, known unfixed production bugs), `CORRECTION_AGENT.md` (agent, benchmark, live results, "Before enabling" list).

## Commands
- Tests (offline, ~100 s): `pytest`. `pytest tests` leaves out the `AI_adjustments` tests.
- Emulator tests: start `docker compose -f docker-compose.emulators.yml up -d`, export `STORAGE_EMULATOR_HOST=http://localhost:4553` and `PUBSUB_EMULATOR_HOST=localhost:8685`, run `pytest -m emulators`. Real-data backtest: `QF_REALDATA=1 pytest -m realdata`.
- Benchmark (stub, no network): `python -m adoption_forecaster.correction.benchmark_cli --module education --seeds 0-4`. Add `--real` for the Eurostat cache and `--live --model <id>` for a real LLM.
- Lint and format: black and ruff, line length 149 (configured in `pyproject.toml`; Ruff checks `E4,E7,E9,F` only). Hooks live in `.pre-commit-config.yaml` (`poetry run pre-commit install`); they run on staged files only, and about 20 existing files are not yet Black-formatted.
- Refresh the lock: `poetry lock` (Python must be 3.11; with uv: `VIRTUAL_ENV=$PWD/.venv uvx --python 3.11 poetry lock`).

## Rules
- Never write an API key or secret into code, config, docs, logs or results files. Live runs read `NVIDIA_API_KEY` from the shell environment; `OPENAI_API_KEY` (AI_adjustments) likewise, and it takes precedence over Secret Manager.
- Flag-off pipeline output must stay byte-identical to the golden CSVs in `tests/golden/`. Do not regenerate goldens to make a test pass.
- The correction gate and the evaluator must use the same definition of a violation (`introduced_violations` with the per-segment caps from `compute_yoy_caps`). Hallucination counts and violation counts of the gate are zero by construction; only MAE gain and `n_corrections` measure the model.
- Hard-to-reverse or outward-facing actions (push, PR, publishing, deleting resources) need the user's go-ahead.

## Things that look wrong but are intended
- The year-over-year cap is adaptive per segment by default (`correction.yoy_cap.mode`); `fixed` keeps 0.10. Adaptive makes the synthetic Education stub gain small (-48.1% to -5.4%): the maintainer decided Education stays adaptive.
- The real CLI with `correction.enabled` stops with `NoSourceStoreError` until a real source store exists (milestone 2). Tests inject a synthetic store.
- `prediction.method` accepts only `ensemble`; the experimental GAM and balancing methods live in the optional `experiments` Poetry group (`poetry install --with experiments`) and are not wired into the pipeline.
- `config.yaml` still holds the previous owner's numeric project numbers and external bucket/topic names; cloud resource names use `adoption-forecaster-*` and `YOUR_GCP_PROJECT_ID` (see README "Cloud resources").

## Not covered by tests
Secret Manager, IAM, the Cloud Run job and the image push need a real GCP project; no emulator covers them. The gate does not bind a citation to its segment (the evaluator checks it independently); see `docs/CORRECTION_AGENT.md` before enabling the flag.
