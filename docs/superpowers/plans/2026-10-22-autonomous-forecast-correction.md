# Autonomous Forecast Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the human researcher from forecast correction: an LLM agent corrects the statistical baseline using cited evidence, is plugged into the pipeline, and is scored on a synthetic benchmark.

**Architecture:** Synthetic data (inputs, hidden truth, source documents) makes the whole pipeline and the agent testable with no real data. A tool-calling agent (planner, researcher, verifier gate) reads a long-format forecast table, proposes cited corrections, and a deterministic verifier accepts or rejects each one. The agent sits between the predictors and `FileWriter` behind a config flag, and the LLM sits behind a thin client (NVIDIA endpoint live, scripted stub in tests).

**Tech Stack:** Python 3.11, Poetry, pandas, pytest, `openai` SDK pointed at the NVIDIA OpenAI-compatible endpoint, pydantic, docker compose with a GCS emulator and a Pub/Sub emulator.

**Spec:** `docs/superpowers/specs/2026-10-22-autonomous-forecast-correction-design.md`

## Global Constraints

- Python `~3.11`, Poetry (`pyproject.toml`); line length 149 (black/ruff as configured).
- Work in the current checkout; never create git worktrees.
- No real input CSVs exist: everything runs from synthetic data.
- The `prod` path (GCS upload + Pub/Sub protobuf notification) must keep working and be tested through the emulators.
- The statistical baseline is kept as the agent's starting point. With the correction flag off, outputs must equal the pre-refactor golden outputs.
- The old `AI_adjustments` agent is a code reference only. The refactor makes it importable and nothing else; it is not a benchmark baseline.
- API key is read from the `NVIDIA_API_KEY` environment variable only. It must never appear in the repo, config, docs, fixtures, logs or exception text.
- Hard constraints: values within [0, 1]; `secondary + tertiary = 1` (Education); historical values untouched; a cap on year-over-year change. The old 1% year-over-year rule is a flag to investigate, not a limit on the correction.
- Citation check: the cited number matches a number in the source within a numeric tolerance, with matching year and unit.
- Headline metric: MAE against truth, per country-segment, averaged. "Beats baseline" means lower aggregate MAE.
- Zero hallucinated citations and zero constraint violations: hard gate with the stub model in CI, reported target with a live model.
- Live web search is out of scope; sources come from the synthetic store.
- Plan's own calls (not in the spec): `max_yoy_change` defaults to 0.10; citation tolerance is 0.005 on the 0-1 scale; on agent failure the pipeline falls back to the baseline and logs the error, but a missing `NVIDIA_API_KEY` with correction enabled raises at startup.

## Review Focus

- Country or segment with no source documents: the baseline value is left unchanged (no invented correction). Pinned in Task 10.
- Two sources that contradict each other: the agent only commits a proposal whose citation verifies, and never averages silently; if it cannot decide, the baseline stays. Pinned in Task 10.
- LLM returns malformed JSON, no tool call, an unknown tool name, a 429 or a network error: bounded retries, then the baseline is preserved and the run does not crash. Pinned in Task 8 and Task 10.
- Baseline has NaN or `":"` placeholders (missing and proxy data): the agent skips them and the output gains no new NaN. Pinned in Task 5 and Task 10.
- Key hygiene: a missing key fails fast at startup when correction is enabled, and the key never shows up in logs or exceptions. Pinned in Task 8 and Task 11.

---

### Task 1: Synthetic world generator

**Files:**
- Create: `quotas_forecasting/synthetic/__init__.py`, `quotas_forecasting/synthetic/world.py`, `quotas_forecasting/synthetic/writers.py`
- Create: `tests/__init__.py`, `tests/synthetic/test_world.py`
- Modify: `pyproject.toml` (add `pytest` to the dev group; add `openai` and `pydantic` to main dependencies now, used from Task 8)

**Interfaces:**
- Produces:
  - `@dataclass World` with `countries: list[str]`, `last_year: int`, `pred_year: int`, `truth_education: pd.DataFrame` (index `(country, education)` with `education` in `{"secondary","tertiary"}`, int year columns through `pred_year`), `truth_age_gender: pd.DataFrame` (columns `country`, `Individual type`, then year columns as strings), `shocks: dict[str, Shock]`
  - `@dataclass Shock` with `year: int`, `delta: float`
  - `generate_world(seed: int, countries: list[str] | None = None, last_year: int = 2024, pred_year: int = 2026, shocked_fraction: float = 0.3) -> World`
  - `write_education_inputs(world: World, input_dir: Path) -> list[Path]`, `write_age_gender_inputs(world: World, input_dir: Path) -> list[Path]`

- [ ] **Step 1: Write failing tests in `tests/synthetic/test_world.py`**
  - `test_generate_world_is_deterministic_by_seed`: two calls with seed 1 are equal; seed 1 vs seed 2 differ.
  - `test_truth_stays_in_unit_interval_and_education_sums_to_one`: all values in [0, 1]; `secondary + tertiary == 1` per country-year within 1e-9.
  - `test_shocks_only_affect_years_after_last_year`: truth up to `last_year` is smooth; a shocked country differs from its unshocked curve by `shock.delta` from `shock.year` on.
  - `test_written_inputs_match_loader_file_patterns`: filenames match `<base>_<YYYY-MM-DDTHH_MM_SS+0000>.csv` for each base name in `config["modules"][...]["files"]`, and inputs contain no years after `last_year`.
  - `test_written_inputs_parse_with_missing_data_handler`: `MissingDataHandler.load_and_process_education_data` runs on a `Loader` pointed at the written dir and returns a frame indexed by `(country, education)` with years `2004..last_year`; the age/gender file passes through `standardize_countries` and has `country`, `Individual type` and string year columns `2017..last_year`.
- [ ] **Step 2: Run `pytest tests/synthetic -v`. Expected: FAIL (module missing).**
- [ ] **Step 3: Implement `generate_world` and the writers.** Use logistic saturation curves per country with seeded noise. Include a few missing values, one country that needs a name-variant from `country_mapping` (e.g. "Czechia"), and the age-group strings the processors parse (`"Males, 16 to 24 years old"`-style, see `AgeGenderProcessor.extract_age_groups` and `extract_gender`). Derive exact column names and file layout by reading `MissingDataHandler.load_and_process_education_data` and the regex patterns in `config.yaml`.
- [ ] **Step 4: Run `pytest tests/synthetic -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: synthetic world generator for inputs and hidden truth`

### Task 2: Local data-dir override and golden end-to-end test

**Files:**
- Modify: `quotas_forecasting/config/config.yaml` (add `defaults.local_data_dir: null`), `quotas_forecasting/data_handlers/loader.py` (`_setup_local_input_folder`, around lines 44-89), `quotas_forecasting/data_handlers/file_writer.py` (`_get_local_path`, around lines 148-154)
- Create: `tests/e2e/test_pipeline_local.py`, `tests/conftest.py`, `tests/golden/` (generated files)

**Interfaces:**
- Consumes: `generate_world`, `write_education_inputs`, `write_age_gender_inputs` (Task 1).
- Produces: fixture `local_config(tmp_path) -> dict` (deep copy of `load_config()` with `defaults.local_data_dir` set to `tmp_path`, `pubsub.enable_notifications=False`, `defaults.auto_fetch=False`); fixture `world`; `run_single_module(...)` from `quotas_forecasting/main.py` is the entry the tests call. When `local_data_dir` is set, the Loader reads and the FileWriter writes under `<local_data_dir>/<Namespace>/{input,output}` and `_setup_local_input_folder` does not move files out of the package dir.

- [ ] **Step 1: Write failing tests.**
  - `test_education_pipeline_writes_expected_outputs`: after `run_single_module("education", "local", ..., pred_year=2026, last_year=2024, config=local_config)`, four CSVs exist for 16-64/16-74 predictions and past-data. Assert the files exist: the processors catch and log exceptions, so a silent failure otherwise looks like success.
  - `test_age_gender_pipeline_writes_expected_outputs`: same for the two Age_Gender files.
  - `test_outputs_in_unit_interval`: every prediction value is in [0, 1].
  - `test_outputs_match_golden` (parametrized over the six files): equal to `tests/golden/<name>` with `pandas.testing.assert_frame_equal(rtol=1e-6)`; if the golden is missing and `UPDATE_GOLDEN=1`, write it.
- [ ] **Step 2: Run `pytest tests/e2e -v`. Expected: FAIL.**
- [ ] **Step 3: Implement the override in `Loader` and `FileWriter`.** When `defaults.local_data_dir` is falsy, behavior is unchanged. Fix any generator gaps surfaced by the pipeline (Task 1 files) until all six outputs are produced. Pin seeds: world seed 7, config `random_state` already 42.
- [ ] **Step 4: Run `UPDATE_GOLDEN=1 pytest tests/e2e -v`, inspect the six goldens for sanity (non-empty, plausible values), then run `pytest tests/e2e -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `test: local end-to-end pipeline on synthetic data with golden outputs`

### Task 3: Emulators and prod-path end-to-end test

**Files:**
- Create: `docker-compose.emulators.yml`, `quotas_forecasting/services/gcp_clients.py`, `tests/e2e/test_pipeline_prod_emulators.py`
- Modify: `quotas_forecasting/data_handlers/loader.py` (prod branch, lines 33-42), `quotas_forecasting/data_handlers/file_writer.py` (prod branch, lines 73-97), `quotas_forecasting/services/message_serializer.py` (lines 39-56)

**Interfaces:**
- Produces: `get_credentials(config: dict) -> google.auth.credentials.Credentials` in `gcp_clients.py`: returns `AnonymousCredentials()` when `STORAGE_EMULATOR_HOST` or `PUBSUB_EMULATOR_HOST` is set, otherwise the existing Secret Manager service-account credentials. `Loader`, `FileWriter` and `MessageSerializer` use it instead of calling Secret Manager or `google.auth.default()` directly. Pytest marker `emulators` (registered in `pyproject.toml`), skipped unless `STORAGE_EMULATOR_HOST` and `PUBSUB_EMULATOR_HOST` are set. Fixture `emulator_config` sets `gcs.data_source="Eurostat"`, `pubsub.enable_notifications=True`, buckets `ml-quotas-forecasting` and `universe-bucket-staging`.

- [ ] **Step 1: Write the failing test `test_prod_pipeline_uploads_and_notifies`** (marker `emulators`). Create both buckets and the topic plus a pull subscription, upload the synthetic inputs under `Eurostat/<Namespace>/input/` using colon timestamps as GCS expects, run `run_single_module(..., mode="prod")` for both modules. Assert: internal bucket holds the outputs under `Eurostat/<Namespace>/output/`; external bucket holds `2026_*_with_past_data_both.csv` under `population_modelling_projections/`; three messages pulled, each parses as `ModellingOutputFile` with `sender == "quotas_forecasting"`, `projection_year == 2026`, and `file_type` in `{AGE_AND_GENDER, EDUCATION_16_TO_64, EDUCATION_16_TO_74}`.
- [ ] **Step 2: Run `docker compose -f docker-compose.emulators.yml up -d`, export the emulator env vars printed in the compose file comments, run `pytest -m emulators -v`. Expected: FAIL.**
- [ ] **Step 3: Write the compose file and `gcp_clients.get_credentials`; switch the three call sites.** Emulator images are the builder's choice; the compose file documents the two env vars to export. Production behavior without emulator env vars must be identical.
- [ ] **Step 4: Run `pytest -m emulators -v` and then `pytest -v` (emulators skipped). Expected: PASS.**
- [ ] **Step 5: Commit** `test: prod path against GCS and Pub/Sub emulators`

### Task 4: Make `AI_adjustments` a package and fix docs drift

**Files:**
- Create: `quotas_forecasting/AI_adjustments/__init__.py`, `tests/test_ai_adjustments_import.py`
- Modify: every `quotas_forecasting/AI_adjustments/*.py` (flat imports such as `from client import AIClient` become `from quotas_forecasting.AI_adjustments.client import AIClient`), `pyproject.toml` (no new deps; `openai` and `pydantic` came in Task 1), `quotas_forecasting/AI_adjustments/README.md`, `README.md`, `docs/ARCHITECTURE.md`
- Delete: `quotas_forecasting/AI_adjustments/requirements.txt`

**Interfaces:**
- Produces: `python -m quotas_forecasting.AI_adjustments.main --help` works from the repo root and from any other cwd.

- [ ] **Step 1: Write failing tests.** `test_modules_import_from_any_cwd` imports `agent`, `client`, `tools`, `rules`, `workflow_enforcer` under `quotas_forecasting.AI_adjustments` with `monkeypatch.chdir(tmp_path)`; `test_cli_help_runs` runs `python -m quotas_forecasting.AI_adjustments.main --help` via `subprocess` from `tmp_path` with `PYTHONPATH` set to the repo root and asserts exit code 0.
- [ ] **Step 2: Run `pytest tests/test_ai_adjustments_import.py -v`. Expected: FAIL (`ModuleNotFoundError: client`).**
- [ ] **Step 3: Rewrite imports and drop the separate requirements file.** No behavior changes. In the READMEs replace "run with this folder as cwd" with the `python -m` invocation and correct the references to `cloudbuild.yaml` and `post-merge-run-forecast.yml` (the file is `post-merge-run-full-pipeline.yml`; `cloudbuild.yaml` does not exist).
- [ ] **Step 4: Run `pytest -v`. Expected: PASS, including the Task 2 goldens.**
- [ ] **Step 5: Commit** `refactor: make AI_adjustments an importable package; fix docs drift`

### Task 5: Forecast table, adapters and hard-constraint verifier

**Files:**
- Create: `quotas_forecasting/correction/__init__.py`, `quotas_forecasting/correction/types.py`, `quotas_forecasting/correction/adapters.py`, `quotas_forecasting/correction/constraints.py`, `tests/correction/test_adapters.py`, `tests/correction/test_constraints.py`
- Modify: `quotas_forecasting/config/config.yaml` (add `correction:` block with `enabled: false`, `max_yoy_change: 0.10`, `citation_tolerance: 0.005`, `model: null`, `max_iterations_per_segment: 6`, `max_llm_retries: 3`)

**Interfaces:**
- Produces:
  - `ForecastTable = pd.DataFrame` with columns `country: str`, `segment: str`, `year: int`, `value: float`, `is_forecast: bool` (value NaN allowed). Segment is the education level (`"secondary"`, `"tertiary"`) for Education and the `Individual type` string for Age_Gender.
  - `to_table(df: pd.DataFrame, module: str, last_year: int) -> ForecastTable`; `from_table(table: ForecastTable, template: pd.DataFrame, module: str) -> pd.DataFrame` (round-trips to the original shape, index and dtypes, including `":"` placeholders for NaN in Age_Gender)
  - `@dataclass Violation` with `kind: str` (`"bounds"`, `"sum_to_one"`, `"historical_changed"`, `"yoy_cap"`), `country: str`, `segment: str`, `year: int`, `detail: str`
  - `check_constraints(baseline: ForecastTable, corrected: ForecastTable, module: str, max_yoy_change: float) -> list[Violation]`

- [ ] **Step 1: Write failing tests.** `test_roundtrip_education_and_age_gender` (using Task 1 world + Task 2 predictor outputs, `from_table(to_table(df)) == df`, NaN and `":"` preserved); `test_bounds_violation`; `test_sum_to_one_violation_education_only`; `test_historical_change_detected`; `test_yoy_cap_violation_at_boundary` (exactly `max_yoy_change` passes, `+1e-6` fails); `test_nan_values_never_violate` (NaN baseline stays NaN, no violation; NaN introduced where baseline had a value is a violation of kind `"bounds"`).
- [ ] **Step 2: Run `pytest tests/correction -v`. Expected: FAIL.**
- [ ] **Step 3: Implement the types, adapters and verifier.** Verifier is pure and deterministic.
- [ ] **Step 4: Run `pytest tests/correction -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: forecast table, adapters and hard-constraint verifier`

### Task 6: Synthetic sources, store and benchmark builder

**Files:**
- Create: `quotas_forecasting/synthetic/sources.py`, `quotas_forecasting/synthetic/benchmark.py`, `quotas_forecasting/correction/sources.py`, `tests/synthetic/test_benchmark.py`

**Interfaces:**
- Consumes: `generate_world`, `World`, `to_table`, the real predictors/processors (Tasks 1, 2, 5).
- Produces:
  - `@dataclass SourceDoc` with `doc_id: str`, `publisher: str`, `country: str`, `year: int`, `unit: str` (`"share"` for 0-1 or `"percent"`), `segment: str`, `text: str`, `value: float | None` (ground-truth fields used only by tests and the evaluator, never shown to the agent)
  - `@dataclass Benchmark` with `module: str`, `world: World | None` (None for real-data backtests, Task 13), `baseline: ForecastTable`, `truth: ForecastTable`, `docs: list[SourceDoc]`, `last_year: int`, `pred_year: int`
  - `build_benchmark(seed: int, module: str, n_countries: int = 8, noise: float = 0.01, wrong_fraction: float = 0.2, conflict_fraction: float = 0.15) -> Benchmark` (baseline comes from running the real pipeline on the world's truncated inputs, so it fails naturally on injected shocks)
  - `make_docs(truth: ForecastTable, seed: int, noise: float, wrong_fraction: float, conflict_fraction: float) -> list[SourceDoc]` (the doc generator, reusable on any truth table)
  - `class SourceStore(Protocol)`: `search(query: str, country: str | None = None, year: int | None = None) -> list[SourceSnippet]` and `read(doc_id: str) -> str`, where `SourceSnippet` has `doc_id`, `publisher`, `country`, `year`, `title`
  - `class SyntheticSourceStore` implementing it over `Benchmark.docs`

- [ ] **Step 1: Write failing tests.** `test_benchmark_deterministic_by_seed`; `test_baseline_error_is_positive_for_shocked_countries` (baseline MAE on shocked country-years is greater than on unshocked ones); `test_some_docs_are_wrong_and_some_conflict` (`wrong_fraction` and `conflict_fraction` respected within one doc on 8 countries); `test_some_countries_have_no_docs`; `test_store_search_filters_by_country_and_year`; `test_store_read_returns_text_containing_the_value_with_unit`.
- [ ] **Step 2: Run `pytest tests/synthetic/test_benchmark.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** Docs are short prose paragraphs mixing the target number with distractor numbers (other years, other segments), in `percent` or `share` units. Dev seeds `0-9`, held-out seeds `100-109` (constants `DEV_SEEDS`, `HELDOUT_SEEDS` in `benchmark.py`); generator patterns for held-out seeds vary the publisher names and phrasing templates so the agent cannot memorize dev patterns.
- [ ] **Step 4: Run `pytest tests/synthetic -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: synthetic source documents, store and benchmark builder`

### Task 7: Citation verifier and evaluation harness

**Files:**
- Create: `quotas_forecasting/correction/citations.py`, `quotas_forecasting/correction/evaluation.py`, `tests/correction/test_citations.py`, `tests/correction/test_evaluation.py`

**Interfaces:**
- Consumes: `SourceStore`, `ForecastTable`, `check_constraints` (Tasks 5, 6).
- Produces:
  - `@dataclass Citation` with `doc_id: str`, `quoted_text: str`, `value: float`, `year: int`, `unit: str`
  - `verify_citation(citation: Citation, store: SourceStore, tolerance: float) -> bool`: true only if the document exists, `quoted_text` occurs in it, a number in the quote equals `value` within `tolerance` after converting `percent` to `share`, and the quote's year equals `citation.year`
  - `@dataclass EvalReport` with `mae_before: float`, `mae_after: float`, `hallucinated_citations: int`, `violations: int`, `n_corrections: int`
  - `evaluate(benchmark: Benchmark, corrected: ForecastTable, corrections: list["Correction"], store: SourceStore, config: dict) -> EvalReport` (MAE over forecast-year rows, per country-segment then averaged; NaN rows excluded)

- [ ] **Step 1: Write failing tests.** `test_verify_accepts_exact_quote`; `test_verify_accepts_percent_unit_conversion`; `test_verify_rejects_value_outside_tolerance`; `test_verify_rejects_wrong_year`; `test_verify_rejects_quote_not_in_doc`; `test_verify_rejects_unknown_doc_id`; `test_evaluate_perfect_correction_gives_zero_mae_after` (set the corrected table to truth); `test_evaluate_counts_hallucinated_citation_and_violation`; `test_evaluate_no_corrections_equals_baseline_mae`.
- [ ] **Step 2: Run `pytest tests/correction/test_citations.py tests/correction/test_evaluation.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** `Correction` (defined in `types.py`, add it here): `country: str`, `segment: str`, `year: int`, `old_value: float`, `new_value: float`, `citation: Citation`, `confidence: float`, `reason: str`.
- [ ] **Step 4: Run `pytest tests/correction -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: citation verifier and evaluation harness`

### Task 8: LLM client (NVIDIA endpoint and scripted stub)

**Files:**
- Create: `quotas_forecasting/correction/llm.py`, `tests/correction/test_llm.py`

**Interfaces:**
- Produces:
  - `@dataclass ToolCall` with `id: str`, `name: str`, `arguments: dict`, `parse_error: str | None = None`; `@dataclass LLMResponse` with `text: str | None`, `tool_calls: list[ToolCall]`
  - `class LLMClient(Protocol)`: `chat(messages: list[dict], tools: list[dict]) -> LLMResponse`
  - `class NvidiaClient(LLMClient)` with `__init__(self, model: str, base_url: str = "https://integrate.api.nvidia.com/v1", max_retries: int = 3, sdk_client: Any | None = None)`; reads `NVIDIA_API_KEY` from the environment and raises `MissingApiKeyError` (a `RuntimeError`) if absent
  - `class ScriptedClient(LLMClient)` with `__init__(self, responses: list[LLMResponse])`; raises `AssertionError("script exhausted")` when called past the end; records the messages it received in `.calls`
  - `get_llm_client(config: dict) -> LLMClient`: `NvidiaClient` built from `config["correction"]["model"]`

- [ ] **Step 1: Write failing tests.** `test_missing_key_raises_clear_error_without_leaking_anything`; `test_key_never_in_repr_or_exception_text` (set `NVIDIA_API_KEY=nvapi-test-secret`, force an SDK error, assert the secret is absent from `repr(client)` and `str(exception)`); `test_tool_call_json_is_parsed_into_arguments`; `test_malformed_tool_arguments_become_empty_dict_with_parse_error_flag` (add `parse_error: str | None` to `ToolCall`); `test_retries_on_429_then_succeeds` and `test_gives_up_after_max_retries_and_raises_llm_unavailable` (inject a fake `sdk_client`, no sleeping: backoff function is injectable); `test_scripted_client_replays_in_order`.
- [ ] **Step 2: Run `pytest tests/correction/test_llm.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement with the `openai` SDK (`OpenAI(base_url=..., api_key=...)`, chat completions with tools).** Wrap SDK exceptions in `LLMUnavailableError` with the key redacted.
- [ ] **Step 4: Run `pytest tests/correction/test_llm.py -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: LLM client for the NVIDIA endpoint with scripted stub`

### Task 9: Agent tools and verifier gate

**Files:**
- Create: `quotas_forecasting/correction/tools.py`, `tests/correction/test_tools.py`

**Interfaces:**
- Consumes: `SourceStore`, `ForecastTable`, `Correction`, `Citation`, `verify_citation`, `check_constraints` (Tasks 5-7).
- Produces:
  - `class CorrectionSession` with `__init__(self, baseline: ForecastTable, module: str, store: SourceStore, config: dict)`; attributes `working: ForecastTable`, `corrections: list[Correction]`, `rejections: list[Rejection]`
  - `CorrectionSession.search_sources(query: str, country: str | None = None, year: int | None = None) -> list[dict]`
  - `CorrectionSession.read_source(doc_id: str) -> str`
  - `CorrectionSession.propose_correction(country: str, segment: str, year: int, new_value: float, citation: dict, confidence: float, reason: str) -> dict` returning `{"accepted": bool, "reason": str}`; accepts only if the cell is a non-NaN forecast cell, the citation verifies, and `check_constraints` on the tentative table returns no violations; Education proposals for `tertiary` are rejected (tertiary is derived as `1 - secondary`, applied automatically on acceptance)
  - `tool_schemas() -> list[dict]` (OpenAI-format tool definitions for the three tools)
  - `@dataclass Rejection` with `proposal: dict`, `reason: str`

- [ ] **Step 1: Write failing tests.** `test_accepts_valid_cited_correction_and_derives_tertiary`; `test_rejects_hallucinated_citation_and_leaves_table_unchanged`; `test_rejects_out_of_bounds_value`; `test_rejects_historical_cell`; `test_rejects_nan_cell`; `test_rejects_yoy_above_cap`; `test_rejects_direct_tertiary_proposal`; `test_search_returns_only_snippets_never_ground_truth_fields`.
- [ ] **Step 2: Run `pytest tests/correction/test_tools.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement.** Tool results are JSON-serializable dicts.
- [ ] **Step 4: Run `pytest tests/correction/test_tools.py -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: agent tools with deterministic verifier gate`

### Task 10: Agent loop (planner, researcher, retries, budget)

**Files:**
- Create: `quotas_forecasting/correction/agent.py`, `tests/correction/test_agent.py`

**Interfaces:**
- Consumes: `LLMClient`, `ScriptedClient`, `CorrectionSession`, `tool_schemas` (Tasks 8, 9).
- Produces:
  - `@dataclass CorrectionResult` with `corrected: ForecastTable`, `corrections: list[Correction]`, `rejections: list[Rejection]`, `log: list[dict]`, `error: str | None`
  - `run_correction(baseline: ForecastTable, module: str, store: SourceStore, llm: LLMClient, config: dict) -> CorrectionResult`. Planner call lists suspicious country-segments (large forecast deviation from recent trend) from a compact summary of the table; for each suspect, a researcher loop of at most `correction.max_iterations_per_segment` LLM turns with the three tools; rejected proposals are fed back to the model as tool output so it can retry. Never raises on LLM failure: returns the best table so far with `error` set.

- [ ] **Step 1: Write failing tests with `ScriptedClient`.**
  - `test_corrects_one_shocked_country_end_to_end` (use `build_benchmark(seed=0, "education")`, scripted plan, search, read, propose with the true cited value; `evaluate(...)` shows `mae_after < mae_before`, zero hallucinations, zero violations).
  - `test_country_without_sources_is_left_unchanged` (script proposes nothing after an empty search; table equals baseline for that country).
  - `test_contradicting_sources_only_verified_citation_is_committed` (two docs disagree; the script first proposes with a fabricated quote, gets rejected, then retries with a real quote; only the second is committed).
  - `test_rejected_proposal_is_fed_back_to_model` (the next `ScriptedClient` call's messages include the rejection reason).
  - `test_iteration_budget_stops_the_loop` (script that never stops proposing invalid values terminates after `max_iterations_per_segment` turns).
  - `test_unknown_tool_and_malformed_arguments_do_not_crash` (including `ToolCall.parse_error`).
  - `test_llm_unavailable_returns_baseline_with_error` (client raising `LLMUnavailableError` on the second call leaves earlier accepted corrections and sets `error`).
  - `test_nan_cells_stay_nan` (baseline with `":"`/NaN placeholders yields a corrected table with the same NaN positions).
  - `test_age_gender_module_runs` (same flow with `module="age_gender"`).
- [ ] **Step 2: Run `pytest tests/correction/test_agent.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement `run_correction` and the planner/researcher system prompts** (prompts live in `quotas_forecasting/correction/prompts.py`; state the hard constraints, that citations are mandatory, that leaving a value unchanged is a valid outcome, and that tertiary is derived).
- [ ] **Step 4: Run `pytest tests/correction -v`. Expected: PASS.**
- [ ] **Step 5: Commit** `feat: autonomous correction agent loop`

### Task 11: Pipeline integration behind a config flag

**Files:**
- Modify: `quotas_forecasting/forecasting/processors.py` (Education: between `predictor.predict(...)` and `_save_education_predictions`, around lines 112-122; Age_Gender: after `predictor.run_with_data(...)` and before the rename/save, around lines 291-327), `quotas_forecasting/main.py` (`run_single_module`, lines 32-78), `quotas_forecasting/config/config.yaml` (`correction.on_failure: "fallback_to_baseline"`)
- Create: `quotas_forecasting/correction/pipeline.py`, `tests/e2e/test_pipeline_with_correction.py`

**Interfaces:**
- Consumes: `get_llm_client`, `run_correction`, `to_table`, `from_table` (Tasks 5, 8, 10).
- Produces: `maybe_correct(df: pd.DataFrame, module: str, last_year: int, config: dict, store: SourceStore, llm: LLMClient | None = None) -> pd.DataFrame` in `pipeline.py`: returns `df` unchanged when `correction.enabled` is false; otherwise converts, runs the agent, converts back. Raises `MissingApiKeyError` at startup when enabled and no `llm` is injected and the key is missing; on agent `error` it logs and returns the best table (or baseline if nothing was accepted). `Processor.run_*_pipeline` gain an optional `correction_store: SourceStore | None = None` and `correction_llm: LLMClient | None = None` (keyword args, default `None`), threaded through `run_single_module`.

- [ ] **Step 1: Write failing tests.**
  - `test_flag_off_matches_golden` (re-run the Task 2 golden comparison with `correction.enabled=False`).
  - `test_flag_on_with_scripted_llm_changes_only_cited_cells` (Education and Age_Gender: diff of output vs baseline CSVs equals exactly the accepted corrections; `secondary + tertiary == 1`).
  - `test_missing_key_with_flag_on_raises_at_startup_and_message_has_no_secret`.
  - `test_agent_failure_falls_back_to_baseline_and_writes_outputs` (`LLMUnavailableError` from the first call: outputs equal the golden).
  - `test_corrected_prod_run_notifies_with_emulators` (marker `emulators`, reuse Task 3 fixtures: corrected files uploaded and the three notifications still sent).
- [ ] **Step 2: Run `pytest tests/e2e/test_pipeline_with_correction.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement `maybe_correct` and the hooks.** With the flag off the processors must call nothing new.
- [ ] **Step 4: Run `pytest -v` (and `pytest -m emulators -v` with emulators up). Expected: PASS.**
- [ ] **Step 5: Commit** `feat: plug correction agent into the pipeline behind config flag`

### Task 12: Benchmark runner and live run

**Files:**
- Create: `quotas_forecasting/correction/benchmark_cli.py`, `tests/correction/test_benchmark_cli.py`, `docs/CORRECTION_AGENT.md`
- Modify: `README.md` (link to the new doc), `.gitignore` (`benchmark_results/`)

**Interfaces:**
- Consumes: `build_benchmark`, `SyntheticSourceStore`, `run_correction`, `evaluate`, `get_llm_client`, `DEV_SEEDS`, `HELDOUT_SEEDS`.
- Produces: `python -m quotas_forecasting.correction.benchmark_cli --module {education,age_gender} --seeds 0-4 [--live] [--heldout] [--n-countries N]` writing `benchmark_results/<timestamp>.json` with per-seed `EvalReport` fields and the aggregate; exit code non-zero when, in stub mode, hallucinations or violations are non-zero. Stub mode uses an oracle `ScriptedClient` built from the benchmark docs (proposes the true cited value for each shocked country that has a verifiable doc).

- [ ] **Step 1: Write failing tests.** `test_stub_mode_gate_passes_with_oracle_script` (aggregate `mae_after < mae_before`, zero hallucinations, zero violations, exit code 0); `test_stub_mode_gate_fails_when_script_hallucinates` (exit code non-zero); `test_live_flag_without_key_exits_with_clear_message`.
- [ ] **Step 2: Run `pytest tests/correction/test_benchmark_cli.py -v`. Expected: FAIL.**
- [ ] **Step 3: Implement the CLI and write `docs/CORRECTION_AGENT.md`** (how to run stub and live, config keys, what the metrics mean, how to add sources in milestone 2).
- [ ] **Step 4: Run `pytest -v`. Expected: PASS.**
- [ ] **Step 5: Live run (acceptance).** With the user's `NVIDIA_API_KEY` exported in their shell, run `python -m quotas_forecasting.correction.benchmark_cli --module education --seeds 0-2 --n-countries 5 --live`, then the same for `age_gender`. Choose a strong tool-calling model from build.nvidia.com and set `correction.model` in `config.yaml` (smoke-test it with one call first). Record the aggregate `mae_before`, `mae_after`, hallucination and violation counts in `docs/CORRECTION_AGENT.md`; report honestly if the live model misses the target.
- [ ] **Step 6: Commit** `feat: benchmark runner, docs and first live run results`

### Task 13: Real-data backtest from Eurostat (runs after Task 12)

**Files:**
- Create: `quotas_forecasting/backtest/__init__.py`, `quotas_forecasting/backtest/eurostat_backtest.py`, `tests/e2e/test_real_backtest.py`
- Modify: `quotas_forecasting/correction/benchmark_cli.py` (add `--real` flag), `pyproject.toml` (register marker `realdata`), `.gitignore` (add `.cache/`), `docs/CORRECTION_AGENT.md`

**Interfaces:**
- Consumes: `EurostatDataFetcher(last_year_of_observations: int, config: dict).fetch_data() -> bool`, `run_single_module`, the Task 3 emulator fixtures, `to_table`, `make_docs`, `Benchmark`, `SyntheticSourceStore`, `evaluate` (Tasks 2, 3, 5, 6, 7).
- Produces:
  - `fetch_eurostat_inputs(cache_dir: Path, last_year: int, config: dict) -> Path`: runs the real fetcher against the GCS emulator, downloads the resulting timestamped CSVs into `<cache_dir>/<Namespace>/input/` and returns `cache_dir`; reuses an existing complete cache without touching the network. Raises `MissingInputFilesError` listing any expected input file (see `config["modules"][...]["files"]`) the fetcher could not produce.
  - `build_real_backtest(cache_dir: Path, module: str, cut_year: int = 2022, pred_year: int = 2024, seed: int = 0) -> Benchmark`: `truth` is the actual published Eurostat values for years after `cut_year`; `baseline` is the pipeline run with `last_year=cut_year` on inputs truncated at `cut_year`; `docs` come from `make_docs(truth, ...)`, so the numbers are real and only the document wrapping and corruption are synthetic; `world` is `None`. Years where Eurostat has no value are NaN and excluded from MAE.

- [ ] **Step 1: Write failing tests (marker `realdata`, skipped unless `QF_REALDATA=1` and the emulator env vars are set).** `test_fetch_populates_cache_with_all_expected_input_files`; `test_second_fetch_uses_cache_without_network` (monkeypatch `requests.get` to raise); `test_backtest_baseline_error_is_finite_and_sane` (`0 < mae_before < 0.2`, truth has no values at or before `cut_year`); `test_backtest_runs_through_evaluate_with_stub_agent` (`ScriptedClient` proposing nothing gives `mae_after == mae_before`); `test_missing_expected_file_raises_with_its_name` (delete one cached input, expect `MissingInputFilesError` naming it).
- [ ] **Step 2: Run `QF_REALDATA=1 pytest -m realdata -v` with emulators up. Expected: FAIL.**
- [ ] **Step 3: Implement the module and the `--real` flag.** If the fetcher cannot reproduce an input file the pipeline needs (a `config.yaml` comment mentions files "provided by GR"), do not fabricate it: finish what works, list the missing files in your report and in `docs/CORRECTION_AGENT.md`, and report DONE_WITH_CONCERNS.
- [ ] **Step 4: Run `pytest -v` (realdata skipped) and `QF_REALDATA=1 pytest -m realdata -v`. Expected: PASS.**
- [ ] **Step 5: Live real-data run (controller, same key rule as Task 12).** `python -m quotas_forecasting.correction.benchmark_cli --module education --real --live`; record the aggregate result in `docs/CORRECTION_AGENT.md`.
- [ ] **Step 6: Commit** `feat: real-data Eurostat backtest`

### Task 14: Maths review and alternative forecasting methods (runs after Task 13)

**Files:**
- Create: `quotas_forecasting/forecasting/experimental/__init__.py`, `quotas_forecasting/forecasting/experimental/gam.py`, `quotas_forecasting/forecasting/experimental/balancing.py`, `quotas_forecasting/backtest/compare.py`, `tests/experimental/test_gam.py`, `tests/experimental/test_balancing.py`, `tests/experimental/test_compare.py`, `docs/MODEL_REVIEW.md`
- Modify: `pyproject.toml` (add `pygam`), `quotas_forecasting/config/config.yaml` (add `prediction.method: "ensemble"`), `quotas_forecasting/forecasting/predictors.py` (read `prediction.method`; `"ensemble"` is the current behavior)

**Interfaces:**
- Consumes: the synthetic benchmark (Task 6), the real-data backtest and cache (Task 13), the existing predictors (`predict_random_walk`, `fit_predict_theta`, logistic and LASSO paths in `predictors.py`).
- Produces:
  - `gam_forecast(series: pd.Series, horizon: int, monotone: bool = True) -> np.ndarray`: penalized-spline GAM on the logit scale (`pygam`), output clipped to [0, 1], non-decreasing when `monotone`
  - `class InfeasibleBalanceError(ValueError)`
  - `entropy_balance_weights(donors: np.ndarray, target: np.ndarray) -> np.ndarray`: non-negative weights over donor rows that sum to 1, minimize KL divergence to uniform, and reproduce the target's feature means; solved through the dual with `scipy.optimize`; raises `InfeasibleBalanceError` when the target is outside the donors' convex hull
  - `kernel_balance_weights(donors: np.ndarray, target: np.ndarray, length_scale: float | None = None) -> np.ndarray`: simplex weights minimizing the RBF-kernel mean discrepancy to the target (median-heuristic length scale when `None`)
  - `proxy_forecast(target_history: pd.Series, donors: pd.DataFrame, method: str) -> pd.Series` with `method` in `{"entropy", "kernel"}`: weighted donor trajectories shifted to the target's last observed level
  - `compare_methods(tables: list[Benchmark], methods: list[str], origins: list[int]) -> pd.DataFrame`: rolling-origin MAE per `(module, method)`, with `methods` drawn from `{"ensemble", "gam", "entropy_proxy", "kernel_proxy"}`

- [ ] **Step 1: Write the maths review as failing tests first, then the document.** For each suspect below write a test or backtest experiment that confirms or refutes it, then record the verdict and the evidence in `docs/MODEL_REVIEW.md` (verdict, evidence, decision). Suspects already noticed while reading the code: (a) `predictors.py:90-91` applies the same backward interpolation twice, with a `# ?` comment; (b) `config.yaml` blend weights are fixed (Education Theta 0.7 / RW 0.3; Age_Gender 1/3 each) with no recorded justification; (c) the LASSO proxy path is inactive because `target_markets: {}`; (d) clipping to [0, 1] versus the logistic ceiling `K_max` 0.995; (e) `replace(0, np.nan)` on Education ratios; (f) LASSO CV with very few proxy rows. Add any further issue found, each backed by evidence, not assertion.
- [ ] **Step 2: Write failing tests.** `test_gam_stays_in_unit_interval_and_is_monotone`; `test_gam_recovers_logistic_truth_within_tolerance` (synthetic logistic series, 2-year horizon, error below the ensemble's on the same series or a fixed 0.02 bound, whichever the review shows is meaningful); `test_entropy_weights_are_nonnegative_sum_to_one_and_match_target_means` (1e-6); `test_entropy_raises_when_target_outside_hull`; `test_kernel_weights_lie_on_simplex`; `test_compare_methods_is_deterministic_and_has_one_row_per_module_method`; `test_default_method_leaves_goldens_unchanged` (the Task 2 golden tests pass untouched with `prediction.method: "ensemble"`).
- [ ] **Step 3: Run `pytest tests/experimental -v`. Expected: FAIL.**
- [ ] **Step 4: Implement the experimental module and `compare_methods`; wire `prediction.method` in `predictors.py` with `"ensemble"` as the default and no change to its numerical results.**
- [ ] **Step 5: Run the comparison on the synthetic dev seeds and on the real-data backtest (rolling origins 2019-2022), then write the results table and the decision into `docs/MODEL_REVIEW.md`.** Decision rule: a method becomes an available option only if it beats the current ensemble's MAE on both the synthetic and the real backtests across origins by a stated margin; otherwise it is documented and left unused. No default is changed in this task. Note in the document that kernel and entropy balancing are survey-weighting tools and are used here only as donor-weighting for proxy forecasts; weighting an actual respondent sample to the quotas is out of this repo's scope.
- [ ] **Step 6: Run `pytest -v`. Expected: PASS, goldens unchanged.**
- [ ] **Step 7: Commit** `feat: maths review and experimental GAM and balancing forecasters`

