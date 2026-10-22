# Autonomous correction agent

## What it is

An LLM agent that corrects the statistical baseline forecast using cited evidence from source documents. It is **one model** playing two roles:

1. **Planner**: one call that reads a compact summary of the forecast table (per country-segment: recent history, forecast, deviation from trend, number of source documents per country) and returns a JSON list of suspicious country-segments.
2. **Researcher**: for each suspect, a bounded tool-calling loop (`search_sources`, `read_source`, `propose_correction`). A proposal carries the new value, a verbatim quote from a document, the value and year the quote states, a confidence and a reason.

Every proposal passes a **deterministic gate** before it touches the table: the cited quote must really occur in the document, the quoted value must match the proposed value within `citation_tolerance`, and the corrected table must satisfy the hard constraints (bounds, shares summing to one, historical values untouched, year-over-year change within `max_yoy_change`, no worsening of an existing violation). Rejections are fed back to the model as the tool result. The model can never write a value the gate has not accepted.

The agent never raises on LLM failure or malformed output: it returns the best table so far with an `error` string.

## Data flow

```
predictor frame --to_table--> baseline ForecastTable --run_correction(store, llm)--> corrected table + corrections + rejections + log
                                                                                   --from_table--> same-shape frame
```

Code: `quotas_forecasting/correction/` (`agent.py` loop, `tools.py` gate and tools, `citations.py`, `constraints.py`, `llm.py`, `sources.py`, `pipeline.py` hook, `evaluation.py`, `oracle.py` and `benchmark_cli.py` for the benchmark).

## Configuration (`correction:` in `quotas_forecasting/config/config.yaml`)

| Key | Meaning |
|---|---|
| `enabled` | Master switch for the pipeline hook. Off by default; with it off nothing is built or called. |
| `max_yoy_change` | Hard cap on the year-over-year change of a corrected value (share scale, e.g. `0.10`). |
| `citation_tolerance` | Maximum absolute difference between the proposed value and the value stated in the quote. |
| `model` | NVIDIA model id (OpenAI-compatible endpoint at build.nvidia.com). `null` until chosen. |
| `max_iterations_per_segment` | Tool-calling turns the researcher may use per country-segment. |
| `max_llm_retries` | Retries for transient LLM errors and for unparseable planner replies. |
| `max_suspects` | Planner suspects beyond this many are logged as ignored, not researched. |
| `on_failure` | Informational: on any agent failure the baseline (plus corrections accepted so far) is kept. |

The API key is read from the `NVIDIA_API_KEY` environment variable only. It is never stored in the repo, the config, logs, exception text or result files.

## Running the benchmark

The benchmark builds a seeded synthetic world, runs the real forecasting pipeline on the data visible up to the last year to get the baseline, and generates source documents (some wrong, some conflicting) about the truth, which contains shocks the baseline misses.

```bash
# Stub mode (default): an oracle script stands in for a perfect model. No network, no key. This is the CI gate.
python -m quotas_forecasting.correction.benchmark_cli --module education --seeds 0-4
python -m quotas_forecasting.correction.benchmark_cli --module age_gender --seeds 0-2 --n-countries 5

# Held-out seeds (100+): defaults to HELDOUT_SEEDS when --seeds is not given
python -m quotas_forecasting.correction.benchmark_cli --module education --heldout

# Live mode: a real model. Export the key in your own shell, never in a file.
export NVIDIA_API_KEY=...   # your key
python -m quotas_forecasting.correction.benchmark_cli --module education --seeds 0-2 --n-countries 5 --live --model <model-id>
```

Options: `--seeds` takes `a-b` ranges and comma lists (`0-2,7`), default `0-2`; `--n-countries N` (default 8); `--model` overrides `correction.model` in memory only; `--out DIR` (default `benchmark_results`, git-ignored). Live runs print the number of LLM calls and elapsed seconds so cost stays visible.

Exit codes: stub mode returns 0 only if the gate passes, 1 otherwise. Live mode returns 0 whenever the run completes (the gate result is reported, not enforced) and 2 if `NVIDIA_API_KEY` or the model is missing.

## Reading the results

Each run writes `benchmark_results/<UTC timestamp>.json` with `config` (mode, module, model, n_countries, seeds), `per_seed` (the fields below plus `seed` and `error`), `aggregate`, `gate_passed`, `llm_calls` (live only) and `elapsed_seconds`.

- `mae_before` / `mae_after`: mean absolute error against the truth over forecast years, computed per country-segment and then averaged over country-segments. The headline metric. The aggregate is the mean over seeds; `relative_mae_change = (after - before) / before` (negative is better).
- `hallucinated_citations`: accepted corrections whose citation does not verify against the source store (quote absent from the document, or value mismatch). Summed over seeds.
- `violations`: hard-constraint violations the corrections introduced or worsened relative to the baseline (the same rule `propose_correction` enforces). Summed over seeds.
- `baseline_violations`: violations already present in the baseline forecast, reported separately and not counted against the agent (console column `base_viol`). Summed over seeds.
- `n_corrections`: accepted corrections. `error`: the agent's failure message for that seed, if any.

## Hard gate versus reported target

- **Stub model (CI):** hard gate. Aggregate `mae_after < mae_before`, zero hallucinated citations and zero introduced or worsened violations; the CLI exits non-zero otherwise. The stub oracle proposes the cited value of a document that agrees with the truth, so a failure means the plumbing or the gate is broken.
- **Live model:** reported target. The same numbers are reported, but a miss does not fail the run: it measures the model, not the code.

## Using it in the real pipeline (milestone 2)

The real CLI cannot enable correction yet: `get_source_store` in `pipeline.py` raises `NoSourceStoreError` because no real source store exists. Until then `correction.enabled` only works when a store is injected (as the tests and benchmark do).

A milestone-2 store implements the `SourceStore` Protocol in `sources.py`:

- `search(query, country=None, year=None) -> list[SourceSnippet]`
- `search_with_total(query, country=None, year=None) -> (snippets, total_matches_before_cap)`. The planner calls `search_with_total("", country=country)` and uses the total as the number of documents for that country, so an empty query must return all documents of the country.
- `read(doc_id) -> str`: the document text, which citations are verified against.

Snippets expose only `doc_id, publisher, country, year, title`. Then implement `get_source_store(config)` to return it.

## Real-data backtest (Eurostat)

`--real` replaces the synthetic world with real Eurostat data: the pipeline gets the inputs cut at `--cut-year` (default 2022) and forecasts `--pred-year` (default 2024; `pred_year - cut_year` must equal `validation.year_difference`, 2). It is scored against the values Eurostat actually published for the years after the cut.

```bash
# stub (oracle) mode, offline once the cache exists
python -m quotas_forecasting.correction.benchmark_cli --module education --real --seeds 0-1
# live mode (same key rule as above)
python -m quotas_forecasting.correction.benchmark_cli --module education --real --live
```

The JSON `config` records `"source": "real-backtest"`, `cut_year` and `pred_year` (`n_countries` is `null`: all fetched countries are used).

**What is real and what is synthetic.** Baseline and truth numbers are real. The baseline is the real pipeline run with `last_year=cut_year` on the inputs truncated at `cut_year`. The truth comes from a second run on the inputs up to `pred_year`, using its observed (non-forecast) cells for the years after the cut. For age_gender this is the published share of internet users per country, gender and age group. For education it is the pipeline's tertiary/secondary share of internet users, which the pipeline derives from three published series (internet use by education level, population by education level, population totals). It is not a single published number. Cells where Eurostat has no value (on the 2026-10-22 cache: Switzerland 2024, in both modules) are NaN and are left out of the MAE. The source documents are still **synthetic prose wrappers around these real numbers** (`make_docs`, same noise, wrong and conflict fractions as the synthetic benchmark). Seeds only change the documents, so `mae_before` is the same for every seed. `Benchmark.world` is `None`.

**What the fetcher could reproduce.** `fetch_eurostat_inputs` runs the repo's own `EurostatDataFetcher` against the GCS emulator, then copies the uploaded files to `.cache/eurostat/<Namespace>/input/`. The emulator is required: without `STORAGE_EMULATOR_HOST` it refuses to run, so no real GCP call can happen. All four input files listed under `modules.*.files` were reproduced, so both modules run:

| Module | File | Eurostat dataset | Reproduced |
|---|---|---|---|
| education | `total_population_by_age_range` | DEMO_PJAN | yes |
| education | `population_ratios_by_education` | EDAT_LFSE_03 | yes |
| education | `internet_penetration_ratios_by_education` | ISOC_CI_IFP_IU | yes |
| age_gender | `internet_penetration_by_age_group_and_gender` | ISOC_CI_IFP_IU | yes |

The fetcher's log line says "EDAT_LFS_9903", but that label is stale: the code fetches `eurostat.dataset_codes[1]`, which is `EDAT_LFSE_03`. If a future fetch cannot produce an expected file, `MissingInputFilesError` lists the missing `<Namespace>/<base name>` entries. No file is fabricated.

**Cache.** The current cache was fetched on **2026-10-22**, with `last_year=2024` (`.cache/eurostat/manifest.json`). Eurostat update stamps (the file-name timestamps): ISOC_CI_IFP_IU 2026-04-17, EDAT_LFSE_03 2026-09-10, DEMO_PJAN 2026-09-25. A complete cache is reused with no network and no emulator, which keeps runs deterministic. `.cache/` is git-ignored.

To refresh the cache (about 7 minutes, mostly EDAT_LFSE_03), delete `.cache/eurostat/`, start the emulators, and rerun the realdata tests (or any `--real` run):

```bash
docker compose -f docker-compose.emulators.yml up -d
export STORAGE_EMULATOR_HOST=http://localhost:4553 PUBSUB_EMULATOR_HOST=localhost:8685
QF_REALDATA=1 .venv/bin/python -m pytest -m realdata -v
docker compose -f docker-compose.emulators.yml down -v
```

**Stub results on the 2026-10-22 cache** (cut 2022, predict 2024, seeds 0-1):

| Module | mae_before | mae_after (oracle) | violations (introduced) | baseline_violations | Gate |
|---|---|---|---|---|---|
| education | 0.0107 | 0.0085 (-20.6%) | 0 | 0 | passed |
| age_gender | 0.0178 | 0.0153 (-13.8%) | 0 | 2 per seed | passed |

The age_gender baseline violations come from the real pipeline itself: its 2023 forecast for Ireland, ages 65-74 (both genders), is +0.147 and +0.178 above the last observed year, more than `max_yoy_change` (0.10). `violations` counts only what the agent introduces or worsens, so these show up in `baseline_violations` and do not fail the gate.

The published Eurostat age_gender data breaks the 0.10 year-over-year cap too. Using the published truth as the 2023-2024 forecast gives 15 `yoy_cap` violations on this cache (for example, Irish men aged 65-74 went from 62.73% in 2022 to 92.16% in 2023). The cap is therefore stricter than reality and can block corrections toward the truth.

## Live results

### Run 1 — education, synthetic benchmark (2026-10-22)

Command: `benchmark_cli --module education --seeds 0-2 --n-countries 5 --live --model nvidia/nemotron-3-ultra-550b-a55b`

| seed | MAE before | MAE after | hallucinated citations | violations | corrections |
|---|---|---|---|---|---|
| 0 | 0.0222 | 0.0141 | 0 | 0 | 1 |
| 1 | 0.0042 | 0.0042 | 0 | 0 | 0 |
| 2 | 0.0090 | 0.0084 | 0 | 0 | 1 |
| all | 0.0118 | 0.0089 (-24.6%) | 0 | 0 | 2 |

45 LLM calls, about 1003 s. Gate passed (reported target met).

How to read this: it is a very small sample (3 seeds x 5 countries, 2 corrections), so it shows the loop works end
to end against a real model and that the gate held, not that the agent reliably improves forecasts. Seed 1 made no
correction. The first attempt with `nvidia/nemotron-3-super-120b-a12b` returned HTTP 404 (model not available to
the API key used), so that model was not evaluated. The endpoint also returned intermittent HTTP 500s that succeeded on retry.
