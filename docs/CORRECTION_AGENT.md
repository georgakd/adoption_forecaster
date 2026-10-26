# Autonomous correction agent

## What it is

An LLM agent that corrects the statistical baseline forecast using cited evidence from source documents. It is **one model** playing two roles:

1. **Planner**: one call that reads a compact summary of the forecast table (per country-segment: recent history, forecast, deviation from trend, number of source documents per country) and returns a JSON list of suspicious country-segments.
2. **Researcher**: for each suspect, a bounded tool-calling loop (`search_sources`, `read_source`, `propose_correction`). A proposal carries the new value, a verbatim quote from a document, the value and year the quote states, a confidence and a reason.

Every proposal passes a **deterministic gate** before it touches the table: the cited quote must really occur in the document, the quoted value must match the proposed value within `citation_tolerance` and be stated for the proposed year, the document must name the country somewhere in its text, and the corrected table must satisfy the hard constraints (bounds, shares summing to one, historical values untouched, year-over-year change within the country-segment's cap (see `yoy_cap`), no worsening of an existing violation). Rejections are fed back to the model as the tool result. The model can never write a value the gate has not accepted.

**Gate limitation (deferred to milestone 2).** The citation check binds year, value and country only. The country check is a substring search over the whole document (so "Niger" matches "Nigeria", and a document that mentions the country anywhere passes), and the segment is not checked at all: a quote about France, women aged 35 to 44, 79.4% in 2025 is accepted as evidence for France, women aged 16 to 24, 2025. `evaluate` counts such citations independently (see `hallucinated_citations` below), but the gate itself does not stop them.

The agent never raises on LLM failure or malformed output: it returns the best table so far with an `error` string.

## Data flow

```
predictor frame --to_table--> baseline ForecastTable --run_correction(store, llm)--> corrected table + corrections + rejections + log
                                                                                   --from_table--> same-shape frame
```

Code: `adoption_forecaster/correction/` (`agent.py` loop, `tools.py` gate and tools, `citations.py`, `constraints.py`, `llm.py`, `sources.py`, `pipeline.py` hook, `evaluation.py`, `oracle.py` and `benchmark_cli.py` for the benchmark).

## Configuration (`correction:` in `adoption_forecaster/config/config.yaml`)

| Key | Meaning |
|---|---|
| `enabled` | Master switch for the pipeline hook. Off by default; with it off nothing is built or called. |
| `max_yoy_change` | Scalar cap on the year-over-year change of a corrected value (share scale, `0.10`): used for every country-segment in `yoy_cap.mode: fixed`, and as the fallback for a segment with no usable history in `adaptive` mode. |
| `yoy_cap` | `{mode: "adaptive", quantile: 0.9, k: 3.0, floor: 0.03, ceiling: 0.30}`. Per country-segment cap computed once from the baseline's historical cells (see "Adaptive year-over-year cap" below); `mode: fixed` keeps the scalar. Validated at startup (`prepare_correction`) and whenever caps are computed: `mode` must be `adaptive` or `fixed`, `0 <= quantile <= 1`, `k > 0`, `0 <= floor <= ceiling`; anything else raises `ValueError`. |
| `citation_tolerance` | Maximum absolute difference between the proposed value and the value stated in the quote. |
| `model` | NVIDIA model id (OpenAI-compatible endpoint at build.nvidia.com). Set to `nvidia/nemotron-3-ultra-550b-a55b`, the model verified working for the key used; `--model` overrides it in memory. |
| `max_iterations_per_segment` | Tool-calling turns the researcher may use per country-segment. |
| `max_llm_retries` | Retries for transient LLM errors and for unparseable planner replies. |
| `max_suspects` | Planner suspects beyond this many are logged as ignored, not researched. |
| `on_failure` | Informational: on any agent failure the baseline (plus corrections accepted so far) is kept. |

The API key is read from the `NVIDIA_API_KEY` environment variable only. It is never stored in the repo, the config, logs, exception text or result files.

## Running the benchmark

The benchmark builds a seeded synthetic world, runs the real forecasting pipeline on the data visible up to the last year to get the baseline, and generates source documents (some wrong, some conflicting) about the truth, which contains shocks the baseline misses.

```bash
# Stub mode (default): an oracle script stands in for a perfect model. No network, no key. This is a local gate: no CI workflow
# runs it (.github/workflows only build the Docker image, compare pipeline outputs and deploy).
python -m adoption_forecaster.correction.benchmark_cli --module education --seeds 0-4
python -m adoption_forecaster.correction.benchmark_cli --module age_gender --seeds 0-2 --n-countries 5

# Held-out seeds (100+): defaults to HELDOUT_SEEDS when --seeds is not given
python -m adoption_forecaster.correction.benchmark_cli --module education --heldout

# Live mode: a real model. Export the key in your own shell, never in a file.
export NVIDIA_API_KEY=...   # your key
python -m adoption_forecaster.correction.benchmark_cli --module education --seeds 0-2 --n-countries 5 --live --model <model-id>
```

Options: `--seeds` takes `a-b` ranges and comma lists (`0-2,7`), default `0-2`; `--n-countries N` (default 8); `--model` overrides `correction.model` in memory only; `--out DIR` (default `benchmark_results`, git-ignored). Live runs print the number of LLM calls and elapsed seconds so cost stays visible.

Exit codes: stub mode returns 0 only if the gate passes, 1 otherwise. Live mode returns 0 whenever the run completes (the gate result is reported, not enforced) and 2 if `NVIDIA_API_KEY` or the model is missing.

## Reading the results

Each run writes `benchmark_results/<UTC timestamp>.json` with `config` (mode, module, model, n_countries, seeds), `per_seed` (the fields below plus `seed`, `error` and `corrections`), `aggregate`, `gate_passed`, `llm_calls` (live only) and `elapsed_seconds`. `corrections` lists every accepted correction of that seed: `module`, `country`, `segment`, `year`, `old_value`, `new_value`, `doc_id`, `quote` (cut to 200 characters) and `confidence`. In the pipeline, `maybe_correct` logs one INFO line per accepted correction with the same fields.

- `mae_before` / `mae_after`: mean absolute error against the truth over forecast years, computed per country-segment and then averaged over country-segments. The headline metric. The aggregate is the mean over seeds; `relative_mae_change = (after - before) / before` (negative is better).
- `hallucinated_citations`: accepted corrections whose citation either fails the gate's own check (quote absent from the document, value or year mismatch, country not named), or, checked independently of the gate, cites a document that is not one of the benchmark's documents or whose own `country`, `segment` and `year` metadata (benchmark ground truth, never shown to the agent) differ from the corrected cell. Only this second part can be non-zero for corrections the gate accepted, for example a quote about another age group of the same country and year. Summed over seeds.
- `violations`: hard-constraint violations the corrections introduced or worsened relative to the baseline (the same rule `propose_correction` enforces). Summed over seeds.
- `baseline_violations`: violations already present in the baseline forecast, reported separately and not counted against the agent (console column `base_viol`). Summed over seeds.
- `n_corrections`: accepted corrections. `error`: the agent's failure message for that seed, if any.

## Hard gate versus reported target

- **Stub model (local, not run by CI):** hard gate. Aggregate `mae_after < mae_before`, zero hallucinated citations and zero introduced or worsened violations; the CLI exits non-zero otherwise. The stub oracle proposes the cited value of a document that agrees with the truth, so a failure means the plumbing or the gate is broken.
- **Live model:** reported target. The same numbers are reported, but a miss does not fail the run.

What measures the model: `violations` and the gate part of `hallucinated_citations` are the gate's own checks re-run on what the gate already accepted, so they are zero by construction for any model, good or bad. Only the MAE change, the number of corrections and the independent metadata part of `hallucinated_citations` (a citation for the wrong country, segment or year) say anything about the model.

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
python -m adoption_forecaster.correction.benchmark_cli --module education --real --seeds 0-1
# live mode (same key rule as above)
python -m adoption_forecaster.correction.benchmark_cli --module education --real --live
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

**Stub results on the 2026-10-22 cache:** an earlier table here (seeds 0-1, fixed 0.10 cap) is superseded by "Stub benchmark, fixed versus adaptive" below.

Under the fixed 0.10 cap the age_gender baseline had 2 violations per seed, from the real pipeline itself: its 2023 forecast for Ireland, ages 65-74 (both genders), is +0.147 and +0.178 above the last observed year, more than the fixed 0.10 cap (this table predates the adaptive cap; under the adaptive default these two moves are within their caps, see below). `violations` counts only what the agent introduces or worsens, so these show up in `baseline_violations` and do not fail the gate.

The published Eurostat age_gender data breaks the 0.10 year-over-year cap too. Using the published truth as the 2023-2024 forecast gives 15 `yoy_cap` violations on this cache (for example, Irish men aged 65-74 went from 62.73% in 2022 to 92.16% in 2023). The fixed cap is therefore stricter than reality and can block corrections toward the truth.

## Adaptive year-over-year cap

The cap is a guard against unattended large moves, not a statistical test: nothing is estimated or tested, it only bounds how far an accepted correction may move a value from one year to the next. One scalar cannot fit both modules (Age_Gender age groups move much more than Education shares), so by default (`yoy_cap.mode: adaptive`, code in `correction/caps.py`) each country-segment gets

`cap = clamp(k * max(own_q, peer_q), floor, ceiling)`

where `own_q` is the `quantile` (0.9) of that series' historical absolute year-over-year changes and `peer_q` the same quantile pooled over the other countries' series of the same segment; `k = 3`, `floor = 0.03`, `ceiling = 0.30`. Only historical cells (`is_forecast == False`, consecutive years, both present) are used, so the cap never depends on forecast values or on the truth. A series (or peer pool) with fewer than 3 historical changes is not used; a key with neither gets `max_yoy_change`. `CorrectionSession` computes the caps once from the baseline; the gate, `evaluate` (`violations` and `baseline_violations`) and the prompts use the same caps. The planner summary shows `cap=` on each row and the researcher's segment brief states its cap; the system prompts state no global number.

**Validation on the real cache** (`k` and `quantile` were chosen after looking at origin cut 2022 / pred 2024; origin cut 2021 / pred 2023 is reported without re-tuning). "Published moves" are the year-over-year changes of the values Eurostat published after the cut (each relative to the previous year), checked against caps computed from the history up to the cut:

| Origin | Module | Published moves | Flagged, fixed 0.10 | Flagged, adaptive | Adaptive cap median (min-max) | Baseline yoy violations fixed / adaptive |
|---|---|---|---|---|---|---|
| cut 2022, pred 2024 (tuning) | age_gender | 502 | 15 | 1 | 0.176 (0.071-0.300) | 2 / 0 |
| cut 2022, pred 2024 (tuning) | education | 84 | 0 | 0 | 0.049 (0.048-0.275) | 0 / 0 |
| cut 2021, pred 2023 (out of sample) | age_gender | 500 | 16 | 2 | 0.179 (0.071-0.300) | 0 / 0 |
| cut 2021, pred 2023 (out of sample) | education | 84 | 0 | 0 | 0.050 (0.048-0.300) | 0 / 0 |

So the adaptive cap is looser than 0.10 for most Age_Gender segments (where the published data moves more) and about half of 0.10 for most Education series (where it is stricter than the fixed cap), and on both origins it flags at most 2 published moves.

**Stub benchmark, fixed versus adaptive** (oracle stub, seeds 0-2; synthetic with `--n-countries 5`; `--real` on the cache). All runs passed the gate with 0 hallucinated citations and 0 introduced violations:

| Data | Module | mae_before | mae_after fixed (corrections) | mae_after adaptive (corrections) |
|---|---|---|---|---|
| synthetic | education | 0.0118 | 0.0061, -48.1% (4) | 0.0111, -5.4% (2) |
| synthetic | age_gender | 0.0140 | 0.0108, -22.6% (63) | 0.0106, -24.1% (64) |
| real, cut 2022 | education | 0.0107 | 0.0092, -13.8% (10) | 0.0092, -13.8% (10) |
| real, cut 2022 | age_gender | 0.0178 | 0.0153, -13.9% (132) | 0.0141, -20.4% (150) |
| real, cut 2021 | education | 0.0092 | 0.0081, -12.1% (8) | 0.0081, -12.1% (8) |
| real, cut 2021 | age_gender | 0.0209 | 0.0184, -11.8% (133) | 0.0176, -16.0% (148) |

Real age_gender `baseline_violations` (cut 2022) fall from 6 (2 per seed) to 0. The synthetic education drop is expected: the synthetic world's shocks (0.03-0.08) are much larger than the year-over-year changes of its smooth history (seed 0 education caps 0.038-0.059), so the adaptive cap rejects corrections toward them. The tests that exercise the agent on those synthetic shocks therefore pin `mode: fixed`; the adaptive path has its own tests (`tests/correction/test_caps.py`).

## Live results

### Run 1 — education, synthetic benchmark (2026-10-22)

**Not a measurement of today's default.** This run was made under the fixed 0.10 cap, before `yoy_cap.mode: adaptive` became the default. Under the adaptive cap the oracle stub's gain on the same synthetic Education benchmark falls from -48.1% to -5.4% (see "Stub benchmark, fixed versus adaptive"), so this live result would very likely be smaller today. It has not been rerun.

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

## Before enabling `correction.enabled`

Open items, none of them done yet. The flag also cannot be enabled from the real CLI until a milestone-2 source store exists (`NoSourceStoreError`).

- **Gate-level segment and country binding.** The gate does not check the segment, and its country check is a substring search over the whole document (see "Gate limitation"). Bind both to the cited document's own metadata in the gate, not only in `evaluate`.
- **Verifier: comma-clause pairing and quote-level country check.** `citations.py` splits clauses only on sentence ends and `;`, so "in 2025 it was 60%, in 2026 it was 62%" can mis-pair numbers and years; the country must appear in the quote, not just somewhere in the document.
- **Education age-range blindness.** `processors.py:117` calls `maybe_correct` once per age group passing only `"education"`, so the agent and the gate cannot tell 16-64 from 16-74: the 16-74 frame gets corrections nobody evaluated, at twice the LLM cost. Only 16-64 is benchmarked.
- **LLM call budget.** The OpenAI SDK's default retries (2) stack on the client's own retries (`max_llm_retries`, 3), the SDK default timeout applies, and there is no overall call or time budget per run. Set `max_retries=0` and an explicit `timeout` on the SDK client, and add a run budget.
- **Untrusted document text.** No rule in `prompts.py` tells the model that document text is untrusted data and that instructions inside it must be ignored.
- **Dedicated script-exhaustion error.** `agent.py:211-214` and `pipeline.py` re-raise every `AssertionError` (meant for a too-short test script); replace this with a dedicated `ScriptExhausted` exception so a real assertion cannot crash the pipeline.
- **Corrections artifact.** Write the accepted corrections (today only logged at INFO) to a file next to the pipeline outputs.
- **CI job.** Add a workflow running `pytest -m "not emulators and not realdata"` plus the stub benchmark; today neither runs in CI.
- **Live runs under the adaptive cap.** Run one live held-out seed and one live age_gender run under the adaptive default; the only recorded live run used the fixed cap.
