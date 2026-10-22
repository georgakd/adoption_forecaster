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
- `violations`: hard-constraint violations in the corrected table. Summed over seeds.
- `n_corrections`: accepted corrections. `error`: the agent's failure message for that seed, if any.

## Hard gate versus reported target

- **Stub model (CI):** hard gate. Aggregate `mae_after < mae_before`, zero hallucinated citations and zero violations; the CLI exits non-zero otherwise. The stub oracle proposes the cited value of a document that agrees with the truth, so a failure means the plumbing or the gate is broken.
- **Live model:** reported target. The same numbers are reported, but a miss does not fail the run: it measures the model, not the code.

## Using it in the real pipeline (milestone 2)

The real CLI cannot enable correction yet: `get_source_store` in `pipeline.py` raises `NoSourceStoreError` because no real source store exists. Until then `correction.enabled` only works when a store is injected (as the tests and benchmark do).

A milestone-2 store implements the `SourceStore` Protocol in `sources.py`:

- `search(query, country=None, year=None) -> list[SourceSnippet]`
- `search_with_total(query, country=None, year=None) -> (snippets, total_matches_before_cap)`. The planner calls `search_with_total("", country=country)` and uses the total as the number of documents for that country, so an empty query must return all documents of the country.
- `read(doc_id) -> str`: the document text, which citations are verified against.

Snippets expose only `doc_id, publisher, country, year, title`. Then implement `get_source_store(config)` to return it.

## Live results

_(to be filled after the first live run)_
