# Autonomous forecast correction — design

Date: 2026-10-22
Status: draft, awaiting user approval

## Intent and why

Today a market researcher corrects the internet-penetration forecasts
(Education and Age_Gender modules) by hand, using external data sources.
The goal is to remove that human completely: an autonomous agent finds its own
external evidence, corrects the forecasts, and can show its sources.

Because nobody reviews its output, trust has to come from automated checks
and from a benchmark that scores it, not from a person reading results.

## Current state (from repo scan)

- Core pipeline: Eurostat data -> Theta / random walk / logistic / LASSO
  ensemble -> CSVs to local disk or GCS -> optional Pub/Sub notification
  (protobuf). Config in `quotas_forecasting/config/config.yaml`. Entry:
  `quotas_forecasting/main.py`.
- `quotas_forecasting/AI_adjustments/`: separate CLI, OpenAI GPT-4o
  tool-calling loop, one country (or country x age/gender) at a time, fixed
  rules (e.g. year-over-year change > 1%), `WorkflowEnforcer` to keep the model
  on sequence. Run by hand, not wired into the pipeline. Flat imports
  (`from client import ...`) so it only runs with its own folder as cwd, and it
  has its own `requirements.txt`.
- No tests. No sample input data in the repo (inputs live in a GCS bucket the
  author cannot access). README mentions files that do not exist
  (`cloudbuild.yaml`, `post-merge-run-forecast.yml`).

## Goals

1. **Synthetic data**: generate realistic fake inputs matching the schemas the
   loader expects, so the whole pipeline runs with no real data and no cloud
   access.
2. **Local cloud emulation**: a docker compose with a GCS emulator and a
   Pub/Sub emulator so the `prod` code path (upload + notification) runs
   end to end locally.
3. **Targeted refactor** before the agent work, verified by end-to-end tests
   on the synthetic data (see Scope of refactor).
4. **Milestone 1 — autonomous correction agent**, scored on a synthetic
   benchmark:
   - starts from the statistical baseline forecast (kept, not removed),
   - plans which countries / segments look suspicious,
   - researches evidence, proposes corrections each with a source citation and
     confidence,
   - a verify step re-checks hard constraints and that each cited number
     appears in its source; failures are rejected or retried,
   - uses a strong model through the NVIDIA API endpoint (OpenAI-compatible),
     behind a thin client so the model/provider can be swapped.
5. **Evaluation harness**: error against known truth before vs after the agent,
   count of hallucinated citations, count of constraint violations. One headline
   number to improve.
6. **Milestone 2 (later, separate design)**: RAG over real, collected sources.

## Non-goals

- Changing the production forecasting behavior or output format.
- Changing Pub/Sub / protobuf contracts with the downstream service.
- Milestone 1 does not use live web search or real sources; that waits until
  the loop works on synthetic data. (Builder's call to revisit.)
- Removing the statistical models. Dropping them is only a later experiment: run
  the LLM-only variant next to the baseline and compare on the benchmark.

## Anti-goals (failure even if it "works")

- An agent that produces confident numbers with no verifiable source.
- Benchmark scores that only reflect the agent memorising the synthetic
  generator (keep generator seeds and "source noise" patterns varied between
  dev and held-out evaluation).
- A refactor that silently changes forecast outputs.

## Constraints

- Python 3.11, Poetry (as in the repo). Work in the current checkout, no git
  worktrees.
- No access to real input CSVs; everything must run from synthetic data.
- Must keep the existing `prod` code path working (GCS + Pub/Sub), tested via
  the emulators.
- NVIDIA API key availability and exact model are not yet known; the client
  must read them from configuration/environment and the benchmark must be
  runnable with a stub/fake model for tests.

## Decisions made with the user

- Remove the human researcher from the loop entirely.
- The agent finds its own sources; user is open to a RAG as milestone 2.
- Synthetic data to simulate the market-research setting.
- Refactor first, lightly; GCS + Pub/Sub emulators in docker compose.
- NVIDIA API endpoint with a strong model is acceptable.
- The agent is plugged into the pipeline in milestone 1: its corrected
  forecasts replace the baseline in the CSV and Pub/Sub outputs, behind a config
  flag (builder's call on the default; the baseline-only path must keep working).
  A prod-path test with the emulators covers corrected output.
- A real-data backtest from the public Eurostat API is added (user approved): cut the real series at 2022, forecast 2024, score against actually published values; source documents stay synthetic wrappers around real numbers. It runs after the synthetic benchmark.
- A maths review of the existing statistical models, with experimental GAM, entropy-balancing and kernel-balancing alternatives compared by rolling-origin backtest, is added (user request). Any method that wins becomes an option behind `prediction.method`; the default stays `ensemble` and the goldens stay unchanged.
- The user has an NVIDIA API key. One live-model benchmark run is part of "done"
  for milestone 1, alongside the stub-model tests. The key is read from the
  `NVIDIA_API_KEY` environment variable only; it must never be written to the
  repo, config files, docs or test fixtures. Model choice is left to the builder
  (budget is the user's own key, so keep live runs small).

## Calls made by me (not yet confirmed by the user)

- Keep the statistical baseline as the agent's starting point.
- New package alongside `AI_adjustments`; do not edit the old agent until the
  new one beats it on the benchmark.
- Milestone 1 success = beats the raw baseline on the synthetic benchmark with
  no hallucinated citations and no constraint violations. (The user was asked
  and did not specify.)
- Order of work: synthetic data -> emulators + e2e tests -> refactor -> agent.
- Live web search deferred past milestone 1.
- Truth and sources: truth is the generator's latent penetration curve. The
  baseline is not artificially degraded; it fails naturally on injected shocks,
  structural breaks and proxy countries. Source documents hold the true value
  with noise; some are wrong or contradict each other, so the agent must judge.
- Hard constraints: values within [0, 1]; secondary + tertiary = 1 (Education);
  historical values untouched; a cap on year-over-year change. The old agent's
  1% year-over-year rule is a flag to investigate, not a limit on the correction.
  These are ported from the existing `AI_adjustments` rules.
- Citation check: the cited number must match a number in the source within a
  numeric tolerance, with matching year and unit.
- Headline metric is MAE against truth, per country-segment, averaged.
  "Beats baseline" means lower aggregate MAE. Zero hallucinations and zero
  violations is a hard gate with the stub model in CI and a reported target
  with a live model.
- Refactor safety: golden outputs captured from synthetic data before the
  refactor, with pinned seeds and a numeric tolerance for the LASSO/logistic
  fits.
- The old agent is a code reference only, not a second benchmark baseline.

## Scope of refactor

In:
- Make `AI_adjustments` an importable package (relative/absolute imports, no
  cwd dependence), one dependency set in `pyproject.toml`.
- Fix docs drift (README file names, mentions of missing files).
- Whatever small changes are needed to point the code at emulators through
  config/env.

Out (for now): splitting large files (`forecasting/utils.py`,
`forecasting/quotas_forecasting.py`, `fetch_data/fetch_data_from_Eurostat.py`),
changing forecasting math.

## Left to the builder

- Statistical model of the synthetic world (growth curves, saturation, noise,
  proxy-country structure, missing data patterns) as long as it matches the
  loader schemas and exercises proxy / missing-data paths.
- How "external source" documents are represented and how corruption is
  injected.
- Agent architecture details (planner / researcher / verifier split, retry
  policy, token budget), tool set, prompt wording.
- Choice of emulator images, test framework, and benchmark metric definitions
  beyond those named above.
