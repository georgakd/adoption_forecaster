# Maths review of the forecasting models

Date: 2026-10-22. Scope: the production ensemble (Theta, smoothed random walk, logistic, and the LASSO proxy path) and three
experimental alternatives: a GAM, entropy balancing and kernel balancing. Every verdict below rests on a test in
`tests/experimental/test_model_review.py` or on a backtest number produced by `quotas_forecasting/backtest/compare.py`. Nothing here
is asserted from reading the code alone.

**Outcome in one paragraph.** Suspect (a) changes no value. The blend weights in (b) were never justified, but the backtests now
support them: none of the alternatives tried does better on both benchmarks. The LASSO proxy path in (c) is unused in production,
and it is broken if switched on: Age_Gender crashes, and once that crash is bypassed it writes 1% into the gaps. The logistic
ceiling in (d) cannot reach the 100% values that about a fifth of the real Age_Gender cells hold. The `replace(0, nan)` in (e) is
needed, but a missing tertiary level silently sets secondary to exactly 1. LASSO with too few proxies (f) crashes or falls back to
the proxies' mean. None of the experimental methods passed the decision rule. `prediction.method` stays `"ensemble"`, the only
accepted value, so the goldens are unchanged.

## How to reproduce

```
python -m quotas_forecasting.backtest.compare --out benchmark_results/model_review.md   # about 26 min, offline, single-threaded BLAS
python -m quotas_forecasting.backtest.compare --anchor-robustness                      # GAM with both anchor choices only
python -m pytest tests/experimental -v                                                  # suspect evidence + method tests (about 25 s)
```

Everything is deterministic: synthetic seeds are fixed, the real-data inputs come from the offline cache `.cache/eurostat/`
(fetched 2026-10-22, manifest `last_year` 2024), and no fetcher, emulator, network or LLM is used. The tables in this document are
copied from that command's output. The command also writes per-origin columns, which are omitted here only where noted.

## Benchmarks

- **Synthetic** (`build_benchmark`, Task 6): 8 countries per seed. The truth follows logistic diffusion curves, plus unforecastable
  shocks after `last_year` = 2024 in about 30% of countries. Forecast years are 2025 and 2026. *Dev seeds 0-9* were used for tuning.
  *Held-out seeds 100-109* were never used for tuning, and the decision rule is applied to them.
- **Real** (`build_real_backtest`, Task 13): the real pipeline runs on Eurostat inputs cut at origin c and predicts c+1 and c+2. It is
  scored on the values Eurostat later published. There are 22 countries, and the Education benchmark covers the 16-64 frame.
  - Reported origins are 2019-2022, the window in which `pred_year - cut_year == validation.year_difference == 2` holds and the cache
    contains the truth.
  - **Education ran at all four origins.**
  - **Age_Gender ran at 2021 and 2022 only.** At 2019 and 2020 the pipeline itself fails with `KeyError: None` inside
    `adjust_100prc_predictions_cases`. That rule needs three valid observations per series between `start_year` (2017) and the
    cut, and some real series have gaps there. Test `test_d_ceiling_rule_needs_three_observations_per_series` pins this; see
    suspect (j).
  - Origin 2018 (Education only) was used for tuning and is not reported as a result. **It is not fully out of sample**: it is
    scored on 2019-2020, and the earliest reported origin (2019) is scored on 2020-2021, so the 2020 truth cells were used both to
    choose `GAM_ANCHOR` and to report results. Nothing earlier can serve instead (Age_Gender cannot run before 2021, see (j)).
    The anchor-robustness check under Results scores both anchor choices to show whether the decision depends on it.
- **Metric**: `correction.evaluation._mae`. This takes the MAE over the forecast cells of each (country, segment), then averages
  over (country, segment). Cells with no published truth are dropped.

## Suspects: verdict, evidence, decision

### (a) `predictors.py:90-91` applies the same backward interpolation twice (`# ?`)

- **Verdict: refuted. The second pass never changes a value.**
- **Evidence:**
  - `test_a_second_backward_interpolation_changes_nothing` (20 random frames with leading, interior and trailing gaps) shows the
    operation is idempotent. The first pass fills the leading and interior gaps, and `limit_direction="backward"` can never fill
    the trailing gaps that remain.
  - `test_a_and_c_lasso_paths_are_never_called_by_the_default_pipeline` shows that the method containing these lines
    (`EducationPredictor.train_test_split_`) is dead code. The same is true of `train_test_split_UK`, `Lasso`,
    `combine_regressors` and `backtest_UK`: the full Education and Age_Gender pipelines run with all five replaced by functions
    that raise.
  - The duplicated pair also appears in the live-when-enabled copy at `proxy_handlers.py:209-210`, where it is equally harmless.
- **Decision:** no numerical change is needed. Deleting the dead methods in `predictors.py` (lines ~66-235) is recommended as a
  separate clean-up. It is not done here because this task must not touch the production numerics.

### (b) Blend weights are fixed with no recorded justification

- **Verdict: confirmed that there was no justification. The backtests now give one: the configured weights are within noise of
  the best of the alternatives tried, and no alternative is better on both benchmarks.**
- **Evidence:** the full pipeline was re-run with each weight vector (`blend_weight_experiment`). A component used alone means its
  weight is 1. The first row of each module is the current config.

| benchmark | module | weights (Education: theta/rw; Age_Gender: theta/rw/logistic) | MAE |
|---|---|---|---|
| synthetic held-out | education | 0.70/0.30 (config) | 0.0195 |
| synthetic held-out | education | 1.00/0.00 | 0.0203 |
| synthetic held-out | education | 0.00/1.00 | 0.0191 |
| synthetic held-out | education | 0.50/0.50 | 0.0192 |
| real (4 origins) | education | 0.70/0.30 (config) | 0.0100 |
| real (4 origins) | education | 1.00/0.00 | 0.0101 |
| real (4 origins) | education | 0.00/1.00 | 0.0119 |
| real (4 origins) | education | 0.50/0.50 | 0.0102 |
| synthetic held-out | age_gender | 0.33/0.33/0.33 (config) | 0.0216 |
| synthetic held-out | age_gender | 1.00/0.00/0.00 | 0.0255 |
| synthetic held-out | age_gender | 0.00/1.00/0.00 | 0.0226 |
| synthetic held-out | age_gender | 0.00/0.00/1.00 | 0.0228 |
| synthetic held-out | age_gender | 0.50/0.50/0.00 | 0.0238 |
| real (2 origins) | age_gender | 0.33/0.33/0.33 (config) | 0.0193 |
| real (2 origins) | age_gender | 1.00/0.00/0.00 | 0.0204 |
| real (2 origins) | age_gender | 0.00/1.00/0.00 | 0.0228 |
| real (2 origins) | age_gender | 0.00/0.00/1.00 | 0.0208 |
| real (2 origins) | age_gender | 0.50/0.50/0.00 | 0.0206 |

- **Reading the table:**
  - **Age_Gender:** the equal blend beats every single component and the logistic-free blend on both benchmarks. Dropping the
    logistic costs 5-10%.
  - **Education:** the benchmarks disagree. The random walk alone is the best on synthetic data (0.0191 against 0.0195) and the
    worst on real data (0.0119 against 0.0100). The 0.7/0.3 blend is the best on real data and within 2% of the best on synthetic
    data.
  - The grid is small and is evaluated on the reporting data. It explains the current weights; it is not a tuning result.
- **Decision:** keep the weights. Record this table as their justification.

### (c) The LASSO proxy path is inactive (`target_markets: {}`)

- **Verdict: confirmed dead in production. If enabled, it is broken in both modules.**
- **Evidence:**
  - *Inactive:* `test_a_and_c_lasso_paths_are_never_called_by_the_default_pipeline` replaces `ProxyHandler.apply_LASSO_...` and
    `quotas_forecasting.train_lasso` with functions that raise. Both pipelines still complete.
  - *Age_Gender, enabled:* `test_c_age_gender_lasso_path_crashes_when_enabled` (Austria as the target, missing 2023-2024, with
    four proxies) raises `TypeError: 'NoneType' object is not subscriptable`. The cause is that `extend_time_series_for_group`
    calls `backtest_model(lasso, X_train, y_train)` without `config`.
  - *Age_Gender, crash bypassed:* `test_c_..._writes_one_percent_when_the_crash_is_bypassed` stubs `backtest_model`. Every filled
    cell then becomes 0.01. The data are still in percent (0-100) when the LASSO output is clipped to `[min_prediction,
    max_prediction] = [0, 1]`, and the later `/100` turns the clip into 1%.
  - *Age_Gender, leakage:* the code also uses `y_train = group[str(year - 1)]`, which is the last column of `X_train`. That leaks
    the target into the features.
  - *Education, enabled:* `test_c_education_lasso_path_needs_exactly_three_hardcoded_targets` raises `IndexError` with one target
    market. `get_proxy_countries` hard-codes three targets (Iceland, North Macedonia, UK) by position, with a lower correlation
    threshold for the second one.
  - *What it is meant to do:* the path backfills a target country's missing recent years from proxy countries. It is not a forecast
    of the prediction horizon.
- **Decision:** leave it disabled. It must not be switched on until it is fixed: the missing `config`, the percent-scale clip, the
  leaking `y_train` and the hard-coded three targets. The balancing donor-weighting below is a tested alternative for this
  backfilling job, but it was not adopted (see the results).

### (d) Clipping to [0, 1] versus the logistic ceiling `K_max` 0.995 and `K_guess_factor`

- **Verdict: confirmed. The logistic component cannot represent saturated series, and the rule that compensates for this is
  deterministic rather than random.**
- **Evidence:**
  - `test_d_logistic_component_cannot_reach_a_saturated_series`: on a series that is constant at 1.0, the logistic forecast is at
    most 0.995 (K is bounded by `K_max`).
  - On real data, 17-19% of Age_Gender series end at 0.995 or above, and 22-28% of the truth cells are at 0.995 or above (diagnostics
    table below). For those cells the logistic component is biased low by at least 0.005 before blending.
  - `test_d_ceiling_rule_b_random_factor_is_one_constant`: rule (b) multiplies by `random.uniform(1, 1/K_max)`, but it re-seeds with
    42 before every row, so every adjusted row gets the same factor of 1.0032132.
  - `K_guess = max(y) * K_guess_factor * K_max` is only the starting point for `curve_fit`. It is always at most 0.985, so it is
    feasible. It does not cause bias; the bound does.
  - The logistic alone still helps the blend (see (b)).
- **Decision:** no change in this task, because changing it would change the goldens. Two fixes are recommended for a separate
  change, each to be evaluated with this harness: let K reach 1.0, or apply the ceiling rule before blending; and replace the fake
  random factor with an explicit constant.

### (e) `.replace(0, np.nan)` on the Education ratios

- **Verdict: the replace is necessary, and harmless for real zeros (shares of exactly 0 do not occur). The suspect led to a real
  defect next to it.**
- **Evidence:**
  - `test_e_replace_zero_undoes_the_groupby_sum_of_an_all_missing_year`: `merge_education_levels` uses `groupby().sum()`, which
    turns an all-missing year into 0. The replace turns it back into a gap.
  - `test_e_a_missing_tertiary_level_makes_secondary_exactly_one`: when only ISCED level 3 is missing in a year,
    `calculate_internet_users_ratio` divides by the sum of levels 1 and 2. Secondary then becomes exactly 1.0 and tertiary becomes
    NaN.
  - This happens in the real data. Portugal has two such history cells at every origin (for example 2006, secondary = 1.0); see the
    diagnostics table.
  - Because S + T = 1 holds everywhere else, these cells are the only reason the per-level forecasts stop summing to 1 before
    normalization. For Portugal at origin 2022 the pre-normalization sum is off by 3.9e-3, against at most 4e-9 for every other
    country. The probe that measured this was run separately from `compare.py` and is not part of the reproducible output.
- **Decision:** keep the replace. Recommended fix, separately: treat a year as missing for both levels when any level is missing.

### (f) LASSO CV with very few proxy rows (`n_splits = min(5, rows)`)

- **Verdict: confirmed. One or two proxy rows cannot work.**
- **Evidence:**
  - `test_f_one_or_two_proxy_rows_cannot_be_cross_validated`: with one proxy, the Education path fails on `KFold(n_splits=1)` with a
    `ValueError`. With two rows, the Age_Gender path uses `cv = n - 1 = 1`, which sklearn rejects.
  - A separate probe (20 replications, about 18 year-features, a common trend) gave these results:

| proxy rows | outcome | error |
|---|---|---|
| 2 | every coefficient is zero, so the prediction is the proxies' mean | 0.133 |
| 3 | not degenerate | about 0.01, the same as a naive trend |
| 4-5 | not degenerate | about 0.004 |

- **Decision:** this only matters if (c) is enabled. Before then, require at least 3 proxies, or replace the path with
  donor-weighting.

### Further issues found

- **(g) Education secondary + tertiary = 1 holds by construction, and the two forecasts are redundant.** The ratios are shares of
  internet users by education (levels 1+2 against level 3), so S = 1 - T in the history (largest deviation 2.2e-16 on real data,
  apart from the Portugal cells in (e)). `test_g_random_walk_and_theta_forecasts_of_the_two_shares_sum_to_one` shows that the
  smoothed random walk is exactly antisymmetric (a median of negated differences), and so is Theta, to 1e-9. The S/(S+T)
  normalization in `EducationPredictor.predict` is therefore a no-op except where (e) has corrupted the history. Forecasting
  tertiary alone and setting secondary = 1 - tertiary (as the experimental methods do) is equivalent and simpler. No change is
  made, to keep the goldens.
- **(h) 16-64 against 16-74.** These are two independent pipeline runs on different input rows (the `Age group` filter in
  `standardize_columns_with_regex`). Neither benchmark scores the 16-74 frame: both capture only the first, 16-64, predictor
  call. This is a limitation of this review, not a defect.
- **(i) The two-year horizon is hard-coded.** `test_i_education_predictor_cannot_forecast_three_years_ahead`: `predict(df, 2022,
  2025)` raises `KeyError`. The `for i in [1, 2]` loop, `theta.return_last_n: 2` and the two-step random walk all assume the
  horizon, and `modules.education.settings.n_predictions` is never read by the Education predictor. With `validation.year_difference
  == 2` the configuration is consistent, but the parameter only appears to be configurable.
- **(j) The ceiling rule assumes three observations per series.** `adjust_100prc_predictions_cases` takes the last three valid
  indices with no guard, so short or gappy series crash the Age_Gender pipeline (`KeyError: None`). This is why the real origins
  2019 and 2020 cannot run.
- **(k) Units.** Education inputs are divided by 100 on load. Age_Gender inputs are divided by 100 only in `create_train_data`,
  after the LASSO extension, which is the cause of the 1% bug in (c). `scaling_factor: 100` is used only by that LASSO path.

### Real-data diagnostics (from `compare.py`)

| origin | module | fact | value |
|---|---|---|---|
| 2019-2022 (each) | education | history cells with secondary == 1 and tertiary NaN | 2 |
| 2019-2022 (each) | education | max abs(secondary + tertiary - 1) in the forecast (after normalization) | 0.0000 |
| 2021 | age_gender | share of series with last value >= K_max (0.995) | 0.1742 |
| 2021 | age_gender | share of truth cells >= 0.995 | 0.2198 |
| 2022 | age_gender | share of series with last value >= K_max (0.995) | 0.1894 |
| 2022 | age_gender | share of truth cells >= 0.995 | 0.2771 |

## Experimental methods

All three are in `quotas_forecasting/forecasting/experimental/` and run only through `compare.py`. They see exactly the history the
pipeline's baseline table holds and forecast the same cells. For Education they forecast the tertiary share only and set secondary
= 1 - tertiary, so the hard constraints ([0, 1] and S + T = 1) hold by construction (`test_method_tables_keep_history_and_satisfy_hard_constraints`).

- **GAM (`gam_forecast`)**
  - *Model:* a `pygam` `LinearGAM` with one cubic P-spline term in the year, fitted on logit(clip(y, 0.001, 0.999)). It uses at most
    8 splines (no more than the number of points), a monotone non-decreasing constraint, and a smoothing penalty chosen by GCV over
    `logspace(-3, 3, 7)`. The output is clipped to [0, 1] and made non-decreasing. Fewer than 4 points fall back to the last value.
  - *Extrapolation:* pygam continues the spline linearly past the last knot. On the logit scale that is a logistic-shaped
    continuation that cannot leave (0, 1). Its slope, however, is estimated from the last few points, which is where spline
    forecasts fail.
  - *Tuning (dev seeds and real origin 2018; the 2018 origin's scored years overlap the 2019 origin's truth, see Benchmarks):*
    anchoring shifts the curve on the logit scale so that it passes through the last observation. It lowered MAE on all three
    tuning sets (dev Education 0.0153 to 0.0152, dev Age_Gender 0.0194 to 0.0187, real Education 2018 0.0135 to 0.0078), so
    `GAM_ANCHOR = True` was fixed before the reported runs. No other option was tuned. Because of the overlap, both anchor
    choices are scored on the reporting data in the anchor-robustness check below.
- **Entropy and kernel balancing (`entropy_balance_weights`, `kernel_balance_weights`, `proxy_forecast`)**
  - *Domain note:* these are survey-weighting and covariate-balancing tools. They reweight units so that the weighted units match
    target moments. They are not forecasters. Here they are used only as donor weighting for proxy forecasts, in the spirit of
    synthetic control. **Weighting an actual respondent sample to the quotas is out of this repo's scope.**
  - *Donors:* the donor units are windows. Each is a series at an anchor year t, described by (level y_t, slope (y_t - y_{t-2})/2),
    with its known path y_{t+1..t+h} - y_t. The pool is every series of the module in the same table, including the target's own
    past. For Education that means tertiary series only.
  - *Forecast:* the forecast is the target's last value plus the weighted donor path, clipped to [0, 1]. In a backtest, donor
    windows end at or before the cut, so nothing is seen from the future.
  - *Entropy balancing* (Hainmueller 2012) uses weights of minimum KL divergence to uniform that reproduce the target's features
    exactly. They are solved through the convex dual with `scipy.optimize` (trust-exact), after a linear-programming hull check, and
    `InfeasibleBalanceError` is raised outside the hull. In `proxy_forecast` an infeasible target falls back to kernel weights. The
    share of fallbacks is reported as `fallback_share`.
  - *Kernel balancing* (cf. Hazlett 2020) uses simplex weights that minimize the RBF-kernel mean discrepancy to the target. Features
    are standardized, the length scale is the median heuristic, and the problem is solved by accelerated projected gradient. It is
    always feasible.
  - *Tuning:* none. Lag 2 and the (level, slope) features were fixed before any run.

## Results

**Decision rule, stated before the results** (`WIN_MARGIN = 0.05` in `compare.py`; test `test_win_margin_is_stated_before_results`):
a method becomes an option for a module only if its MAE is at least 5% lower than the ensemble's on both the synthetic held-out seeds
and the mean over the real origins.

Synthetic, held-out seeds 100-109 (origin 2024). The dev seeds show the same ordering.

| module | method | MAE | fallback_share |
|---|---|---|---|
| education | ensemble | 0.0195 | |
| education | gam | 0.0183 | |
| education | entropy_proxy | 0.0187 | 0.4125 |
| education | kernel_proxy | 0.0184 | 0.0000 |
| age_gender | ensemble | 0.0216 | |
| age_gender | gam | 0.0218 | |
| age_gender | entropy_proxy | 0.0199 | 0.1260 |
| age_gender | kernel_proxy | 0.0198 | 0.0000 |

Synthetic, dev seeds 0-9 (used for GAM tuning):

| module | ensemble | gam | entropy_proxy | kernel_proxy |
|---|---|---|---|---|
| education | 0.0161 | 0.0152 | 0.0152 | 0.0155 |
| age_gender | 0.0183 | 0.0187 | 0.0165 | 0.0167 |

Real Eurostat backtest (cut c, predict c+1 and c+2). Age_Gender origins 2019 and 2020 cannot run; see (j).

| module | method | MAE (mean) | 2019 | 2020 | 2021 | 2022 | fallback_share |
|---|---|---|---|---|---|---|---|
| education | ensemble | 0.0100 | 0.0093 | 0.0109 | 0.0092 | 0.0107 | |
| education | gam | 0.0110 | 0.0079 | 0.0091 | 0.0113 | 0.0156 | |
| education | entropy_proxy | 0.0113 | 0.0111 | 0.0126 | 0.0106 | 0.0111 | 0.0455 |
| education | kernel_proxy | 0.0117 | 0.0117 | 0.0132 | 0.0108 | 0.0113 | 0.0000 |
| age_gender | ensemble | 0.0193 | - | - | 0.0209 | 0.0178 | |
| age_gender | gam | 0.0216 | - | - | 0.0225 | 0.0208 | |
| age_gender | entropy_proxy | 0.0199 | - | - | 0.0193 | 0.0204 | 0.0436 |
| age_gender | kernel_proxy | 0.0199 | - | - | 0.0215 | 0.0184 | 0.0076 |

Decision (relative MAE reduction against the ensemble; positive is better):

| module | method | gain synthetic (held-out) | gain real | wins |
|---|---|---|---|---|
| education | gam | +6.0% | -9.3% | no |
| education | entropy_proxy | +4.2% | -12.9% | no |
| education | kernel_proxy | +5.5% | -17.0% | no |
| age_gender | gam | -0.7% | -11.9% | no |
| age_gender | entropy_proxy | +8.1% | -2.7% | no |
| age_gender | kernel_proxy | +8.7% | -3.1% | no |

**GAM anchor robustness.** The tuning origin overlaps the reported truth (see Benchmarks), so the GAM was scored with both anchor
choices on the reporting data (`python -m quotas_forecasting.backtest.compare --anchor-robustness`, 265 s):

| module | benchmark | ensemble MAE | GAM, anchor=False | GAM, anchor=True (configured) | gain, anchor=False | gain, anchor=True | wins (either) |
|---|---|---|---|---|---|---|---|
| education | synthetic held-out | 0.0195 | 0.0184 | 0.0183 | +5.8% | +6.0% | |
| education | real (2019-2022) | 0.0100 | 0.0154 | 0.0110 | -53.2% | -9.3% | no |
| age_gender | synthetic held-out | 0.0216 | 0.0226 | 0.0218 | -4.6% | -0.7% | |
| age_gender | real (2021-2022) | 0.0193 | 0.0197 | 0.0216 | -2.0% | -11.9% | no |

Real Education GAM by origin, anchor=False: 0.0132 / 0.0144 / 0.0158 / 0.0181 for 2019-2022 (anchor=True: 0.0079 / 0.0091 / 0.0113 /
0.0156). **The decision does not depend on the anchor choice:** with either value the GAM loses to the ensemble on real data in both
modules and fails the rule. The configured anchor is not uniformly the better one: on real Age_Gender the unanchored GAM is closer
to the ensemble (-2.0% against -11.9%), but it still loses.

**Reading the results honestly.**

- **No method wins.** Every alternative is worse than the ensemble on real data. The methods that win on synthetic data (balancing
  on Age_Gender, GAM and kernel on Education) do so on a benchmark whose truth is generated from the same logistic family their
  donors and splines describe well. The synthetic benchmark flatters curve-shaped methods.
- **The GAM is unstable at the boundary, as expected.** It is the best Education method at origins 2019 and 2020 and clearly the
  worst at 2022 (0.0156 against 0.0107). The end-slope extrapolation transmits the last few years' changes, which is the known
  failure mode.
- **The balancing proxies come close but do not win.** On real Age_Gender, entropy balancing wins at 2021 but loses at 2022.
  Entropy balancing is infeasible for 41% of synthetic Education targets (the target's level and slope lie outside the donor hull,
  since there are only 8 countries), so there it is effectively kernel weighting.
- **Decision:** no method becomes an option, and no default changes. `prediction.method` accepts only `"ensemble"`
  (`predictors.prediction_method`, which `get_predictor` validates). The experimental code stays in the repo, documented and
  tested, for re-evaluation as more real origins become available. A re-evaluation should not tune on 2019-2022 real data.

## Limits of this review

- Only 4 real Education origins and 2 real Age_Gender origins could be scored, so the differences of a few percent above are within
  origin-to-origin variation. The rule's 5% margin on both benchmarks is what guards against over-reading them.
- The blend-weight grid in (b) is small and is reported on the evaluation data; it explains the current weights rather than
  optimizing them.
- The 16-74 Education frame is not scored (h).
- Dependencies: `pygam` (and `threadpoolctl`, used by `compare.main`) are in the optional Poetry group `experiments`
  (`poetry install --with experiments`). The Docker image is built from `pyproject.toml` + `poetry.lock` with
  `poetry install --only=main`, so it never installs them; the lock file is regenerated by the maintainer. `gam_forecast` imports
  pygam lazily and raises an ImportError naming the group when it is missing; the tests that need it are skipped without it.
  pygam 0.12 constrains `scipy < 1.17`; locally it moved scipy from 1.17.1 to 1.16.3. The golden tests pass unchanged on 1.16.3.
