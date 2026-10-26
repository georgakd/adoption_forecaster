"""Penalized-spline GAM trend forecaster on the logit scale (pygam).

The share y in [0, 1] is mapped to z = logit(clip(y, EPS, 1 - EPS)) and z is fitted with one cubic P-spline term in the year,
the smoothing penalty chosen by GCV over ``LAMBDAS``. Beyond the last observed year pygam continues the spline linearly, so on
the logit scale the forecast follows the fitted end slope, i.e. a logistic-shaped continuation that can never leave (0, 1).
That end slope is exactly where spline forecasts tend to fail (it is estimated from the last few points only); the review
measures it rather than assuming it works.

pygam is an optional dependency (the "experiments" Poetry group) and is imported only when ``gam_forecast`` runs.
"""

import contextlib
import io

import numpy as np
import pandas as pd
from scipy.special import expit, logit

EPS = 1e-3  # 0% and 100% shares are mapped to logit(0.001) and logit(0.999)
MAX_SPLINES = 8
MIN_POINTS = 4  # cubic splines need more than spline_order (3) basis functions; fewer points fall back to the last value
LAMBDAS = np.logspace(-3, 3, 7)


def gam_forecast(series: pd.Series, horizon: int, monotone: bool = True, anchor: bool = False) -> np.ndarray:
    """Forecast the ``horizon`` years after the last observed year of ``series`` (year index, values in [0, 1], NaN allowed).

    ``monotone`` constrains the spline to be non-decreasing (and the returned path is non-decreasing). ``anchor`` shifts the
    forecast on the logit scale so the fitted curve passes through the last observation. Output is clipped to [0, 1].
    """
    try:
        from pygam import LinearGAM, s
    except ImportError as exc:
        raise ImportError("gam_forecast needs pygam: install the 'experiments' group: poetry install --with experiments") from exc
    observed = series.dropna().astype(float)
    if len(observed) < MIN_POINTS:
        return np.full(horizon, float(np.clip(observed.iloc[-1], 0, 1)))
    x = observed.index.to_numpy(dtype=float).reshape(-1, 1)
    z = logit(np.clip(observed.to_numpy(), EPS, 1 - EPS))
    term = s(0, n_splines=min(MAX_SPLINES, len(observed)), constraints="monotonic_inc" if monotone else None)
    with contextlib.redirect_stdout(io.StringIO()):  # pygam prints "did not converge" for some constrained fits
        gam = LinearGAM(term).gridsearch(x, z, lam=LAMBDAS, progress=False)
    future = x[-1, 0] + np.arange(1, horizon + 1, dtype=float)
    z_future = gam.predict(future.reshape(-1, 1))
    if anchor:
        z_future = z_future + z[-1] - gam.predict(x[-1:])[0]
    out = np.clip(expit(z_future), 0.0, 1.0)
    return np.maximum.accumulate(out) if monotone else out
