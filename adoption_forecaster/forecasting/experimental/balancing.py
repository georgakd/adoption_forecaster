"""Entropy and kernel balancing used as DONOR WEIGHTING for proxy (analog) forecasts.

Both are survey-weighting / covariate-balancing tools: they reweight a set of units so that the weighted units match target
moments. They are not forecasters. Here the "units" are donor windows: a donor series observed at an anchor year t, described by
its level and recent slope at t, together with its known path over the next ``horizon`` years. The weights make the weighted donor
windows resemble the target's current state; the forecast is the weighted donor path shifted to the target's last observed level
(like a synthetic control built from analog moments in other countries' histories). Weighting survey respondents to quotas is out
of scope for this repo.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize
from scipy.spatial.distance import cdist, pdist
from scipy.special import logsumexp, softmax

FEATURE_LAG = 2  # slope feature = (y_t - y_{t-FEATURE_LAG}) / FEATURE_LAG
BALANCE_TOL = 1e-6
KERNEL_ITERATIONS = 1000


class InfeasibleBalanceError(ValueError):
    """The target's feature means cannot be reproduced by non-negative donor weights (target outside the donors' convex hull)."""


def _standardize(donors: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean, std = donors.mean(axis=0), donors.std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return (donors - mean) / std, (target - mean) / std


def entropy_balance_weights(donors: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Weights w >= 0, sum(w) = 1, w @ donors = target, of minimal KL divergence to uniform (Hainmueller 2012), via the dual.

    The primal solution is w_i ∝ exp(lambda' x_i); lambda minimizes the convex dual log-sum-exp(D lambda), D = donors - target.
    Raises InfeasibleBalanceError when the target is outside the donors' convex hull (checked by a linear program first).
    """
    donors, target = np.asarray(donors, dtype=float), np.asarray(target, dtype=float)
    constant = donors.std(axis=0) == 0
    if np.any(np.abs(donors[0, constant] - target[constant]) > BALANCE_TOL):
        raise InfeasibleBalanceError("target differs from a feature that is constant across donors")
    x, t = _standardize(donors[:, ~constant], target[~constant])
    n = len(x)
    feasible = linprog(np.zeros(n), A_eq=np.vstack([x.T, np.ones(n)]), b_eq=np.append(t, 1.0), bounds=(0, None), method="highs")
    if feasible.status != 0:
        raise InfeasibleBalanceError("target lies outside the donors' convex hull")
    d = x - t

    def dual(lam):
        return logsumexp(d @ lam)

    def grad(lam):
        return d.T @ softmax(d @ lam)

    def hess(lam):
        p = softmax(d @ lam)
        m = d.T @ p
        return (d.T * p) @ d - np.outer(m, m)

    res = minimize(dual, np.zeros(d.shape[1]), jac=grad, hess=hess, method="trust-exact", options={"gtol": 1e-12, "maxiter": 1000})
    w = softmax(d @ res.x)
    if np.max(np.abs(w @ donors - target)) > BALANCE_TOL:
        raise InfeasibleBalanceError("target is on the edge of the donors' convex hull; exact balance needs zero weights")
    return w


def _project_to_simplex(v: np.ndarray) -> np.ndarray:
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - 1
    rho = np.nonzero(u - css / np.arange(1, len(v) + 1) > 0)[0][-1]
    return np.maximum(v - css[rho] / (rho + 1), 0.0)


def kernel_balance_weights(donors: np.ndarray, target: np.ndarray, length_scale: float | None = None) -> np.ndarray:
    """Simplex weights minimizing the RBF-kernel mean discrepancy ||sum_i w_i phi(x_i) - phi(target)||^2 (cf. Hazlett 2020).

    Features are standardized on the donors; ``length_scale=None`` uses the median pairwise donor distance. Always feasible: a
    target outside the hull gets weight on its nearest donors. Solved by accelerated projected gradient (deterministic).
    """
    x, t = _standardize(np.asarray(donors, dtype=float), np.asarray(target, dtype=float))
    n = len(x)
    if n == 1:
        return np.ones(1)
    if length_scale is None:
        distances = pdist(x)
        length_scale = float(np.median(distances)) if np.median(distances) > 0 else 1.0
    gram = np.exp(-cdist(x, x, "sqeuclidean") / (2 * length_scale**2))
    k_target = np.exp(-cdist(x, t.reshape(1, -1), "sqeuclidean")[:, 0] / (2 * length_scale**2))
    step = 1.0 / (2 * np.linalg.eigvalsh(gram)[-1])
    w = y = np.full(n, 1.0 / n)
    momentum = 1.0
    for _ in range(KERNEL_ITERATIONS):
        w_next = _project_to_simplex(y - step * 2 * (gram @ y - k_target))
        momentum_next = (1 + np.sqrt(1 + 4 * momentum**2)) / 2
        y = w_next + (momentum - 1) / momentum_next * (w_next - w)
        converged = np.max(np.abs(w_next - w)) < 1e-8
        w, momentum = w_next, momentum_next
        if converged:
            break
    return w / w.sum()


def _features(values: np.ndarray, pos: int) -> np.ndarray:
    return np.array([values[pos], (values[pos] - values[pos - FEATURE_LAG]) / FEATURE_LAG])


def _donor_windows(donors: pd.DataFrame, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Features at every anchor year with FEATURE_LAG years of history and ``horizon`` known years after, plus the paths."""
    features, paths = [], []
    for _, row in donors.iterrows():
        row = row.astype(float)
        if row.notna().sum() < 2:
            continue
        values = row.loc[row.first_valid_index() : row.last_valid_index()].interpolate().to_numpy()
        for pos in range(FEATURE_LAG, len(values) - horizon):
            features.append(_features(values, pos))
            paths.append(values[pos + 1 : pos + horizon + 1] - values[pos])
    return np.array(features), np.array(paths)


def proxy_forecast(target_history: pd.Series, donors: pd.DataFrame, method: str, horizon: int = 2) -> pd.Series:
    """Weighted donor paths shifted to the target's last observed level, for the ``horizon`` years after it, clipped to [0, 1].

    ``method`` is "entropy" or "kernel". Entropy balancing is exact but needs the target inside the donors' hull; when it is not,
    the kernel weights are used instead and ``result.attrs["weights"]`` records which weights were used.
    """
    if method not in ("entropy", "kernel"):
        raise ValueError(f"Unknown balancing method: {method}. Must be 'entropy' or 'kernel'")
    history = target_history.astype(float)
    history = history.loc[history.first_valid_index() : history.last_valid_index()].interpolate()
    last_year, last_value = int(history.index[-1]), float(history.iloc[-1])
    index = pd.Index(range(last_year + 1, last_year + horizon + 1))
    features, paths = _donor_windows(donors, horizon)
    if len(history) <= FEATURE_LAG or len(features) == 0:
        result = pd.Series(np.full(horizon, np.clip(last_value, 0, 1)), index=index)
        result.attrs["weights"] = "none"
        return result
    target = _features(history.to_numpy(), len(history) - 1)
    used = method
    if method == "entropy":
        try:
            w = entropy_balance_weights(features, target)
        except InfeasibleBalanceError:
            w, used = kernel_balance_weights(features, target), "kernel"
    else:
        w = kernel_balance_weights(features, target)
    result = pd.Series(np.clip(last_value + w @ paths, 0.0, 1.0), index=index)
    result.attrs["weights"] = used
    return result
