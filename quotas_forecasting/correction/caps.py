"""Per country-segment year-over-year caps, computed from historical observations only (never from forecast values or truth).

Adaptive mode: ``cap = clamp(k * max(own_q, peer_q), floor, ceiling)``, where ``own_q`` is the ``quantile`` of the series' historical
absolute year-over-year changes and ``peer_q`` the same quantile pooled over the other countries' series of that segment. A series or
peer pool with fewer than MIN_CHANGES changes is not used; when neither is usable the key gets ``fallback``. Fixed mode gives every key
``fallback`` (the scalar ``max_yoy_change``). The cap guards against unattended large moves; it is not a statistical test.
"""

import numpy as np

from quotas_forecasting.correction.types import ForecastTable

MIN_CHANGES = 3


def _historical_changes(baseline: ForecastTable) -> dict[tuple[str, str], np.ndarray]:
    """|value(y) - value(y-1)| per (country, segment) over consecutive historical years where both values are present."""
    history = baseline[~baseline["is_forecast"]].sort_values(["country", "segment", "year"])
    changes = {}
    for (country, segment), rows in history.groupby(["country", "segment"]):
        consecutive = np.diff(rows["year"].to_numpy()) == 1
        diffs = np.abs(np.diff(rows["value"].to_numpy(dtype=float)))[consecutive]
        changes[(str(country), str(segment))] = diffs[~np.isnan(diffs)]
    return changes


def compute_yoy_caps(baseline: ForecastTable, settings: dict, fallback: float) -> dict[tuple[str, str], float]:
    """The yoy cap for every (country, segment) of ``baseline``; ``settings`` is the ``correction.yoy_cap`` config block."""
    keys = sorted({(str(c), str(s)) for c, s in zip(baseline["country"], baseline["segment"])})
    if settings["mode"] == "fixed":
        return {key: fallback for key in keys}
    if settings["mode"] != "adaptive":
        raise ValueError(f"Unknown correction.yoy_cap.mode: {settings['mode']!r}. Must be 'adaptive' or 'fixed'")
    changes = _historical_changes(baseline)
    caps = {}
    for country, segment in keys:
        own = changes.get((country, segment), np.empty(0))
        peers = np.concatenate([np.empty(0)] + [d for (c, s), d in changes.items() if s == segment and c != country])
        usable = [np.quantile(d, settings["quantile"]) for d in (own, peers) if len(d) >= MIN_CHANGES]
        caps[(country, segment)] = float(np.clip(settings["k"] * max(usable), settings["floor"], settings["ceiling"])) if usable else fallback
    return caps
