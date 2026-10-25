"""Historical replay with an explicit quote-freshness guard.

A common way to fabricate arbitrage in a backtest: when one pair's feed has a
gap, a replay engine keeps that pair's last cached price and compares it with
fresh prices on the other legs. The legs then describe two different moments,
and their product drifts away from 1 by roughly the market move during the
gap, which is far larger than any real cross-rate mispricing.

`replay` reproduces that failure on demand (pairs can be served stale) and
shows how a per-leg age check removes it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import RateTable, quotes_from_mids
from .graph import best_cycle_exhaustive, log_weights


@dataclass
class ReplayResult:
    days: int
    signals: int                  # days where the best cycle beat the threshold
    executable: int               # signals whose legs were all fresh
    best_bps: list = field(default_factory=list)


def stale_snapshot(table: RateTable, t: int, stale_prob: float, max_lag: int,
                   rng: np.random.Generator):
    """Mid matrix at day t where each unordered pair may be served from an older day.

    Returns (mid, age) with age[i, j] = how many days old the quote for (i, j) is.
    """
    n = len(table.currencies)
    mid = table.mid_matrix(t).copy()
    age = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < stale_prob:
                lag = int(rng.integers(1, max_lag + 1))
                old = table.mid_matrix(max(t - lag, 0))
                mid[i, j], mid[j, i] = old[i, j], old[j, i]
                age[i, j] = age[j, i] = lag
    return mid, age


def replay(table: RateTable, spread_bps: float, threshold_bps: float,
           stale_prob: float = 0.0, max_lag: int = 3, max_age: int | None = 0,
           seed: int = 0, length: int = 3) -> ReplayResult:
    """Scan every day for the best cycle.

    `threshold_bps` is the gross return a cycle must beat to count as a
    signal (set it to your round-trip fee estimate). With `max_age` set, a
    signal is executable only if every leg's quote is at most that many days old.
    """
    rng = np.random.default_rng(seed)
    res = ReplayResult(days=len(table.dates), signals=0, executable=0)
    for t in range(len(table.dates)):
        mid, age = stale_snapshot(table, t, stale_prob, max_lag, rng)
        bid, _ = quotes_from_mids(mid, spread_bps)
        w = log_weights(bid)
        cycle, cw = best_cycle_exhaustive(w, length)
        ret_bps = (np.exp(-cw) - 1.0) * 1e4
        res.best_bps.append(float(ret_bps))
        if ret_bps > threshold_bps:
            res.signals += 1
            legs = [(cycle[k], cycle[(k + 1) % length]) for k in range(length)]
            if max_age is None or all(age[i, j] <= max_age for i, j in legs):
                res.executable += 1
    return res
