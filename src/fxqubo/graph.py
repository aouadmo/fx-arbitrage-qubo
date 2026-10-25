"""Currency graph and classical cycle detection.

Edge i -> j means "convert currency i into currency j". Converting one unit of
i yields rate[i, j] units of j, where rate is the price we actually get: the
bid of pair i/j. With weight w[i, j] = -ln(rate[i, j]), a closed cycle is
profitable exactly when the sum of its weights is negative.
"""

from __future__ import annotations

from itertools import permutations

import numpy as np


def log_weights(rates: np.ndarray) -> np.ndarray:
    """w[i, j] = -ln(rate[i, j]); diagonal set to 0."""
    with np.errstate(divide="ignore"):
        w = -np.log(rates)
    np.fill_diagonal(w, 0.0)
    return w


def gauge_fix(w: np.ndarray) -> np.ndarray:
    """Remove the part of the weights that cancels around every cycle.

    Adding a node potential, w'[i, j] = w[i, j] + phi[i] - phi[j], leaves every
    cycle weight unchanged (the potentials telescope). Choosing phi by least
    squares strips out the price levels (JPY vs GBP, say) and leaves only the
    spreads and mispricings, which are four to five orders of magnitude smaller.
    This is the same freedom as a gauge choice for a lattice field.
    """
    n = w.shape[0]
    rows, rhs = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                r = np.zeros(n); r[i], r[j] = 1.0, -1.0
                rows.append(r); rhs.append(-w[i, j])
    phi, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    out = w + phi[:, None] - phi[None, :]
    np.fill_diagonal(out, 0.0)
    return out


def cycle_weight(w: np.ndarray, cycle) -> float:
    """Sum of edge weights around a closed cycle (tuple of node indices)."""
    k = len(cycle)
    return float(sum(w[cycle[i], cycle[(i + 1) % k]] for i in range(k)))


def cycle_return_bps(w: np.ndarray, cycle) -> float:
    """Gross return of trading once around the cycle, in basis points."""
    return float((np.exp(-cycle_weight(w, cycle)) - 1.0) * 1e4)


def enumerate_cycles(n: int, length: int = 3) -> list[tuple[int, ...]]:
    """All directed simple cycles of a given length on a complete graph,
    each listed once (canonical rotation: smallest node first).

    Enumerating raw permutations would list every cycle `length` times,
    once per rotation; this keeps one representative.
    """
    out = []
    for perm in permutations(range(n), length):
        if perm[0] == min(perm):
            out.append(perm)
    return out


def best_cycle_exhaustive(w: np.ndarray, length: int = 3):
    """Exact answer by enumeration: (cycle, weight) with the minimum weight."""
    best, best_w = None, np.inf
    for c in enumerate_cycles(w.shape[0], length):
        cw = cycle_weight(w, c)
        if cw < best_w:
            best, best_w = c, cw
    return best, best_w


def bellman_ford_negative_cycle(w: np.ndarray, tol: float = 0.0):
    """Return one negative cycle (list of nodes) if any exists, else None.

    Standard Bellman-Ford from a virtual source connected to every node, so
    negative cycles anywhere in the graph are found. `tol` lets tiny negative
    sums from floating-point noise be ignored.
    """
    n = w.shape[0]
    dist = np.zeros(n)
    pred = np.full(n, -1)
    last = -1
    for _ in range(n):
        last = -1
        for i in range(n):
            for j in range(n):
                if i != j and dist[i] + w[i, j] < dist[j] - tol:
                    dist[j] = dist[i] + w[i, j]
                    pred[j] = i
                    last = j
        if last == -1:
            return None
    # Walk back n steps to land inside the cycle, then extract it.
    v = last
    for _ in range(n):
        v = pred[v]
    cycle, u = [v], pred[v]
    while u != v:
        cycle.append(u)
        u = pred[u]
    cycle.reverse()
    return cycle
