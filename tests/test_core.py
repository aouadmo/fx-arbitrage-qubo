import itertools

import numpy as np
import pytest

from fxqubo.data import quotes_from_mids
from fxqubo.graph import (bellman_ford_negative_cycle, best_cycle_exhaustive,
                          cycle_weight, enumerate_cycles, log_weights)
from fxqubo.qubo import (cycle_selection_qubo, decode_cycle_selection,
                         decode_edge_selection, edge_selection_qubo)
from fxqubo.replay import replay
from fxqubo.data import RateTable
from fxqubo.solvers import qubo_to_ising, simulated_annealing, simulated_bifurcation


def consistent_mids(n=5, seed=1):
    """Cross rates derived from one set of per-EUR rates: arbitrage-free by construction."""
    r = np.exp(np.random.default_rng(seed).normal(0, 1, n))
    r[0] = 1.0
    return r[None, :] / r[:, None]


def with_arbitrage(mid, cycle, edge_bps):
    """Raise one leg so that `cycle` returns about `edge_bps` before spreads."""
    m = mid.copy()
    i, j = cycle[0], cycle[1]
    m[i, j] *= 1 + edge_bps / 1e4
    return m


def test_enumeration_counts_each_cycle_once():
    for n in (4, 5, 8):
        cyc = enumerate_cycles(n, 3)
        assert len(cyc) == n * (n - 1) * (n - 2) // 3
        assert len({frozenset(itertools.combinations(c + (c[0],), 2)) for c in cyc}) == len(cyc)


def test_consistent_rates_have_zero_cycles():
    w = log_weights(consistent_mids())
    for c in enumerate_cycles(5, 3):
        assert abs(cycle_weight(w, c)) < 1e-12
    assert bellman_ford_negative_cycle(w, tol=1e-12) is None


def test_spread_makes_every_cycle_lose():
    bid, _ = quotes_from_mids(consistent_mids(), spread_bps=2.0)
    w = log_weights(bid)
    _, best = best_cycle_exhaustive(w)
    assert best > 0  # every round trip pays three half-spreads


def test_injected_arbitrage_is_found_by_every_method():
    mid = with_arbitrage(consistent_mids(6), (1, 3, 4), 25.0)
    bid, _ = quotes_from_mids(mid, 1.0)
    w = log_weights(bid)
    # Raising one leg makes every 3-cycle through edge 1->3 equally profitable,
    # so check the optimal weight and the edge, not one particular tie.
    target, tw = best_cycle_exhaustive(w)
    legs = {(target[k], target[(k + 1) % 3]) for k in range(3)}
    assert (1, 3) in legs and tw < 0
    bf = bellman_ford_negative_cycle(w)
    assert bf is not None and cycle_weight(w, tuple(bf)) < 0

    qa = cycle_selection_qubo(w)
    # Formulation A can be solved exactly by reading the diagonal ...
    assert cycle_weight(w, qa.labels[int(np.argmin(np.diag(qa.Q)))]) == pytest.approx(tw)
    # ... while single-flip annealing reliably lands on a feasible one-hot point
    # but not always the best one: leaving a one-hot state needs two flips
    # through a penalty barrier. The benchmark measures how often it succeeds.
    xa, ea = simulated_annealing(qa, sweeps=500, restarts=2)
    assert decode_cycle_selection(qa, xa) is not None

    qb = edge_selection_qubo(w)
    xb, eb = simulated_bifurcation(qb)
    assert cycle_weight(w, decode_edge_selection(qb, xb)) == pytest.approx(tw)
    assert eb == pytest.approx(tw * qb.scale, abs=1e-9)


def test_qubo_energy_equals_cycle_weight_on_feasible_points():
    w = log_weights(quotes_from_mids(consistent_mids(5, 3), 1.5)[0])
    qa = cycle_selection_qubo(w, normalize=False)
    for k, c in enumerate(qa.labels):
        x = np.zeros(qa.n); x[k] = 1
        assert qa.energy(x) == pytest.approx(cycle_weight(w, c))
    qb = edge_selection_qubo(w, normalize=False)
    for c in enumerate_cycles(5, 3):
        x = np.zeros(qb.n)
        for k in range(3):
            x[qb.labels.index((c[k], c[(k + 1) % 3]))] = 1
        assert qb.energy(x) == pytest.approx(cycle_weight(w, c))
        assert decode_edge_selection(qb, x) == c


def test_infeasible_points_cost_more_than_any_cycle():
    w = log_weights(quotes_from_mids(consistent_mids(5, 4), 1.0)[0])
    qb = edge_selection_qubo(w, normalize=False)
    worst_cycle = max(cycle_weight(w, c) for c in enumerate_cycles(5, 3))
    rng = np.random.default_rng(0)
    for _ in range(300):
        x = rng.integers(0, 2, qb.n)
        if decode_edge_selection(qb, x) is None:
            assert qb.energy(x) > worst_cycle


def test_gauge_fix_preserves_every_cycle():
    from fxqubo.graph import gauge_fix
    w = log_weights(quotes_from_mids(with_arbitrage(consistent_mids(6, 2), (0, 2, 5), 7.0), 1.0)[0])
    g = gauge_fix(w)
    assert np.abs(g).max() < 1e-2 < np.abs(w).max()
    for length in (3, 4):
        for c in enumerate_cycles(6, length):
            assert cycle_weight(g, c) == pytest.approx(cycle_weight(w, c), abs=1e-12)


def test_ising_mapping_preserves_energy():
    w = log_weights(quotes_from_mids(consistent_mids(4, 5), 1.0)[0])
    q = edge_selection_qubo(w)
    J, h, c = qubo_to_ising(q)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.integers(0, 2, q.n)
        s = 2 * x - 1
        assert s @ J @ s + h @ s + c == pytest.approx(q.energy(x))


def test_stale_quotes_fabricate_signals_and_guard_removes_them():
    rng = np.random.default_rng(7)
    days, n = 60, 6
    # random-walk per-EUR rates with ~0.5% daily moves: consistent on every day
    per_eur = np.exp(np.cumsum(rng.normal(0, 0.005, (days, n)), axis=0))
    per_eur[:, 0] = 1.0
    table = RateTable([f"d{t:03d}" for t in range(days)], [f"C{k}" for k in range(n)], per_eur)
    clean = replay(table, spread_bps=1.0, threshold_bps=6.0, stale_prob=0.0)
    assert clean.signals == 0
    stale = replay(table, spread_bps=1.0, threshold_bps=6.0, stale_prob=0.3, max_age=0)
    assert stale.signals > 10
    assert stale.executable == 0
