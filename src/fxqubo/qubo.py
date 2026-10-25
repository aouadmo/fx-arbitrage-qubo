"""Two QUBO formulations of "find the most profitable cycle".

A QUBO is: minimise E(x) = x^T Q x + offset over binary x, with Q upper
triangular (linear terms on the diagonal, because x_i^2 = x_i).

Formulation A, cycle selection (one-hot):
    one binary variable per pre-enumerated cycle; objective = cycle weight;
    penalty P * (sum x - 1)^2 forces exactly one choice. Every feasible
    assignment is a single cycle, so the ground state can be read off the
    diagonal: the QUBO adds no search power over plain enumeration.

Formulation B, edge selection:
    one binary variable per directed edge; objective = sum of chosen edge
    weights; penalties enforce flow conservation at each node, at most one
    outgoing edge per node, no 2-cycles, and exactly K chosen edges. For
    K = 3 the feasible set is exactly the directed 3-cycles. Here the solver
    has to search, which is where heuristic solvers can be compared.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .graph import enumerate_cycles, gauge_fix


@dataclass
class Qubo:
    Q: np.ndarray            # upper-triangular coefficient matrix
    offset: float
    labels: list             # what each variable means
    scale: float = 1.0       # energy = scale * (cycle weight) on feasible points
    penalty: float = 1.0     # constraint weight used to build Q

    def energy(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(x @ self.Q @ x + self.offset)

    @property
    def n(self) -> int:
        return self.Q.shape[0]


class _Builder:
    """Accumulates linear/quadratic terms into an upper-triangular Q."""

    def __init__(self, n: int):
        self.Q = np.zeros((n, n))
        self.offset = 0.0

    def lin(self, i, c):
        self.Q[i, i] += c

    def quad(self, i, j, c):
        if i == j:
            self.Q[i, i] += c
        else:
            a, b = (i, j) if i < j else (j, i)
            self.Q[a, b] += c

    def square(self, coeffs: dict, const: float, weight: float):
        """Add weight * (sum_i coeffs[i] x_i + const)^2."""
        items = list(coeffs.items())
        for idx, (i, a) in enumerate(items):
            self.lin(i, weight * (a * a + 2 * a * const))
            for j, b in items[idx + 1:]:
                self.quad(i, j, weight * 2 * a * b)
        self.offset += weight * const * const


def cycle_selection_qubo(w: np.ndarray, length: int = 3, penalty: float | None = None,
                         normalize: bool = True) -> Qubo:
    """Formulation A.

    With `normalize`, cycle weights are divided by their largest magnitude so
    the objective and the one-hot penalty live on the same scale. Without it,
    weights of order 1e-4 sit under a penalty of order 1 and a heuristic
    solver cannot tell the cycles apart.
    """
    cycles = enumerate_cycles(w.shape[0], length)
    weights = np.array([sum(w[c[k], c[(k + 1) % length]] for k in range(length)) for c in cycles])
    scale = 1.0 / np.abs(weights).max() if normalize and np.abs(weights).max() > 0 else 1.0
    weights = weights * scale
    if penalty is None:
        # Tuned on the benchmark: just above the largest objective swing. A much
        # larger penalty freezes single-flip annealing in the first one-hot
        # state it reaches; a smaller one lets two-hot states win.
        penalty = 2.5 if normalize else 2 * (np.abs(weights).max() + 1.0)
    b = _Builder(len(cycles))
    for i, cw in enumerate(weights):
        b.lin(i, cw)
    b.square({i: 1.0 for i in range(len(cycles))}, -1.0, penalty)
    return Qubo(b.Q, b.offset, cycles, scale, penalty)


def edge_selection_qubo(w: np.ndarray, length: int = 3, penalty: float | None = None,
                        normalize: bool = True) -> Qubo:
    """Formulation B.

    With `normalize`, weights are first gauge-fixed (cycle weights unchanged,
    price levels removed) and then divided by their largest magnitude.
    """
    n = w.shape[0]
    scale = 1.0
    if normalize:
        w = gauge_fix(w)
        m = np.abs(w).max()
        scale = 1.0 / m if m > 0 else 1.0
        w = w * scale
    edges = [(i, j) for i in range(n) for j in range(n) if i != j]
    idx = {e: k for k, e in enumerate(edges)}
    max_w = max(abs(w[i, j]) for i, j in edges)
    if penalty is None:
        penalty = 3.0 if normalize else 2 * (length * max_w + 1.0)
    b = _Builder(len(edges))
    for (i, j), k in idx.items():
        b.lin(k, w[i, j])
    # flow conservation: out(v) - in(v) = 0
    for v in range(n):
        coeffs = {}
        for u in range(n):
            if u != v:
                coeffs[idx[(v, u)]] = coeffs.get(idx[(v, u)], 0) + 1.0
                coeffs[idx[(u, v)]] = coeffs.get(idx[(u, v)], 0) - 1.0
        b.square(coeffs, 0.0, penalty)
    # at most one outgoing edge per node: penalise every pair of outgoing edges
    for v in range(n):
        outs = [idx[(v, u)] for u in range(n) if u != v]
        for a in range(len(outs)):
            for c in range(a + 1, len(outs)):
                b.quad(outs[a], outs[c], penalty)
    # no 2-cycles
    for i in range(n):
        for j in range(i + 1, n):
            b.quad(idx[(i, j)], idx[(j, i)], penalty)
    # exactly `length` edges
    b.square({k: 1.0 for k in range(len(edges))}, -float(length), penalty)
    return Qubo(b.Q, b.offset, edges, scale, penalty)


def decode_cycle_selection(q: Qubo, x) -> tuple | None:
    chosen = [q.labels[i] for i, v in enumerate(x) if v]
    return chosen[0] if len(chosen) == 1 else None


def decode_edge_selection(q: Qubo, x, length: int = 3) -> tuple | None:
    """Return the cycle (canonical rotation) if x encodes one valid cycle, else None."""
    chosen = [q.labels[i] for i, v in enumerate(x) if v]
    if len(chosen) != length:
        return None
    nxt = dict(chosen)
    if len(nxt) != length:
        return None
    start = min(nxt)
    cycle, v = [start], nxt.get(start)
    while v is not None and v != start and len(cycle) <= length:
        cycle.append(v)
        v = nxt.get(v)
    if v != start or len(cycle) != length:
        return None
    return tuple(cycle)
