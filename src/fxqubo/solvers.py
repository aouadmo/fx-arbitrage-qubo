"""Heuristic QUBO solvers in plain NumPy.

- `simulated_annealing`: single-bit-flip Metropolis with a geometric schedule.
- `simulated_bifurcation`: ballistic simulated bifurcation (bSB), the
  algorithm family behind Toshiba's SQBM+ (Goto et al., Science Advances
  7, eabe7953, 2021). This is an independent textbook implementation, not
  Toshiba's product.

Both return the best binary vector found and its energy.
"""

from __future__ import annotations

import numpy as np

from .qubo import Qubo


def _sym(Q: np.ndarray) -> np.ndarray:
    """Symmetric matrix S with x^T Q x = x^T S x."""
    return (Q + Q.T) / 2.0


def simulated_annealing(q: Qubo, sweeps: int = 1000, restarts: int = 2,
                        t_start: float | None = None, t_end: float = 1e-3,
                        rng: np.random.Generator | None = None):
    rng = rng or np.random.default_rng(0)
    S = _sym(q.Q)
    n = q.n
    diag = np.diag(S).copy()
    if t_start is None:
        # Start hot enough to cross the constraint barrier, not hotter: time
        # spent far above the penalty scale is wasted sweeps.
        t_start = q.penalty
    temps = np.geomspace(t_start, t_end, sweeps)
    best_x, best_e = None, np.inf
    for _ in range(restarts):
        x = rng.integers(0, 2, n).astype(float)
        field = S @ x            # field[i] = sum_j S_ij x_j
        e = float(x @ S @ x)
        for T in temps:
            for i in rng.permutation(n):
                # Energy change of flipping bit i
                s = 1.0 - 2.0 * x[i]                     # +1 if 0->1, -1 if 1->0
                de = s * (2.0 * field[i] - 2.0 * diag[i] * x[i] + diag[i])
                if de <= 0 or rng.random() < np.exp(-de / T):
                    x[i] += s
                    field += s * S[:, i]
                    e += de
        if e < best_e:
            best_e, best_x = e, x.copy()
    return best_x.astype(int), best_e + q.offset


def qubo_to_ising(q: Qubo):
    """Map x = (1 + s) / 2. Returns (J, h, c) with E = s^T J s + h^T s + c, J zero-diagonal."""
    S = _sym(q.Q)
    J = S / 4.0
    np.fill_diagonal(J, 0.0)
    h = S.sum(axis=1) / 2.0
    c = S.sum() / 4.0 + np.trace(S) / 4.0 + q.offset
    return J, h, c


def simulated_bifurcation(q: Qubo, steps: int = 6000, replicas: int = 64,
                          dt: float = 0.5, rng: np.random.Generator | None = None):
    """Ballistic SB minimising the Ising energy of `q`.

    Positions x in [-1, 1] evolve under an adiabatically raised pump a(t);
    walls at |x| = 1 are inelastic (x clipped, momentum zeroed). The final
    spins are sign(x). All replicas run in parallel as matrix columns.
    """
    rng = rng or np.random.default_rng(0)
    J, h, c = qubo_to_ising(q)
    n = q.n
    a0 = 1.0
    scale = np.sqrt((J ** 2).sum() / max(n * (n - 1), 1)) + np.abs(h).mean() / np.sqrt(n) + 1e-12
    c0 = 0.5 / (scale * np.sqrt(n))
    x = rng.uniform(-0.1, 0.1, (n, replicas))
    y = rng.uniform(-0.1, 0.1, (n, replicas))
    for k in range(steps):
        a = a0 * k / steps
        # force = -dE/ds for E = s^T J s + h^T s
        force = -(2.0 * J @ x + h[:, None])
        y += (-(a0 - a) * x + c0 * force) * dt
        x += a0 * y * dt
        wall = np.abs(x) > 1.0
        x[wall] = np.sign(x[wall])
        y[wall] = 0.0
    s = np.where(x >= 0, 1.0, -1.0)
    energies = np.einsum("ir,ij,jr->r", s, J, s) + h @ s + c
    r = int(np.argmin(energies))
    return ((s[:, r] + 1) / 2).astype(int), float(energies[r])
