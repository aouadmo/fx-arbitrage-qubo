# Results

Data: ECB euro reference rates, 2016-01-04 to 2026-10-23, 2768 business days, currencies EUR, USD, JPY, GBP, CHF, AUD, CAD, NZD. File SHA-256 `d0c1083db615c0c6...`. Spread assumption 1.0 bp, fee threshold 6.0 bp per 3-leg cycle.

## 1. Baseline

Largest |cycle log-sum| on mid rates over all days and all 112 three-cycles: 1.1e-15. Reference rates are quoted against one currency, so they are arbitrage-free by construction. With a 1.0 bp spread every cycle loses 1.50 bp gross.

## 2. Stale quotes

| Pairs served stale | Days | Signals above threshold | Left after freshness guard | Max best-cycle return (bp) |
|---|---|---|---|---|
| 0% | 2768 | 0 | 0 | -1.5 |
| 2% | 2768 | 1087 | 0 | 566.6 |
| 5% | 2768 | 2013 | 0 | 474.3 |
| 10% | 2768 | 2547 | 0 | 706.2 |
| 20% | 2768 | 2755 | 0 | 765.8 |

![stale data](stale_data.png)

## 3. Solver benchmark

100 instances built from random ECB days with 1.0 bp spreads, 0.3 bp quote noise and one injected mispricing of 3, 5 or 10 bp net. Formulation A has 112 variables, B has 56. SA: 1000 sweeps x 2 restarts. SB: 6000 steps x 64 replicas. Penalties and schedules were tuned once on separate instances (seed 5) and then frozen.

| Method | Valid cycle | Profitable | Best 3-cycle | Median regret (bp) | ms per instance |
|---|---|---|---|---|---|
| Bellman-Ford (classical) | 100% | 100% | n/a (any length) | n/a | 0.2 |
| A: read the diagonal | 100% | 100% | 100% | 0.00 | 0.0 |
| A: simulated annealing | 100% | 90% | 30% | 0.25 | 343.9 |
| A: simulated bifurcation | 83% | 83% | 63% | 0.00 | 491.2 |
| B: simulated annealing | 89% | 15% | 1% | 6.50 | 182.6 |
| B: simulated bifurcation | 100% | 100% | 100% | 0.00 | 219.5 |

![solvers](solvers.png)
