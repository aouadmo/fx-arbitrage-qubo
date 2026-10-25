"""Reproduce every number in RESULTS.md.

    python experiments/run_all.py            # full run (~10 min on a laptop)
    python experiments/run_all.py --quick    # smaller instance counts

Outputs go to results/: results.json, RESULTS.md and two figures.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fxqubo import CURRENCIES  # noqa: E402
from fxqubo.data import DEFAULT_CACHE, download, load, quotes_from_mids, sha256  # noqa: E402
from fxqubo.graph import (bellman_ford_negative_cycle, best_cycle_exhaustive,  # noqa: E402
                          cycle_weight, enumerate_cycles, log_weights)
from fxqubo.qubo import (cycle_selection_qubo, decode_cycle_selection,  # noqa: E402
                         decode_edge_selection, edge_selection_qubo)
from fxqubo.replay import replay  # noqa: E402
from fxqubo.solvers import simulated_annealing, simulated_bifurcation  # noqa: E402

START, END = "2016-01-01", "2026-10-23"
SPREAD_BPS = 1.0          # assumed full bid-ask spread on every pair
FEE_BPS = 6.0             # assumed round-trip cost threshold for a 3-leg cycle


def exp_baseline(table):
    """Reference rates are mutually consistent, so every cycle is flat before spreads."""
    worst = 0.0
    for t in range(len(table.dates)):
        w = log_weights(table.mid_matrix(t))
        for c in enumerate_cycles(len(CURRENCIES), 3):
            worst = max(worst, abs(cycle_weight(w, c)))
    bid, _ = quotes_from_mids(table.mid_matrix(len(table.dates) - 1), SPREAD_BPS)
    _, best = best_cycle_exhaustive(log_weights(bid))
    return {"max_abs_cycle_log_sum_mid": worst,
            "best_cycle_return_bps_with_spread": float((np.exp(-best) - 1) * 1e4)}


def exp_stale(table):
    out = []
    for p in (0.0, 0.02, 0.05, 0.10, 0.20):
        r = replay(table, SPREAD_BPS, FEE_BPS, stale_prob=p, max_lag=3, max_age=0, seed=42)
        out.append({"stale_prob": p, "days": r.days, "signals": r.signals,
                    "executable_after_guard": r.executable,
                    "median_best_bps": float(np.median(r.best_bps)),
                    "max_best_bps": float(np.max(r.best_bps))})
    return out


def make_instance(table, rng, edge_bps):
    """A real ECB day with spreads, small quote noise, and one injected mispricing."""
    n = len(CURRENCIES)
    t = int(rng.integers(len(table.dates)))
    mid = table.mid_matrix(t)
    noise = np.exp(rng.normal(0, 0.3e-4, (n, n)))       # ~0.3 bp independent quote noise
    bid, _ = quotes_from_mids(mid * noise, SPREAD_BPS)
    i, j = rng.choice(n, 2, replace=False)
    bid[i, j] *= 1 + edge_bps / 1e4
    return log_weights(bid)


def exp_solvers(table, instances, sa_sweeps, sb_steps):
    rng = np.random.default_rng(2026)
    methods = ["bellman_ford", "A_readoff", "A_SA", "A_SB", "B_SA", "B_SB"]
    stats = {m: {"optimal": 0, "feasible": 0, "profitable": 0, "regret_bps": [], "seconds": 0.0}
             for m in methods}
    for k in range(instances):
        edge = float(rng.choice([3.0, 5.0, 10.0]))
        w = make_instance(table, rng, edge + 1.5)   # +1.5 bp offsets the three half-spreads
        _, best = best_cycle_exhaustive(w)

        def score(name, cycle, dt, any_length=False):
            stats[name]["seconds"] += dt
            if cycle is None:
                return
            cw = cycle_weight(w, tuple(cycle))
            stats[name]["feasible"] += 1
            stats[name]["profitable"] += cw < 0
            if not any_length:
                stats[name]["regret_bps"].append((cw - best) * 1e4)
                stats[name]["optimal"] += abs(cw - best) < 1e-12

        # Bellman-Ford answers a different question: "is there any negative
        # cycle, of any length?" It is scored on detection only.
        t0 = time.perf_counter(); bf = bellman_ford_negative_cycle(w)
        score("bellman_ford", bf, time.perf_counter() - t0, any_length=True)

        qa = cycle_selection_qubo(w)
        t0 = time.perf_counter(); c = qa.labels[int(np.argmin(np.diag(qa.Q)))]
        score("A_readoff", c, time.perf_counter() - t0)
        t0 = time.perf_counter(); x, _ = simulated_annealing(qa, sweeps=sa_sweeps, rng=np.random.default_rng(k))
        score("A_SA", decode_cycle_selection(qa, x), time.perf_counter() - t0)
        t0 = time.perf_counter(); x, _ = simulated_bifurcation(qa, steps=sb_steps, rng=np.random.default_rng(k))
        score("A_SB", decode_cycle_selection(qa, x), time.perf_counter() - t0)

        qb = edge_selection_qubo(w)
        t0 = time.perf_counter(); x, _ = simulated_annealing(qb, sweeps=sa_sweeps, rng=np.random.default_rng(k))
        score("B_SA", decode_edge_selection(qb, x), time.perf_counter() - t0)
        t0 = time.perf_counter(); x, _ = simulated_bifurcation(qb, steps=sb_steps, rng=np.random.default_rng(k))
        score("B_SB", decode_edge_selection(qb, x), time.perf_counter() - t0)
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{instances} instances", flush=True)
    for m in methods:
        s = stats[m]
        s["optimal_rate"] = s["optimal"] / instances
        s["profitable_rate"] = s["profitable"] / instances
        s["median_regret_bps"] = float(np.median(s["regret_bps"])) if s["regret_bps"] else None
        del s["regret_bps"]
        s["feasible_rate"] = s["feasible"] / instances
        s["ms_per_instance"] = 1e3 * s["seconds"] / instances
    return {"instances": instances, "sa_sweeps": sa_sweeps, "sb_steps": sb_steps,
            "variables": {"A": len(enumerate_cycles(len(CURRENCIES), 3)),
                          "B": len(CURRENCIES) * (len(CURRENCIES) - 1)},
            "methods": stats}


def figures(res, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st = res["stale_data"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    xs = np.arange(len(st))
    ax.bar(xs - 0.2, [r["signals"] for r in st], 0.4, label="signals above fee threshold")
    ax.bar(xs + 0.2, [r["executable_after_guard"] for r in st], 0.4, label="left after freshness guard")
    for x, r in zip(xs, st):
        ax.text(x + 0.2, 20, str(r["executable_after_guard"]), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(xs, [f"{int(r['stale_prob'] * 100)}%" for r in st])
    ax.set_xlabel("share of pairs served from a stale cache")
    ax.set_ylabel(f"days (of {st[0]['days']})")
    ax.set_title("Stale quotes fabricate arbitrage; a per-leg age check removes it")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "stale_data.png", dpi=160); plt.close(fig)

    m = res["solvers"]["methods"]
    names = ["bellman_ford", "A_readoff", "A_SA", "A_SB", "B_SA", "B_SB"]
    labels = ["Bellman-Ford", "A: read-off", "A: SA", "A: SB", "B: SA", "B: SB"]
    fig, ax = plt.subplots(figsize=(6, 3.4))
    xs = np.arange(len(names))
    ax.bar(xs - 0.27, [m[n]["feasible_rate"] for n in names], 0.27, label="valid cycle")
    ax.bar(xs, [m[n]["profitable_rate"] for n in names], 0.27, label="profitable cycle")
    ax.bar(xs + 0.27, [m[n]["optimal_rate"] if n != "bellman_ford" else 0 for n in names], 0.27,
           label="best 3-cycle")
    ax.set_xticks(xs, labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of instances")
    ax.set_title(f"Solver benchmark, {res['solvers']['instances']} instances, 8 currencies")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(outdir / "solvers.png", dpi=160); plt.close(fig)


def write_markdown(res, path):
    b, st, sv = res["baseline"], res["stale_data"], res["solvers"]
    lines = [
        "# Results",
        "",
        f"Data: ECB euro reference rates, {res['data']['first_day']} to {res['data']['last_day']}, "
        f"{res['data']['days']} business days, currencies {', '.join(CURRENCIES)}. "
        f"File SHA-256 `{res['data']['sha256'][:16]}...`. Spread assumption {SPREAD_BPS} bp, "
        f"fee threshold {FEE_BPS} bp per 3-leg cycle.",
        "",
        "## 1. Baseline",
        "",
        f"Largest |cycle log-sum| on mid rates over all days and all 112 three-cycles: "
        f"{b['max_abs_cycle_log_sum_mid']:.1e}. Reference rates are quoted against one currency, so "
        f"they are arbitrage-free by construction. With a {SPREAD_BPS} bp spread every cycle loses "
        f"{-b['best_cycle_return_bps_with_spread']:.2f} bp gross.",
        "",
        "## 2. Stale quotes",
        "",
        "| Pairs served stale | Days | Signals above threshold | Left after freshness guard | Max best-cycle return (bp) |",
        "|---|---|---|---|---|",
    ]
    for r in st:
        lines.append(f"| {r['stale_prob']:.0%} | {r['days']} | {r['signals']} | {r['executable_after_guard']} | {r['max_best_bps']:.1f} |")
    lines += ["", "![stale data](stale_data.png)", "",
              "## 3. Solver benchmark", "",
              f"{sv['instances']} instances built from random ECB days with {SPREAD_BPS} bp spreads, 0.3 bp quote "
              f"noise and one injected mispricing of 3, 5 or 10 bp net. Formulation A has {sv['variables']['A']} "
              f"variables, B has {sv['variables']['B']}. SA: {sv['sa_sweeps']} sweeps x 2 restarts. "
              f"SB: {sv['sb_steps']} steps x 64 replicas. Penalties and schedules were tuned once on "
              f"separate instances (seed 5) and then frozen.", "",
              "| Method | Valid cycle | Profitable | Best 3-cycle | Median regret (bp) | ms per instance |",
              "|---|---|---|---|---|---|"]
    label = {"bellman_ford": "Bellman-Ford (classical)", "A_readoff": "A: read the diagonal",
             "A_SA": "A: simulated annealing", "A_SB": "A: simulated bifurcation",
             "B_SA": "B: simulated annealing", "B_SB": "B: simulated bifurcation"}
    for k, v in sv["methods"].items():
        reg = "n/a" if v["median_regret_bps"] is None else f"{v['median_regret_bps']:.2f}"
        opt = "n/a (any length)" if k == "bellman_ford" else f"{v['optimal_rate']:.0%}"
        lines.append(f"| {label[k]} | {v['feasible_rate']:.0%} | {v['profitable_rate']:.0%} | {opt} | {reg} | {v['ms_per_instance']:.1f} |")
    lines += ["", "![solvers](solvers.png)", ""]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    outdir = ROOT / "results"
    outdir.mkdir(exist_ok=True)
    path = download(DEFAULT_CACHE)
    table = load(path, CURRENCIES, START, END)
    res = {"data": {"source": "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-hist.zip",
                    "sha256": sha256(path), "first_day": table.dates[0],
                    "last_day": table.dates[-1], "days": len(table.dates)}}
    print("baseline ..."); res["baseline"] = exp_baseline(table)
    print("stale data ..."); res["stale_data"] = exp_stale(table)
    print("solvers ...")
    res["solvers"] = exp_solvers(table, 30 if args.quick else 100, 1000, 6000)
    (outdir / "results.json").write_text(json.dumps(res, indent=2))
    figures(res, outdir)
    write_markdown(res, outdir / "RESULTS.md")
    print((outdir / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
