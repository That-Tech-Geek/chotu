"""Reproducible Pareto sweep + leakage-safe experiments for the reviewer's
experiment matrix. Run: `python examples/pareto_sweep.py`."""
from __future__ import annotations

from economic_engine.simulation.pareto import (
    baseline_points,
    pareto_rows,
    sweep_lambdas,
)
from economic_engine.simulation.prior_leakage import run_leakage_safe_experiment


def main():
    lambdas = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0]
    print("Pareto sweep (deal_rate vs surplus):")
    rows = sweep_lambdas(lambdas=lambdas, n_eval=30)
    for r in rows:
        print(
            f"  l={r['lambda']:5.2f}  deal_rate={r['deal_rate']:.3f}  "
            f"surplus={r['avg_surplus']:.3f}",
        )
    frontier = pareto_rows(rows)
    print(f"\nEfficient frontier ({len(frontier)}/{len(rows)} points):")
    for r in frontier:
        print(
            f"  l={r['lambda']:5.2f}  deal_rate={r['deal_rate']:.3f}  "
            f"surplus={r['avg_surplus']:.3f}",
        )
    print("\nBaselines:")
    for name, pts in baseline_points(n_eval=30).items():
        print(
            f"  {name:12} deal_rate={pts['deal_rate']:.3f}  "
            f"surplus={pts['avg_surplus']:.3f}",
        )
    print("\nLeakage-safe prior audit (data at t only):")
    res = run_leakage_safe_experiment(n=40)
    for k, v in res.items():
        print(f"  {k}: {v:.3f}")
    print("\nActionable read:")
    meta = {
        0:   "risk-neutral (max surplus)",
        1:   "default Kelly-shaped",
        2:   "robust CVaR",
        5:   "ultra-safe",
    }
    for r in rows:
        lam = r["lambda"]
        tag = meta.get(int(lam), "tuned")
        print(f"  lambda={lam}: {tag}")


if __name__ == "__main__":
    main()
