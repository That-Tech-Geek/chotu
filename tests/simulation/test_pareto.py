import numpy as np

from economic_engine.simulation.ablation import run_ablation
from economic_engine.simulation.calibration import calibration_experiment
from economic_engine.simulation.frontier_benchmark import run_frontier_benchmark
from economic_engine.simulation.pareto import baseline_points, pareto_rows, sweep_lambdas
from economic_engine.simulation.prior_leakage import run_leakage_safe_experiment


def test_pareto_sweep_returns_swept_points():
    rows = sweep_lambdas(lambdas=[0.0, 1.0], n_eval=8)
    assert len(rows) == 2
    for r in rows:
        assert 0.0 <= r["deal_rate"] <= 1.0
        assert r["avg_surplus"] is not None


def test_pareto_rows_pick_nondominated():
    rows = [
        {"lambda": 0.0, "deal_rate": 1.0, "avg_surplus": 1.0, "regret": 0.0},
        {"lambda": 2.0, "deal_rate": 0.8, "avg_surplus": 2.0, "regret": 0.0},
        {"lambda": 5.0, "deal_rate": 0.6, "avg_surplus": 1.5, "regret": 0.0},
    ]
    frontier = pareto_rows(rows)
    assert 1 <= len(frontier) <= 2


def test_baselines_evaluated():
    res = baseline_points(n_eval=5)
    assert "chotu" not in res  # baselines only
    for name in ("fixed", "concession", "tft", "random", "nash"):
        assert name in res


def test_leakage_safe_audit_runs():
    res = run_leakage_safe_experiment(n=12)
    for k in (
        "avg_surplus_early",
        "avg_surplus_late",
        "prior_error_first_half",
        "prior_error_second_half",
    ):
        assert k in res


def test_ablation_meta_runs():
    res = run_ablation(n=6)
    assert "A_no_learning" in res
    for k, v in res.items():
        assert 0.0 <= v["deal_rate"] <= 1.0


def test_calibration_bands():
    res = calibration_experiment(n=30)
    assert 0.0 <= res.brier_score <= 1.0
    assert len(res.reliability) >= 1


def test_frontier_benchmark_smoke():
    res = run_frontier_benchmark(n=6)
    assert res["chotu_frontier_size"] >= 0
    assert len(res["global_frontier"]) > 0
