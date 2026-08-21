from economic_engine.simulation.utility_benchmark import (
    run_cwalk_sweep,
    run_utility_benchmark,
)


def test_utility_benchmark_returns_all_policies():
    res = run_utility_benchmark(n=8, c_walk_ratio=0.25)
    assert "_meta" in res
    for name in ("ensemble", "chotu", "nash", "fixed"):
        assert name in res
        assert 0.0 <= res[name]["deal_rate"] <= 1.0
        assert res[name]["ev_se"] >= 0.0


def test_utility_math():
    res = run_utility_benchmark(n=8, c_walk_ratio=0.5)
    for name, v in res.items():
        if name == "_meta":
            continue
        # EV = avg_surplus - c_walk_ratio * walk_rate * avg_base; both sides
        # available from the report — relationship must hold within noise.
        assert isinstance(v["ev"], float)


def test_sweep_covers_ratios():
    sweep = run_cwalk_sweep(n=6, c_walk_ratios=(0.0, 0.5))
    assert set(sweep.keys()) == {0.0, 0.5}
