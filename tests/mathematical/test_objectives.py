import numpy as np

from economic_engine.models.cost_engine import CostEngine
from economic_engine.optimization.objectives import (
    cvar,
    fractional_kelly,
    value_of_information,
)
from economic_engine.state.canonical import CostComponents


def test_cvar_is_tail_mean():
    returns = np.linspace(-10, 10, 100)
    alpha = 0.95
    expected_tail = returns[: int((1 - alpha) * 100)].mean()
    assert cvar(returns, alpha=alpha) == expected_tail


def test_fractional_kelly_bounds():
    k = fractional_kelly(win_prob=0.6, win_return=1.0, fraction=0.25)
    assert 0.0 < k < 1.0


def test_voi_sign():
    assert value_of_information(5, 10, 1) > 0


def test_cost_engine_distribution():
    costs = CostComponents(purchase=100, freight_mean=10, freight_std=5)
    rng = np.random.default_rng(0)
    dist = CostEngine().landed_cost(costs, rng=rng)
    assert dist.p10 < dist.p25 < dist.p50 < dist.p75 < dist.p90 < dist.p95
    assert abs(dist.mean - (costs.purchase + costs.freight_mean)) < costs.freight_std
