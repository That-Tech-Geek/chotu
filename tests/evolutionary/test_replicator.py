import numpy as np

from economic_engine.evolutionary.replicator import StrategyPopulation


def test_replicator_weights_sum_norm():
    strategies = ["A", "B"]
    pop = StrategyPopulation(strategies)
    pop.record_fitness("A", profit=10)
    pop.record_fitness("B", profit=1)
    pop.step()
    assert abs(pop.weights.sum() - 1.0) < 1e-6


def test_sampling_returns_strategy():
    pop = StrategyPopulation(["A"])
    rng = np.random.default_rng(0)
    pop.record_fitness("A", profit=0)
    assert pop.sample(rng) in pop.strategies
