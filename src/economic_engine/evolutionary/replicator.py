"""Replicator dynamics: dx_i = x_i (pi_i - pi_bar). Offline evolution."""
from __future__ import annotations

import numpy as np


class StrategyPopulation:
    def __init__(
        self,
        strategies: list[str],
        initial_weights: np.ndarray | None = None,
    ):
        self.strategies = strategies
        if initial_weights is None:
            self.weights = np.full(len(strategies), 1.0 / len(strategies))
        else:
            w = np.asarray(initial_weights, dtype=float)
            self.weights = w / w.sum()
        self.fitness = np.zeros(len(strategies))

    def record_fitness(
        self,
        strategy: str,
        profit: float,
        risk: float = 0.0,
        failure: float = 0.0,
        relationship_damage: float = 0.0,
        lifetime_value: float = 0.0,
    ) -> None:
        idx = self.strategies.index(strategy)
        self.fitness[idx] = (
            profit - risk - failure - relationship_damage + lifetime_value
        )

    def step(self, learning_rate: float = 0.1) -> None:
        pi = self.fitness
        bar = float((pi * self.weights).sum())
        delta = self.weights * (pi - bar)
        self.weights = np.clip(self.weights + learning_rate * delta, 1e-4, None)
        self.weights /= self.weights.sum()

    def sample(self, rng: np.random.Generator) -> str:
        return self.strategies[rng.choice(len(self.strategies), p=self.weights)]
