"""CVaR, fractional-Kelly and Value-of-Information — pure numpy."""
from __future__ import annotations

import math

import numpy as np


def cvar(returns: np.ndarray, alpha: float = 0.95) -> float:
    """Mean of the lower (1-alpha) tail. Protects against tail risk."""
    if returns.size == 0:
        return 0.0
    tail_fraction = max(int((1.0 - alpha) * returns.size), 1)
    worst = np.sort(returns)[:tail_fraction]
    return float(worst.mean())


def fractional_kelly(
    win_prob: float,
    win_return: float,
    loss_return: float = -1.0,
    fraction: float = 0.25,
) -> float:
    """Optimal aggressiveness: f* = argmax E[log(W + fR)]. Frac-Kelly dampens
    full-Kelly to a quarter (or given fraction)."""
    if win_prob <= 0.0 or win_prob >= 1.0:
        return 0.0
    w = abs(win_return)
    p, q = win_prob, 1.0 - win_prob
    b = w / abs(loss_return)
    kelly = max(p - q / b, 0.0)
    return float(kelly * fraction)


def value_of_information(
    current_ev: float,
    expected_ev_with_info: float,
    cost_of_asking: float,
) -> float:
    return float(expected_ev_with_info - current_ev - cost_of_asking)
