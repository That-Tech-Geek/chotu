"""Calibration: how well predicted acceptance probabilities map to observed
outcomes (Brier score, reliability by decile). This catches
P(pred|θ)(u posterior-response) being honest, not just point-estimates."""
from __future__ import annotations

import numpy as np

from economic_engine.negotiation.opponent import OpponentLatent, OpponentState


class CalibrationReport:
    def __init__(self, brier_score: float, reliability: list[tuple]):
        self.brier_score = brier_score
        self.reliability = reliability

    def table(self) -> str:
        lines = [f"brier_score: {self.brier_score:.4f}"]
        for decile, p_mean, obs_rate in self.reliability:
            lines.append(
                f"  [{decile:.0%}, {decile+0.1:.0%}): "
                f"pred={p_mean:.3f}, observed={obs_rate:.3f}",
            )
        return "\n".join(lines)


def measure_decision_calibration(
    opponent: OpponentState,
    predicted_acceptances: list[float],
    outcomes: list[bool],
    n_bins: int = 10,
) -> CalibrationReport:
    p = np.asarray(predicted_acceptances)
    y = np.asarray(outcomes).astype(float)
    brier = float(np.mean((p - y) ** 2))
    bins = (p * n_bins).astype(int).clip(0, n_bins - 1)
    reliability = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum():
            reliability.append((b / n_bins, float(p[mask].mean()), float(y[mask].mean())))
    return CalibrationReport(brier_score=brier, reliability=reliability)


def calibration_experiment(n: int = 30, seed: int = 0) -> CalibrationReport:
    """Run OpponentState updates vs a synthetic latent model and measure the
    Brier/reliability. We sample a known respondent, update the belief each
    round, and score how well belief predicts the observed accept."""
    rng = np.random.default_rng(seed)
    true_reservation = float(rng.uniform(60, 140))
    true_patience = float(rng.uniform(0.2, 0.9))
    opponent = OpponentState(
        OpponentLatent(
            reservation_price=true_reservation,
            reservation_std=20.0,
            patience=true_patience,
        )
    )
    predicted = []
    outcomes = []
    for i in range(n):
        base_lines = float(rng.uniform(80, 120))
        # sample plausibly above reservation, then adjust
        offer_price = base_lines * 0.8
        eff_res_discount = 0.97 - i * 0.002
        p_accept = opponent.response_probability(offer_price * eff_res_discount)
        accepted = rng.random() < p_accept
        predicted.append(float(p_accept))
        outcomes.append(bool(accepted))
        opponent.update_from_round(price=offer_price, accepted=bool(accepted))
    return measure_decision_calibration(opponent, predicted, outcomes)
