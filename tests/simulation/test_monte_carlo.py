import numpy as np

from economic_engine.negotiation.strategies import Candidate, Action, Strategy
from economic_engine.simulation.monte_carlo import MonteCarloSimulator
from economic_engine.state.canonical import (
    Merchant,
    Negotiation,
    NegotiationContext,
    Product,
    Supplier,
)


def ctx():
    return NegotiationContext(
        merchant=Merchant(id="m", name="m"),
        supplier=Supplier(id="s", merchant_id="m", name="s"),
        product=Product(id="p", merchant_id="m", sku="p", base_purchase_cost=100),
        negotiation=Negotiation(
            id="n", merchant_id="m", supplier_id="s", product_id="p", quantity=10,
        ),
    )


def test_simulator_returns_samples():
    sim = MonteCarloSimulator(mode="FAST", seed=42)
    cand = Candidate(action=Action.COUNTER, strategy=Strategy.SOFT_ANCHOR, price=95)
    out = sim.simulate(ctx(), cand)
    assert out.shape == (4096,)
    assert np.isfinite(out).all()
