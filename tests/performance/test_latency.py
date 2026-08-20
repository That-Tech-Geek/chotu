import time

from economic_engine.negotiation.engine import NegotiationEngine
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


def test_decide_latency():
    engine = NegotiationEngine(mc_mode="FAST")
    start = time.perf_counter()
    engine.decide(ctx())
    latency_ms = (time.perf_counter() - start) * 1000
    # hot-path budget: ~50ms P50 target, here asserted within engineering margin
    assert latency_ms < 500
