import numpy as np

from economic_engine.models.supplier_model import SupplierPosterior
from economic_engine.state.canonical import (
    Constraints,
    Inventory,
    Merchant,
    Negotiation,
    NegotiationContext,
    Product,
    Supplier,
    TextSignals,
)
from economic_engine.text.embedder import HashingEmbedder
from economic_engine.text.extractor import extract


def make_ctx() -> NegotiationContext:
    return NegotiationContext(
        merchant=Merchant(id="m", name="m"),
        supplier=Supplier(id="s", merchant_id="m", name="s"),
        product=Product(id="p", merchant_id="m", sku="p", base_purchase_cost=100),
        negotiation=Negotiation(
            id="n", merchant_id="m", supplier_id="s", product_id="p", quantity=10,
        ),
        constraints=Constraints(),
    )


def test_extractor_signals():
    sig = extract("that price is too high, maybe we can work with you, urgent")
    assert sig.price_resistance > 0
    assert sig.concession_willingness > 0
    assert sig.urgency > 0


def test_embedder_determinism():
    e = HashingEmbedder(dim=32)
    v1 = e.embed("hello world hello world")
    v2 = e.embed("hello world hello world")
    assert np.allclose(v1, v2)


def test_supplier_posterior_updates():
    p = SupplierPosterior()
    before = p.reservation_mean
    p.update_from_round(
        round_index=0, proposed_price=100, response="ACCEPT",
    )
    assert p.reservation_mean != before


def test_inventory_available():
    inv = Inventory(product_id="p", on_hand=10, reserved=2)
    assert inv.available == 8
