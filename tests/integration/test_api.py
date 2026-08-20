from fastapi.testclient import TestClient

from economic_engine.api.main import app
from economic_engine.state.canonical import (
    Merchant,
    Negotiation,
    NegotiationContext,
    Product,
    Supplier,
)


client = TestClient(app)


def ctx_dict() -> dict:
    c = NegotiationContext(
        merchant=Merchant(id="m", name="m"),
        supplier=Supplier(id="s", merchant_id="m", name="s"),
        product=Product(id="p", merchant_id="m", sku="p", base_purchase_cost=100),
        negotiation=Negotiation(
            id="n", merchant_id="m", supplier_id="s", product_id="p", quantity=10,
        ),
    )
    return c.model_dump(mode="json")


def test_lifecycle():
    resp = client.post(
        "/v1/negotiations",
        json={"merchant_id": "m", "supplier_id": "s",
              "product_id": "p", "quantity": 10, "message": "hi"},
    )
    assert resp.status_code == 200
    nid = resp.json()["negotiation_id"]
    resp = client.post(f"/v1/negotiations/{nid}/events", json={
        "message": "offer", "actor": "supplier",
    })
    assert resp.status_code == 200
    resp = client.post(f"/v1/negotiations/{nid}/decide", json={"context": ctx_dict()})
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] in {
        "ACCEPT", "COUNTER", "CHANGE_QUANTITY", "CHANGE_PAYMENT_TERMS",
        "CHANGE_DELIVERY", "BUNDLE", "ASK_INFORMATION", "WAIT", "WALKAWAY",
    }
    assert "strategy" in data and "reason_codes" in data


def test_404():
    resp = client.get("/v1/negotiations/does-not-exist")
    assert resp.status_code == 404
