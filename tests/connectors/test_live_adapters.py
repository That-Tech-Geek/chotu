"""Verify real request construction for each provider via httpx.MockTransport —
the adapters' actual HTTP code paths run; only the network is faked."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json

import httpx
import pytest

from economic_engine.connectors.providers import (
    EmailConnector,
    MockConnector,
    WhatsAppConnector,
)
from economic_engine.connectors.razorpay import RazorpayAdapter
from economic_engine.connectors.shopify import ShopifyAdapter
from economic_engine.state.canonical import Deal


def _mock_transport(captured: list, responder):
    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return responder(request)

    return httpx.MockTransport(handler)


class _TestTransportClient:
    """Swap httpx.post/get within a test using a MockTransport client."""

    def __init__(self, transport):
        self.client = httpx.Client(transport=transport)

    def __enter__(self):
        self.orig_post, self.orig_get = httpx.post, httpx.get
        httpx.post = self.client.post
        httpx.get = self.client.get
        return self

    def __exit__(self, *a):
        httpx.post, httpx.get = self.orig_post, self.orig_get


def test_razorpay_payment_link_live():
    captured = []
    with _TestTransportClient(_mock_transport(
        captured,
        lambda r: httpx.Response(200, json={"id": "plink_1", "status": "created"}),
    )):
        adapter = RazorpayAdapter(key_id="rzp_test_x", secret="sekrit")
        out = adapter.create_payment_link("neg-9", amount=123.45)
    assert out["live"] is True and out["id"] == "plink_1"
    req = captured[0]
    assert str(req.url) == "https://api.razorpay.com/v1/payment_links"
    body = json.loads(req.content)
    assert body["amount"] == 12345  # paise
    assert body["reference_id"] == "neg-9"
    expected = base64.b64encode(b"rzp_test_x:sekrit").decode()
    assert req.headers["Authorization"] == f"Basic {expected}"


def test_razorpay_offline_when_no_keys(monkeypatch):
    monkeypatch.delenv("RAZORPAY_KEY_ID", raising=False)
    monkeypatch.delenv("RAZORPAY_KEY_SECRET", raising=False)
    adapter = RazorpayAdapter()
    out = adapter.create_payment_link("n", 10.0)
    assert out["live"] is False


def test_razorpay_webhook_and_payment_signature():
    adapter = RazorpayAdapter(key_id="k", secret="s", webhook_secret="w")
    payload = b'{"event":"payment.captured"}'
    sig = hmac.new(b"w", payload, hashlib.sha256).hexdigest()
    assert adapter.verify_webhook(payload, sig)
    assert not adapter.verify_webhook(payload, "bad")
    psig = hmac.new(b"s", b"o1|p1", hashlib.sha256).hexdigest()
    assert adapter.verify_payment_signature("o1", "p1", psig)
    assert not adapter.verify_payment_signature("o1", "p2", psig)


def test_shopify_fetch_product_live():
    captured = []
    with _TestTransportClient(_mock_transport(
        captured,
        lambda r: httpx.Response(200, json={"products": [
            {"id": 7, "variants": [
                {"id": 70, "sku": "ABC", "price": "99.50", "inventory_quantity": 12},
            ]},
        ]}),
    )):
        adapter = ShopifyAdapter(shop="demo-store", access_token="tok")
        out = adapter.fetch_product("ABC")
    assert out["live"] is True and out["price"] == 99.50
    req = captured[0]
    assert req.url.host == "demo-store.myshopify.com"
    assert req.headers["X-Shopify-Access-Token"] == "tok"


def test_shopify_webhook_signature():
    adapter = ShopifyAdapter(secret="shpss_x")
    payload = b'{"id": 123}'
    sig = base64.b64encode(
        hmac.new(b"shpss_x", payload, hashlib.sha256).digest()
    ).decode()
    assert adapter.webhook_signature(payload, sig)
    assert not adapter.webhook_signature(payload, "tampered")


def test_email_connector_real_post():
    captured = []
    with _TestTransportClient(_mock_transport(
        captured, lambda r: httpx.Response(200, json={"ok": True}),
    )):
        conn = EmailConnector(webhook_url="https://relay.internal/mail")
        out = conn.send(Deal(price=95.0, quantity=100))
    assert out["success"] is True and out["status"] == 200
    body = json.loads(captured[0].content)
    assert body["channel"] == "email" and body["deal"]["price"] == 95.0


def test_whatsapp_connector_error_path():
    with _TestTransportClient(_mock_transport(
        [], lambda r: httpx.Response(500, json={"err": "boom"}),
    )):
        conn = WhatsAppConnector(webhook_url="https://relay.internal/wa")
        out = conn.send(Deal(price=1.0))
    assert out["success"] is False


def test_mock_connector_records_only():
    conn = MockConnector()
    out = conn.send(Deal(price=5.0))
    assert out["mock"] is True and len(conn.sent) == 1
