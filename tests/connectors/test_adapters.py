from economic_engine.connectors.razorpay import RazorpayAdapter
from economic_engine.connectors.shopify import ShopifyAdapter


def test_razorpay_webhook_verify_and_topics():
    adapter = RazorpayAdapter(webhook_secret="s")
    assert adapter.webhook_events()
    assert not adapter.verify_webhook(b"{}", "bad")


def test_shopify_webhook_topics():
    adapter = ShopifyAdapter(secret="s")
    assert "products/update" in adapter.webhook_topics()
