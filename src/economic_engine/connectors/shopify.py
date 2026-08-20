"""Shopify Commerce/Inventory adapter (mockable)."""
from __future__ import annotations

import hashlib
import hmac


class ShopifyAdapter:
    def __init__(self, shop: str = "", secret: str = ""):
        self.shop = shop
        self.secret = secret

    def fetch_product(self, sku: str) -> dict:
        return {"sku": sku, "price": 0, "inventory": 0}

    def fetch_order(self, order_id: str) -> dict:
        return {"order_id": order_id, "items": []}

    def webhook_signature(self, payload: bytes, signature: str) -> bool:
        if not self.secret:
            return False
        expected = hmac.new(
            self.secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def webhook_topics(self) -> list[str]:
        return ["inventory_items/update", "products/update"]
