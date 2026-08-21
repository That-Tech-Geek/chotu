"""Shopify Commerce/Inventory adapter: real Admin REST calls against
{shop}.myshopify.com/admin/api/2024-10 with an X-Shopify-Access-Token from
env, plus HMAC-SHA256 base64 webhook verification. Offline when unconfigured."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os

import httpx


class ShopifyAdapter:
    API_VERSION = "2024-10"

    def __init__(
        self,
        shop: str | None = None,
        access_token: str | None = None,
        secret: str | None = None,
        timeout: float = 10.0,
    ):
        self.shop = (
            shop if shop is not None else os.environ.get("SHOPIFY_SHOP", "")
        )
        self.access_token = (
            access_token
            if access_token is not None
            else os.environ.get("SHOPIFY_ACCESS_TOKEN", "")
        )
        self.secret = (
            secret if secret is not None else os.environ.get("SHOPIFY_SECRET", "")
        )
        self.timeout = timeout

    @property
    def live(self) -> bool:
        return bool(self.shop and self.access_token)

    def _base(self) -> str:
        return f"https://{self.shop}.myshopify.com/admin/api/{self.API_VERSION}"

    def _headers(self) -> dict:
        return {"X-Shopify-Access-Token": self.access_token,
                "Content-Type": "application/json"}

    def fetch_product(self, sku: str) -> dict:
        """GET /products.json — finds first variant matching sku."""
        if not self.live:
            return {"sku": sku, "price": 0, "inventory": 0, "live": False}
        resp = httpx.get(
            f"{self._base()}/products.json",
            headers=self._headers(),
            params={"fields": "id,title,variants", "limit": 250},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        for product in resp.json().get("products", []):
            for variant in product.get("variants", []):
                if variant.get("sku") == sku:
                    return {
                        "sku": sku,
                        "product_id": product["id"],
                        "variant_id": variant["id"],
                        "price": float(variant.get("price", 0)),
                        "inventory": variant.get("inventory_quantity", 0),
                        "live": True,
                    }
        return {"sku": sku, "found": False, "live": True}

    def fetch_order(self, order_id: str) -> dict:
        """GET /orders/{id}.json."""
        if not self.live:
            return {"order_id": order_id, "items": [], "live": False}
        resp = httpx.get(
            f"{self._base()}/orders/{order_id}.json",
            headers=self._headers(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        order = resp.json().get("order", {})
        return {**order, "live": True}

    def graphql(self, query: str, variables: dict | None = None) -> dict:
        """POST /graphql.json — Admin GraphQL API (customer-data queries)."""
        if not self.live:
            return {"data": None, "live": False}
        resp = httpx.post(
            f"{self._base()}/graphql.json",
            headers=self._headers(),
            json={"query": query, "variables": variables or {}},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {**resp.json(), "live": True}

    def webhook_signature(self, payload: bytes, signature: str) -> bool:
        """Shopify sends HMAC-SHA256 in base64 over the raw body."""
        if not self.secret:
            return False
        expected = base64.b64encode(
            hmac.new(self.secret.encode(), payload, hashlib.sha256).digest()
        ).decode()
        return hmac.compare_digest(expected, signature)

    def webhook_topics(self) -> list[str]:
        return ["inventory_items/update", "products/update", "orders/create"]
