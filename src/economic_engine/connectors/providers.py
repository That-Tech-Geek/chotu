"""Provider protocol interfaces. Wire Shopify/Razorpay/ERP/3PL adapters
behind these contracts so the engine never depends on provider schemas."""
from __future__ import annotations

from typing import Protocol


class PaymentProvider(Protocol):
    def create_payment_link(self, negotiation_id: str, amount: float) -> dict: ...
    def fetch_payment(self, payment_id: str) -> dict: ...
    def verify_webhook(self, payload: bytes, signature: str) -> bool: ...


class CommerceProvider(Protocol):
    def fetch_product(self, sku: str) -> dict: ...
    def fetch_order(self, order_id: str) -> dict: ...


class InventoryProvider(Protocol):
    def fetch_inventory(self, product_id: str) -> dict: ...


class LogisticsProvider(Protocol):
    def create_shipment(self, offer: dict) -> dict: ...
    def track_shipment(self, tracking_id: str) -> dict: ...


class ActionConnector(Protocol):
    """Side-effect connector: where 'execute a negotiation action' becomes
    a real email/WhatsApp/API call. Implementations are infra-only; the brain
    never calls them directly — only the runtime does."""

    def send(self, deal: "Deal") -> dict: ...


class EmailConnector:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    def send(self, deal: "Deal") -> dict:
        if not self.webhook_url:
            return {"success": False, "error": "webhook_url not configured"}
        return {"success": True, "webhook_url": self.webhook_url,
                "action": "email"}


class WhatsAppConnector:
    def __init__(self, webhook_url: str | None = None):
        self.webhook_url = webhook_url

    def send(self, deal: "Deal") -> dict:
        if not self.webhook_url:
            return {"success": False, "error": "webhook_url not configured"}
        return {"success": True, "webhook_url": self.webhook_url,
                "action": "whatsapp"}
