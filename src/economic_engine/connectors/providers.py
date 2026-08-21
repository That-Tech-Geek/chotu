"""Provider protocol interfaces. Wire Shopify/Razorpay/ERP/3PL adapters
behind these contracts so the engine never depends on provider schemas."""
from __future__ import annotations

import os
from typing import Protocol

import httpx


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
    """POSTs the deal to a configured outbound webhook (SendGrid / SES relay /
    internal mailer). Real HTTP side effect."""

    def __init__(self, webhook_url: str | None = None,
                 auth_header: str | None = None, timeout: float = 10.0):
        self.webhook_url = webhook_url or os.environ.get("EMAIL_WEBHOOK_URL", "")
        self.auth_header = auth_header or os.environ.get("EMAIL_WEBHOOK_AUTH", "")
        self.timeout = timeout

    def send(self, deal: "Deal") -> dict:
        if not self.webhook_url:
            return {"success": False, "error": "webhook_url not configured"}
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        try:
            resp = httpx.post(
                self.webhook_url,
                headers=headers,
                json={"channel": "email", "deal": deal.model_dump(mode="json")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return {"success": True, "status": resp.status_code, "action": "email"}
        except httpx.HTTPError as e:
            return {"success": False, "error": str(e), "action": "email"}


class WhatsAppConnector:
    """POSTs the deal to a configured WhatsApp Business Cloud API endpoint
    (or a relay webhook). Real HTTP side effect."""

    def __init__(self, webhook_url: str | None = None,
                 auth_header: str | None = None, timeout: float = 10.0):
        self.webhook_url = webhook_url or os.environ.get("WHATSAPP_WEBHOOK_URL", "")
        self.auth_header = auth_header or os.environ.get("WHATSAPP_WEBHOOK_AUTH", "")
        self.timeout = timeout

    def send(self, deal: "Deal") -> dict:
        if not self.webhook_url:
            return {"success": False, "error": "webhook_url not configured"}
        headers = {"Content-Type": "application/json"}
        if self.auth_header:
            headers["Authorization"] = self.auth_header
        try:
            resp = httpx.post(
                self.webhook_url,
                headers=headers,
                json={"channel": "whatsapp", "deal": deal.model_dump(mode="json")},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            return {"success": True, "status": resp.status_code, "action": "whatsapp"}
        except httpx.HTTPError as e:
            return {"success": False, "error": str(e), "action": "whatsapp"}


class MockConnector:
    """Test-only: records sends, never leaves the process."""

    def __init__(self):
        self.sent: list[dict] = []

    def send(self, deal: "Deal") -> dict:
        self.sent.append(deal.model_dump(mode="json"))
        return {"success": True, "mock": True}
