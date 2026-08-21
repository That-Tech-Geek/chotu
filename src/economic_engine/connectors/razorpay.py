"""Razorpay PaymentProvider adapter: real REST calls against
api.razorpay.com with basic-auth key_id/key_secret from env, plus HMAC
webhook verification. Falls back to offline mode when keys are absent."""
from __future__ import annotations

import hashlib
import hmac
import os

import httpx


class RazorpayAdapter:
    BASE_URL = "https://api.razorpay.com/v1"

    def __init__(
        self,
        key_id: str | None = None,
        secret: str | None = None,
        webhook_secret: str | None = None,
        timeout: float = 10.0,
    ):
        self.key_id = key_id if key_id is not None else os.environ.get("RAZORPAY_KEY_ID", "")
        self.secret = secret if secret is not None else os.environ.get("RAZORPAY_KEY_SECRET", "")
        self.webhook_secret = (
            webhook_secret
            if webhook_secret is not None
            else os.environ.get("RAZORPAY_WEBHOOK_SECRET", "")
        )
        self.timeout = timeout

    @property
    def live(self) -> bool:
        return bool(self.key_id and self.secret)

    def _auth(self) -> httpx.BasicAuth:
        return httpx.BasicAuth(self.key_id, self.secret)

    def create_order(self, amount_paise: int, currency: str = "INR",
                     receipt: str | None = None, notes: dict | None = None) -> dict:
        """POST /v1/orders — real Razorpay order."""
        if not self.live:
            return {"id": "offline", "status": "offline", "live": False}
        resp = httpx.post(
            f"{self.BASE_URL}/orders",
            auth=self._auth(),
            json={
                "amount": amount_paise,
                "currency": currency,
                "receipt": receipt,
                "notes": notes or {},
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {**resp.json(), "live": True}

    def create_payment_link(self, negotiation_id: str, amount: float,
                            currency: str = "INR",
                            action_id: str | None = None) -> dict:
        """POST /v1/payment_links — amount is in rupees; API wants paise.
        reference_id doubles as the provider-level idempotency key: prefer
        the Chotu action_id so the provider can de-duplicate retries."""
        if not self.live:
            return {"id": f"plv_{negotiation_id}", "amount": amount,
                    "status": "offline", "live": False}
        reference = action_id or negotiation_id
        resp = httpx.post(
            f"{self.BASE_URL}/payment_links",
            auth=self._auth(),
            json={
                "amount": int(round(amount * 100)),
                "currency": currency,
                "reference_id": reference,
                "description": f"negotiation:{negotiation_id}",
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {**resp.json(), "live": True}

    def fetch_payment(self, payment_id: str) -> dict:
        """GET /v1/payments/{id}."""
        if not self.live:
            return {"id": payment_id, "status": "offline", "live": False}
        resp = httpx.get(
            f"{self.BASE_URL}/payments/{payment_id}",
            auth=self._auth(),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return {**resp.json(), "live": True}

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        """Razorpay signs webhooks with HMAC-SHA256 over the raw body."""
        if not self.webhook_secret:
            return False
        expected = hmac.new(
            self.webhook_secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def verify_payment_signature(self, order_id: str, payment_id: str,
                                 signature: str) -> bool:
        """Frontend completion signature: HMAC-SHA256(order_id|payment_id)."""
        if not self.secret:
            return False
        body = f"{order_id}|{payment_id}"
        expected = hmac.new(
            self.secret.encode(), body.encode(), hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def webhook_events(self) -> list[str]:
        return ["payment.captured", "payment.failed", "order.paid"]
