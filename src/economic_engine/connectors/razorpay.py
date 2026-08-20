"""Razorpay PaymentProvider adapter (mockable, no hard dep on razorpay sdk)."""
from __future__ import annotations

import hashlib
import hmac


class RazorpayAdapter:
    def __init__(self, key_id: str = "", secret: str = "", webhook_secret: str = ""):
        self.key_id = key_id
        self.secret = secret
        self.webhook_secret = webhook_secret

    def create_payment_link(self, negotiation_id: str, amount: float) -> dict:
        return {"id": f"plv_{negotiation_id}", "amount": amount, "status": "created"}

    def fetch_payment(self, payment_id: str) -> dict:
        return {"id": payment_id, "status": "captured", "amount": 0}

    def verify_webhook(self, payload: bytes, signature: str) -> bool:
        if not self.webhook_secret:
            return False
        expected = hmac.new(
            self.webhook_secret.encode(), payload, hashlib.sha256
        ).hexdigest()
        return hmac.compare_digest(expected, signature)

    def webhook_events(self) -> list[str]:
        return ["payment.captured", "payment.failed", "order.paid"]
