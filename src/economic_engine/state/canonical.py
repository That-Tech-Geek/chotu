"""Canonical state models. Every external system (Shopify, Razorpay, ERP,
3PL) is normalized into these - the engine never depends on provider schemas."""
from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Optional

import pydantic


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Merchant(pydantic.BaseModel):
    id: str
    name: str
    margin_floor: float = 0.0
    risk_tolerance: float = 0.5


class Supplier(pydantic.BaseModel):
    id: str
    merchant_id: str
    name: str
    reliability_history: float = 0.5
    metadata: dict = {}


class Product(pydantic.BaseModel):
    id: str
    merchant_id: str
    sku: str
    base_purchase_cost: float
    category: Optional[str] = None


class Inventory(pydantic.BaseModel):
    product_id: str
    on_hand: float
    reserved: float = 0.0

    @property
    def available(self) -> float:
        return max(self.on_hand - self.reserved, 0.0)


class Order(pydantic.BaseModel):
    id: str
    product_id: str
    quantity: float
    unit_price: float


class Demand(pydantic.BaseModel):
    product_id: str
    mean: float
    std: float = 0.0


class CostComponents(pydantic.BaseModel):
    """Component means/stdevs feeding the landed-cost distribution."""
    purchase: float
    freight_mean: float = 0.0
    freight_std: float = 0.0
    handling_mean: float = 0.0
    handling_std: float = 0.0
    inventory_mean: float = 0.0
    inventory_std: float = 0.0
    financing_mean: float = 0.0
    financing_std: float = 0.0
    failure_mean: float = 0.0
    failure_std: float = 0.0
    delay_mean: float = 0.0
    delay_std: float = 0.0


class Logistics(pydantic.BaseModel):
    lead_time_days: float = 0.0
    lead_time_std: float = 0.0
    carrier: Optional[str] = None


class Offer(pydantic.BaseModel):
    price: Optional[float] = None
    quantity: Optional[float] = None
    payment_terms_days: Optional[int] = None
    delivery_days: Optional[int] = None
    bundle: list[str] = []
    actor: str = "merchant"


class Round(pydantic.BaseModel):
    index: int
    offer: Offer
    response: Optional[str] = None
    at: datetime = pydantic.Field(default_factory=utcnow)


class NegotiationStatus(str, enum.Enum):
    OPEN = "OPEN"
    AGREED = "AGREED"
    WALKED_AWAY = "WALKED_AWAY"
    CLOSED = "CLOSED"


class Negotiation(pydantic.BaseModel):
    id: str
    merchant_id: str
    supplier_id: str
    product_id: str
    quantity: float
    status: NegotiationStatus = NegotiationStatus.OPEN
    rounds: list[Round] = []
    agreed_price: Optional[float] = None
    reservation_price: Optional[float] = None
    deadline: Optional[datetime] = None
    created_at: datetime = pydantic.Field(default_factory=utcnow)


class Relationship(pydantic.BaseModel):
    supplier_id: str
    merchant_id: str
    lifetime_value: float = 0.0
    interaction_count: int = 0
    reputation: float = 0.5


class TextSignals(pydantic.BaseModel):
    sentiment: float = 0.0
    urgency: float = 0.0
    price_resistance: float = 0.0
    concession_willingness: float = 0.0
    deadline_signal: float = 0.0
    uncertainty: float = 0.0
    finality: float = 0.0
    relationship_signal: float = 0.0


class Constraints(pydantic.BaseModel):
    max_transaction_limit: Optional[float] = None
    max_discount_pct: float = 1.0
    max_offer: Optional[float] = None
    requires_approval_above: Optional[float] = None
    data_freshness_seconds: int = 3600


class NegotiationContext(pydantic.BaseModel):
    """Everything the decision engine needs for one decision call."""
    merchant: Merchant
    supplier: Supplier
    product: Product
    negotiation: Negotiation
    inventory: Optional[Inventory] = None
    demand: Optional[Demand] = None
    costs: Optional[CostComponents] = None
    logistics: Optional[Logistics] = None
    relationship: Optional[Relationship] = None
    text_signals: Optional[TextSignals] = None
    constraints: Constraints = pydantic.Field(default_factory=Constraints)
