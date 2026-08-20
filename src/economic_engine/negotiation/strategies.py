"""Initial strategy population + candidate action generator."""
from __future__ import annotations

import enum

import pydantic


class Strategy(str, enum.Enum):
    AGGRESSIVE_ANCHOR = "AGGRESSIVE_ANCHOR"
    SOFT_ANCHOR = "SOFT_ANCHOR"
    GRADUAL_CONCESSION = "GRADUAL_CONCESSION"
    QUANTITY_BUNDLE = "QUANTITY_BUNDLE"
    PAYMENT_TERM_SWAP = "PAYMENT_TERM_SWAP"
    DELIVERY_SWAP = "DELIVERY_SWAP"
    INFORMATION_SEEKING = "INFORMATION_SEEKING"
    DEADLINE_PRESSURE = "DEADLINE_PRESSURE"
    RELATIONSHIP_OFFER = "RELATIONSHIP_OFFER"
    WALKAWAY = "WALKAWAY"


class Action(str, enum.Enum):
    ACCEPT = "ACCEPT"
    COUNTER = "COUNTER"
    CHANGE_QUANTITY = "CHANGE_QUANTITY"
    CHANGE_PAYMENT_TERMS = "CHANGE_PAYMENT_TERMS"
    CHANGE_DELIVERY = "CHANGE_DELIVERY"
    BUNDLE = "BUNDLE"
    ASK_INFORMATION = "ASK_INFORMATION"
    WAIT = "WAIT"
    WALKAWAY = "WALKAWAY"


class Candidate(pydantic.BaseModel):
    action: Action
    strategy: Strategy
    price: float | None = None
    quantity: float | None = None
    payment_terms_days: int | None = None
    delivery_days: int | None = None
    note: str = ""


def generate_candidates(
    current_price: float,
    landed_mean: float,
    reservation_price: float | None,
    quantity: float,
) -> list[Candidate]:
    """Enumerate diverse candidate actions for the planner to score."""
    mid = current_price * 0.95
    floor = max(reservation_price or landed_mean, landed_mean * 0.5)
    gap = current_price - floor
    candidates = [
        Candidate(
            action=Action.ACCEPT,
            strategy=Strategy.SOFT_ANCHOR,
            price=current_price,
            note="accept the tabled offer",
        ),
        Candidate(
            action=Action.COUNTER,
            strategy=Strategy.GRADUAL_CONCESSION,
            price=max(floor + 0.05 * gap, floor),
            note="small step toward reservation",
        ),
        Candidate(
            action=Action.COUNTER,
            strategy=Strategy.AGGRESSIVE_ANCHOR,
            price=floor + 0.3 * gap,
            note="hard anchor",
        ),
        Candidate(
            action=Action.CHANGE_QUANTITY,
            strategy=Strategy.QUANTITY_BUNDLE,
            price=mid,
            quantity=quantity * 1.5,
            note="bundle quantity for price relief",
        ),
        Candidate(
            action=Action.CHANGE_PAYMENT_TERMS,
            strategy=Strategy.PAYMENT_TERM_SWAP,
            price=mid,
            payment_terms_days=30,
            note="swap terms for price",
        ),
        Candidate(
            action=Action.CHANGE_DELIVERY,
            strategy=Strategy.DELIVERY_SWAP,
            price=mid,
            delivery_days=14,
            note="swap delivery for price",
        ),
        Candidate(
            action=Action.ASK_INFORMATION,
            strategy=Strategy.INFORMATION_SEEKING,
            note="probe reservation / urgency",
        ),
        Candidate(
            action=Action.WAIT,
            strategy=Strategy.SOFT_ANCHOR,
            note="wait for better data",
        ),
        Candidate(
            action=Action.WALKAWAY,
            strategy=Strategy.WALKAWAY,
            note="exit if reservation violated",
        ),
    ]
    return candidates
