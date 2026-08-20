"""Lexicon-based signal extraction. Deterministic, no LLM."""
from __future__ import annotations

import re
from typing import Iterable

from economic_engine.state.canonical import TextSignals

_WORD_LISTS = {
    "urgency": [
        "urgent", "asap", "immediately", "deadline", "hurry", "quick",
        "time-sensitive", "expedite",
    ],
    "concession_willingness": [
        "discount", "flexible", "concede", "negotiable", "work with you",
        "better deal", "meet in the middle",
    ],
    "price_resistance": [
        "too high", "expensive", "over budget", "can't afford", "pricey",
        "reduce the price", "cheaper",
    ],
    "finality": [
        "final", "take it or leave it", "best and final", "last offer",
        "non-negotiable", "no more",
    ],
    "relationship_signal": [
        "long-term", "partnership", "trust", "relationship", "reliable",
        "together", "collaborate",
    ],
    "uncertainty": [
        "maybe", "uncertain", "depends", "tentative", "if", "possibly",
        "not sure",
    ],
    "deadline_signal": [
        "by friday", "end of week", "eod", "eow", "next monday",
        "this quarter", "deadline",
    ],
}
_POS = re.compile(
    r"\b(good|great|happy|please|thanks|thank you|excellent|agree|deal|"
    r"accept|perfect|love)\b", re.I,
)
_NEG = re.compile(
    r"\b(bad|unhappy|problem|issue|dispute|worst|reject|refuse|fail|hate|"
    r"annoy)\b", re.I,
)


def extract(message: str | Iterable[str]) -> TextSignals:
    text = message if isinstance(message, str) else " ".join(message)
    lower = text.lower()
    hits = {
        key: sum(1 for w in words if w in lower)
        for key, words in _WORD_LISTS.items()
    }
    pos = len(_POS.findall(lower))
    neg = len(_NEG.findall(lower))
    sentiment = 0.0
    if pos + neg:
        sentiment = (pos - neg) / (pos + neg)
    sig = TextSignals(sentiment=sentiment)
    for key, count in hits.items():
        setattr(sig, key, min(count / 3.0, 1.0))
    return sig
