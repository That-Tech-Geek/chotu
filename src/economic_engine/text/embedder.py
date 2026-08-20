"""Deterministic hashing embedder. No model download, no RAM bloat, so a
Vercel bundle stays small. Cosine-comparable via bag-of-ngram hashing."""
from __future__ import annotations

import hashlib
import re
import unicodedata

import numpy as np

_TOK = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, dim: int = 128, ngrams: int = 2):
        self.dim = dim
        self.ngrams = ngrams

    def _tokens(self, text: str) -> list[str]:
        norm = unicodedata.normalize("NFKD", text.lower())
        return _TOK.findall(norm)

    def embed(self, text: str) -> np.ndarray:
        toks = self._tokens(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for i, tok in enumerate(toks):
            feats = [tok]
            for n in range(2, self.ngrams + 1):
                if i + n <= len(toks):
                    feats.append(" ".join(toks[i:i + n]))
            for f in feats:
                h = int.from_bytes(hashlib.md5(f.encode()).digest()[:8], "big")
                vec[h % self.dim] += 1.0
        norm = float(np.linalg.norm(vec))
        if norm:
            vec /= norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.embed(t) for t in texts])
