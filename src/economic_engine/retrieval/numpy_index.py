"""In-memory cosine retrieval for small corpora. RAM-safe for Vercel."""
from __future__ import annotations

import numpy as np


class NumpyCosineIndex:
    def __init__(self, max_items: int = 50_000):
        self.max_items = max_items
        self._vectors: list[np.ndarray] = []
        self._ids: list[str] = []
        self._mat: np.ndarray | None = None

    def add(self, vectors: np.ndarray, ids: list[str]) -> None:
        vecs = np.atleast_2d(np.asarray(vectors, dtype=np.float32))
        if len(self._ids) + len(ids) > self.max_items:
            raise ValueError("index full")
        self._ids.extend(ids)
        self._vectors.extend(list(vecs))
        self._mat = np.stack(self._vectors)

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        if self._mat is None or not self._ids:
            return []
        q = np.asarray(query, dtype=np.float32)
        sims = (self._mat @ q) / (np.linalg.norm(self._mat, axis=1) *
                                  np.linalg.norm(q) + 1e-9)
        k = min(k, len(self._ids))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        return [(self._ids[i], float(sims[i])) for i in idx]

    def __len__(self) -> int:
        return len(self._ids)
