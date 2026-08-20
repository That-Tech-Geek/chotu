"""pgvector (Supabase) retrieval over REST. Server-side ANN, no RAM."""
from __future__ import annotations

import httpx
import numpy as np


class PGVectorIndex:
    """Cosine retrieval against a Supabase `negotiation_embeddings` table
    via RPC `match_embeddings(query_embedding, match_count)`."""

    def __init__(self, url: str, key: str, timeout: float = 3.0):
        self.url = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
        }
        self.timeout = timeout

    def search(self, query: np.ndarray, k: int = 5) -> list[tuple[str, float]]:
        vec = np.asarray(query, dtype=np.float32).tolist()
        resp = httpx.post(
            f"{self.url}/rest/v1/rpc/match_embeddings",
            headers=self.headers,
            json={"query_embedding": vec, "match_count": k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        rows = resp.json()
        return [(r["id"], float(r.get("similarity", 0.0))) for r in rows]
