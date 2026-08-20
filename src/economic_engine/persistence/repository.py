"""Supabase REST-backed repository + in-memory fallback for tests/dev."""
from __future__ import annotations

import httpx


class Repository:
    def upsert(self, table: str, rows: list[dict]):
        raise NotImplementedError

    def query(self, table: str, filters: dict | None = None) -> list[dict]:
        raise NotImplementedError


class InMemoryRepository(Repository):
    def __init__(self):
        self.tables: dict[str, list[dict]] = {}

    def upsert(self, table: str, rows: list[dict]):
        self.tables.setdefault(table, []).extend(rows)

    def query(self, table: str, filters: dict | None = None) -> list[dict]:
        rows = self.tables.get(table, [])
        if not filters:
            return rows
        return [
            row
            for row in rows
            if all(filter_value(row, k, v) for k, v in filters.items())
        ]


class SupabaseRepository(Repository):
    def __init__(self, url: str, key: str, timeout: float = 5.0):
        self.url = url.rstrip("/")
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }
        self.timeout = timeout

    def upsert(self, table: str, rows: list[dict]):
        resp = httpx.post(
            f"{self.url}/rest/v1/{table}",
            headers={**self.headers, "Prefer": "return=minimal"},
            json=rows,
            timeout=self.timeout,
        )
        resp.raise_for_status()

    def query(self, table: str, filters: dict | None = None) -> list[dict]:
        params = {k: f"eq.{v}" for k, v in (filters or {}).items()}
        resp = httpx.get(
            f"{self.url}/rest/v1/{table}",
            headers=self.headers,
            params=params,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()


def filter_value(row: dict, key: str, value) -> bool:
    return str(row.get(key)) == str(value)
