"""Dataset ingestion: CSV/API -> schema detection -> column mapping ->
validation -> canonical -> training/eval. Mirror the plan's pipeline."""
from __future__ import annotations

import csv
import io


def load_csv_bytes(data: bytes) -> list[dict]:
    text = data.decode()
    reader = csv.DictReader(io.StringIO(text))
    return list(reader)


def map_to_canonical(rows: list[dict], mapping: dict[str, str]) -> list[dict]:
    canonical = []
    for row in rows:
        canon = {dst: row.get(src) for dst, src in mapping.items()}
        canonical.append(canon)
    return canonical
