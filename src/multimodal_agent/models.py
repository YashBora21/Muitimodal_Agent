from __future__ import annotations

from typing import TypedDict


class Asset(TypedDict, total=False):
    id: str
    db_id: str
    kind: str
    name: str
    content: str
    data: bytes
    duration: float
