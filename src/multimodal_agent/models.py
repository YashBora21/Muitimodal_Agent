from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


class Asset(TypedDict, total=False):
    id: str
    kind: str
    name: str
    content: str
    data: bytes
    duration: float


class State(TypedDict, total=False):
    request: str
    uploads: list[dict[str, Any]]
    messages: Annotated[list[dict[str, str]], operator.add]
    assets: Annotated[list[Asset], operator.add]
    current_asset_ids: list[str]
    extracted: str
    answer: str
    decision: str
    logs: list[str]
