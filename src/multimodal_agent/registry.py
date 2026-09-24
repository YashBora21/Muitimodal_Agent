import re
from typing import Any

from .models import Asset

ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)


def compact(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    section = limit // 3
    middle = len(content) // 2
    return "\n\n".join(
        [
            content[:section],
            "[...content shortened...]",
            content[middle - section // 2:middle + section // 2],
            "[...content shortened...]",
            content[-section:],
        ]
    )


def asset_index(
    assets: list[Asset],
    current_ids: list[str] | None = None,
) -> str:
    if not assets:
        return "(no assets)"
    current_ids = current_ids or []
    youtube_assets = [asset for asset in assets if asset["kind"] == "youtube"]
    lines = []
    for asset in assets:
        extra = " [current upload]" if asset["id"] in current_ids else ""
        if asset["kind"] == "youtube":
            extra += f" [youtube #{youtube_assets.index(asset) + 1}]"
        lines.append(f"- {asset['id']}: {asset['kind']}{extra} | {asset['name']}")
    return "\n".join(lines)


def referenced_youtube_ids(request: str, assets: list[Asset]) -> list[str]:
    normalized = request.lower()
    if "youtube" not in normalized or not any(
        word in normalized for word in ("video", "url", "link")
    ):
        return []

    youtube_assets = [asset for asset in assets if asset["kind"] == "youtube"]
    positions = [int(match.group(1)) for match in ORDINAL_RE.finditer(request)]
    return list(
        dict.fromkeys(
            youtube_assets[position - 1]["id"]
            for position in positions
            if 1 <= position <= len(youtube_assets)
        )
    )


def new_asset(
    assets: list[Asset],
    kind: str,
    name: str,
    content: str,
    **extra: Any,
) -> Asset:
    number = sum(asset["kind"] == kind for asset in assets) + 1
    return {
        "id": f"{kind}-{number}",
        "kind": kind,
        "name": name,
        "content": content,
        **extra,
    }
