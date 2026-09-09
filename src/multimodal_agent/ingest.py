import mimetypes
from pathlib import Path
from typing import Any

from .extraction.audio import transcribe
from .extraction.image import vision
from .extraction.pdf import read_pdf
from .models import Asset
from .registry import new_asset


def ingest_upload(
    upload: dict[str, Any],
    existing: list[Asset],
) -> list[Asset]:
    name = upload["name"]
    data = upload["data"]
    mime = (
        upload.get("mime")
        or mimetypes.guess_type(name)[0]
        or "application/octet-stream"
    )

    if mime.startswith("image/"):
        return [new_asset(existing, "image", name, vision(data, mime))]

    if mime == "application/pdf" or name.lower().endswith(".pdf"):
        return read_pdf(data, name, existing)

    audio_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".webm"}
    if mime.startswith("audio/") or Path(name).suffix.lower() in audio_extensions:
        content, duration = transcribe(data, name)
        return [
            new_asset(
                existing,
                "audio",
                name,
                content,
                duration=duration,
            )
        ]

    raise ValueError(f"Unsupported file: {name}")
