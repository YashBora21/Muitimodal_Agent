import logging
import mimetypes
from pathlib import Path
from typing import Any

from .extraction.audio import transcribe
from .extraction.image import vision
from .extraction.pdf import read_pdf
from .models import Asset
from .registry import new_asset

logger = logging.getLogger(__name__)


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
    logger.info("Extraction | start name=%s mime=%s bytes=%s", name, mime, len(data))

    if mime.startswith("image/"):
        assets = [new_asset(existing, "image", name, vision(data, mime))]
        logger.info("Extraction | complete kind=image assets=%s chars=%s", len(assets), len(assets[0]["content"]))
        return assets

    if mime == "application/pdf" or name.lower().endswith(".pdf"):
        assets = read_pdf(data, name, existing)
        logger.info("Extraction | complete kind=pdf assets=%s chars=%s", len(assets), sum(len(asset.get("content", "")) for asset in assets))
        return assets

    audio_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".mpeg", ".webm"}
    if mime.startswith("audio/") or Path(name).suffix.lower() in audio_extensions:
        content, duration = transcribe(data, name)
        assets = [
            new_asset(
                existing,
                "audio",
                name,
                content,
                duration=duration,
            )
        ]
        logger.info("Extraction | complete kind=audio assets=%s chars=%s", len(assets), len(content))
        return assets

    logger.warning("Extraction | unsupported name=%s mime=%s", name, mime)
    raise ValueError(f"Unsupported file: {name}")
