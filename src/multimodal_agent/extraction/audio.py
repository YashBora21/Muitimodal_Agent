import io
import subprocess
import tempfile
import wave
from pathlib import Path

from ..config import TRANSCRIBE_MODEL, client


def audio_duration(data: bytes, suffix: str) -> float | None:
    if suffix.lower() == ".wav":
        try:
            with wave.open(io.BytesIO(data)) as audio:
                return round(audio.getnframes() / audio.getframerate(), 2)
        except (wave.Error, ZeroDivisionError):
            return None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as file:
            file.write(data)
            file.flush()
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "csv=p=0",
                    file.name,
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
        return round(float(result.stdout), 2)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def transcribe(data: bytes, name: str) -> tuple[str, float | None]:
    file = io.BytesIO(data)
    file.name = name
    response = client().audio.transcriptions.create(
        model=TRANSCRIBE_MODEL,
        file=file,
    )
    return response.text.strip(), audio_duration(data, Path(name).suffix)
