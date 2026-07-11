from functools import lru_cache
from faster_whisper import WhisperModel
from app.config import settings


@lru_cache
def _get_model() -> WhisperModel:
    return WhisperModel(
        settings.whisper_model_size,
        device="cpu",        # ← force CPU
        compute_type="int8"  # ← int8 is fine on CPU
    )


def transcribe_audio(path: str) -> tuple[str, float]:
    model = _get_model()
    segments, info = model.transcribe(path)
    text = " ".join(segment.text.strip() for segment in segments)
    return text.strip(), info.duration