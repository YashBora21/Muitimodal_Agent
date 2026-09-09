from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")
TRANSCRIBE_MODEL = os.getenv("GROQ_TRANSCRIBE_MODEL", "whisper-large-v3-turbo")
TESSDATA = Path(os.getenv("TESSDATA_PREFIX", r"C:\Program Files\Tesseract-OCR\tessdata"))

MAX_ITERATIONS = 8
TOOL_RESULT_LIMIT = 4000

YOUTUBE_RE = re.compile(
    r"https?://(?:www\.)?(?:youtube\.com|youtu\.be)/[^\s<>()\[\]\"']+",
    re.IGNORECASE,
)


def client() -> OpenAI:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set in .env")
    return OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
